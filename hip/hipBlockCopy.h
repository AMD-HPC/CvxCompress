// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIP_BLOCK_COPY_H
#define HIP_BLOCK_COPY_H

#include <hip/hip_runtime.h>
#include <rocprim/block/block_reduce.hpp>

using bcopy_float4_vec = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using bcopy_int4_vec   = __attribute__((__vector_size__(4 * sizeof(int)))) int;

static constexpr int BCOPY_ZPB = 8;

// ---------------------------------------------------------------------------
// CopyTo: grid → wavelet layout.
//
// Grid: (wnx/32, wny/32, wnz/ZPB).  Requires ex,ey,ez >= 32.
//
// Copies ex*ey*ez samples starting at (x0,y0,z0) in the source grid.
// Positions beyond (ex,ey,ez) are zero-filled.
// RMS is computed over the extraction window only.
// ---------------------------------------------------------------------------
template<bool DO_COPY, bool COMPUTE_RMS>
__launch_bounds__(256)
__global__ void copyToWaveletKernelOpt(
    const float* __restrict__ d_src,
    int ldimx, int ldimxy,
    int x0, int y0, int z0,
    int ex, int ey, int ez,
    float* __restrict__ d_dst,
    int wnx, int wny, int wnz,
    double* __restrict__ d_partial_sums)
{
    constexpr int SLC = 2;

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    int tile_x0 = blockIdx.x * 32;
    int tile_y0 = blockIdx.y * 32;
    int gx = tile_x0 + xg * 4;
    int gy = tile_y0 + yr;
    int z_start = blockIdx.z * BCOPY_ZPB;

    bool y_in_range = (gy < ey);

    uint32_t src_row_byte = ((y0 + gy) * ldimx + x0 + gx) * (uint32_t)sizeof(float);

    // ---- Phase 1: Load ZPB planes (or zero-fill beyond extraction) ----
    bcopy_float4_vec regs[BCOPY_ZPB];
    constexpr bcopy_float4_vec zero_vec = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int dz = 0; dz < BCOPY_ZPB; ++dz) {
        int iz = z_start + dz;
        if (y_in_range && iz < ez) {
            auto plane_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                const_cast<float*>(d_src + (long)(z0 + iz) * ldimxy),
                0, -1, 0x00027000);
            regs[dz] = __builtin_bit_cast(bcopy_float4_vec,
                __builtin_amdgcn_raw_buffer_load_b128(
                    plane_rsrc, src_row_byte, 0, SLC));
        } else {
            regs[dz] = zero_vec;
        }
    }

    // ---- Phase 2: Zero-fill x lanes beyond extraction ----
    int x_data_loc = ex - tile_x0;
    if (x_data_loc > 32) x_data_loc = 32;

    if (x_data_loc < 32) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (xg * 4 + k >= x_data_loc) {
                #pragma unroll
                for (int dz = 0; dz < BCOPY_ZPB; ++dz)
                    regs[dz][k] = 0.0f;
            }
        }
    }

    // ---- Phase 3: Store ZPB planes ----
    if constexpr (DO_COPY) {
        uint32_t dst_byte = (gx + gy * wnx) * (uint32_t)sizeof(float);

        #pragma unroll
        for (int dz = 0; dz < BCOPY_ZPB; ++dz) {
            auto plane_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                d_dst + (long)(z_start + dz) * wnx * wny,
                0, -1, 0x00027000);
            auto vi = __builtin_bit_cast(bcopy_int4_vec, regs[dz]);
            __builtin_amdgcn_raw_buffer_store_b128(
                vi, plane_rsrc, dst_byte, 0, SLC);
        }
    }

    // ---- Optional: RMS (only extraction window samples) ----
    if constexpr (COMPUTE_RMS) {
        double thread_sum = 0.0;
        bool y_real = (gy < ey);
        int ex_loc_rms = ex - tile_x0;
        if (ex_loc_rms > 32) ex_loc_rms = 32;
        bool x_partial = (ex_loc_rms < 32);

        if (x_partial) {
            #pragma unroll
            for (int dz = 0; dz < BCOPY_ZPB; ++dz) {
                int iz = z_start + dz;
                if (y_real && iz < ez) {
                    if (xg * 4 + 0 < ex_loc_rms) { double d = (double)regs[dz][0]; thread_sum += d * d; }
                    if (xg * 4 + 1 < ex_loc_rms) { double d = (double)regs[dz][1]; thread_sum += d * d; }
                    if (xg * 4 + 2 < ex_loc_rms) { double d = (double)regs[dz][2]; thread_sum += d * d; }
                    if (xg * 4 + 3 < ex_loc_rms) { double d = (double)regs[dz][3]; thread_sum += d * d; }
                }
            }
        } else {
            #pragma unroll
            for (int dz = 0; dz < BCOPY_ZPB; ++dz) {
                int iz = z_start + dz;
                if (y_real && iz < ez) {
                    double d0 = (double)regs[dz][0], d1 = (double)regs[dz][1];
                    double d2 = (double)regs[dz][2], d3 = (double)regs[dz][3];
                    thread_sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                }
            }
        }

        using BlockReduce = rocprim::block_reduce<double, 256>;
        __shared__ typename BlockReduce::storage_type reduce_storage;
        double block_sum = 0.0;
        BlockReduce().reduce(thread_sum, block_sum, reduce_storage);

        if (tid == 0) {
            int block_id = blockIdx.z * gridDim.x * gridDim.y
                         + blockIdx.y * gridDim.x + blockIdx.x;
            d_partial_sums[block_id] = block_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// CopyFrom: wavelet layout → grid.
//
// Grid: (wnx/32, wny/32, wnz/ZPB).
// Load ZPB planes from wavelet → store extraction-window planes to grid.
// ---------------------------------------------------------------------------
__launch_bounds__(256)
__global__ void copyFromWaveletKernelOpt(
    const float* __restrict__ d_src,
    int wnx, int wny, int wnz,
    float* __restrict__ d_dst,
    int ldimx, int ldimxy,
    int x0, int y0, int z0,
    int ex, int ey, int ez)
{
    constexpr int SLC = 2;

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    int tile_x0 = blockIdx.x * 32;
    int tile_y0 = blockIdx.y * 32;
    int gx = tile_x0 + xg * 4;
    int gy = tile_y0 + yr;
    int z_start = blockIdx.z * BCOPY_ZPB;

    if (gy >= ey) return;
    if (gx >= ex) return;

    uint32_t src_byte = (gx + gy * wnx) * (uint32_t)sizeof(float);
    long wav_plane = (long)wnx * wny;

    bcopy_float4_vec regs[BCOPY_ZPB];
    #pragma unroll
    for (int dz = 0; dz < BCOPY_ZPB; ++dz) {
        auto plane_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<float*>(d_src + (long)(z_start + dz) * wav_plane),
            0, -1, 0x00027000);
        regs[dz] = __builtin_bit_cast(bcopy_float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                plane_rsrc, src_byte, 0, SLC));
    }

    uint32_t dst_byte = ((y0 + gy) * ldimx + x0 + gx) * (uint32_t)sizeof(float);
    bool x_all_in = (gx + 3 < ex);

    if (x_all_in) {
        #pragma unroll
        for (int dz = 0; dz < BCOPY_ZPB; ++dz) {
            int iz = z_start + dz;
            if (iz < ez) {
                auto plane_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                    d_dst + (long)(z0 + iz) * ldimxy,
                    0, -1, 0x00027000);
                auto vi = __builtin_bit_cast(bcopy_int4_vec, regs[dz]);
                __builtin_amdgcn_raw_buffer_store_b128(
                    vi, plane_rsrc, dst_byte, 0, SLC);
            }
        }
    } else {
        #pragma unroll
        for (int dz = 0; dz < BCOPY_ZPB; ++dz) {
            int iz = z_start + dz;
            if (iz < ez) {
                auto plane_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                    d_dst + (long)(z0 + iz) * ldimxy,
                    0, -1, 0x00027000);
                float e0 = regs[dz][0], e1 = regs[dz][1];
                float e2 = regs[dz][2], e3 = regs[dz][3];
                if (gx + 0 < ex) __builtin_amdgcn_raw_buffer_store_b32(
                    __builtin_bit_cast(int, e0), plane_rsrc, dst_byte + 0, 0, SLC);
                if (gx + 1 < ex) __builtin_amdgcn_raw_buffer_store_b32(
                    __builtin_bit_cast(int, e1), plane_rsrc, dst_byte + 4, 0, SLC);
                if (gx + 2 < ex) __builtin_amdgcn_raw_buffer_store_b32(
                    __builtin_bit_cast(int, e2), plane_rsrc, dst_byte + 8, 0, SLC);
                if (gx + 3 < ex) __builtin_amdgcn_raw_buffer_store_b32(
                    __builtin_bit_cast(int, e3), plane_rsrc, dst_byte + 12, 0, SLC);
            }
        }
    }
}

__global__ void copyFromWaveletKernel(
    const float* __restrict__ d_src,
    int wnx, int wny, int wnz,
    float* __restrict__ d_dst,
    int ldimx, int ldimxy,
    int x0, int y0, int z0,
    int ex, int ey, int ez)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    long total_out = (long)ex * ey * ez;
    if (gid >= total_out) return;

    int iz = gid / (ex * ey);
    int rem = gid - iz * ex * ey;
    int iy = rem / ex;
    int ix = rem - iy * ex;

    long src_offset = (long)iz * wnx * wny + (long)iy * wnx + ix;
    float val = d_src[src_offset];

    long dst_offset = (long)(z0 + iz) * ldimxy + (long)(y0 + iy) * ldimx + (x0 + ix);
    d_dst[dst_offset] = val;
}

// Reduce partial sums → RMS.  Single block, 256 threads.
__launch_bounds__(256)
__global__ void reducePartialSumsToRMS(
    const double* __restrict__ d_partial_sums,
    double* __restrict__ d_rms_out,
    int num_partials,
    long total_samples)
{
    using BlockReduce = rocprim::block_reduce<double, 256>;
    __shared__ typename BlockReduce::storage_type reduce_storage;

    double thread_sum = 0.0;
    for (int i = threadIdx.x; i < num_partials; i += 256)
        thread_sum += d_partial_sums[i];

    double block_sum = 0.0;
    BlockReduce().reduce(thread_sum, block_sum, reduce_storage);

    if (threadIdx.x == 0)
        *d_rms_out = sqrt(block_sum / (double)total_samples);
}

#endif
