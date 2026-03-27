// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIPWAVELET_RLE_INVERSE_H
#define HIPWAVELET_RLE_INVERSE_H

// Fused inverse: RLE decode + dequantize + inverse wavelet ZYX.
// Mirrors the forward fused kernel in hipWaveletRLE.h.
// Compressed data read from global. LDS used for wavelet XY transform.
// Per-block layout: [1024B zline_meta] [RLE data]. Meta is embedded in-band.

#include <hip/hip_runtime.h>
#include <rocprim/block/block_scan.hpp>
#include "ds79.h"
#include "Run_Length_Escape_Codes.hxx"
#include "hipRLEDecode.h"
#include "hipWaveletRLE.h"

using wrli_float4_vec = ds79_float4_vec;

__launch_bounds__(256, 2)
__global__ void waveletRLEInverseFusedKernel(
    const unsigned char* __restrict__ input,
    const size_t* __restrict__ block_sizes,
    const size_t* __restrict__ block_offsets,
    float* __restrict__ output,
    float inv_scale,
    int ldimx, int ldimxy,
    int header_size)
{
    constexpr int PLANES = 32;
    constexpr int BATCH  = 8;
    constexpr int SLC    = 2;
    constexpr int NTHREADS = 256;
    using BlockScan = rocprim::block_scan<int, NTHREADS>;

    __shared__ union {
        float wavelet[BATCH * 1024];
        typename BlockScan::storage_type scan;
    } lds;

    int tid = threadIdx.x;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    float actual_inv_scale;
    const unsigned char* data_base;
    const unsigned char* block_in;
    if (header_size != 0) {
        const int* hdr = (const int*)input;
        int num_blocks = hdr[0];
        int num_mulfacs = hdr[1];
        const float* mulfacs = (const float*)(input + 8 + 8 * num_blocks);
        float mf = (num_mulfacs == 1) ? mulfacs[0] : mulfacs[bid];
        actual_inv_scale = 1.0f / mf;
        data_base = input + 8 + 8 * num_blocks + 4 * num_mulfacs;
        block_in = data_base + ((const size_t*)(input + 8))[bid];
    } else {
        actual_inv_scale = inv_scale;
        data_base = input;
        if (block_offsets != nullptr)
            block_in = data_base + block_offsets[bid];
        else
            block_in = data_base + (size_t)bid * 4 * WRLE_LDS_BYTES;
    }

    // Block layout: [1024B zline_meta] [RLE data]
    const unsigned char* meta_in = block_in;
    const unsigned char* rle_in  = block_in + WRLE_META_PER_BLOCK;

    int xg = tid % 8;
    int yr = tid / 8;

    wrli_float4_vec regs[PLANES];

    // ---- Phase 1: Decode z-lines (4 passes) ----
    int stream_offset = 0;
    for (int x_off = 0; x_off < 4; ++x_off) {
        int my_bytes = (int)meta_in[x_off * 256 + tid];

        int my_offset, pass_total;
        BlockScan().exclusive_scan(my_bytes, my_offset, 0, pass_total, lds.scan);
        __syncthreads();

        float zline[32];
        wrle_zline_decode(rle_in + stream_offset + my_offset, my_bytes,
                          actual_inv_scale, zline);

        #pragma unroll
        for (int z = 0; z < 32; ++z)
            regs[z][x_off] = zline[z];

        stream_offset += pass_total;
        __syncthreads();
    }

    // ---- Phase 2: Inverse X+Y transform in LDS ----
    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            wrli_float4_vec v = regs[pb + dp];
            int x0 = xg * 4;
            lds.wavelet[dp * 1024 + (x0+0) * 32 + (yr ^ (x0+0))] = v[0];
            lds.wavelet[dp * 1024 + (x0+1) * 32 + (yr ^ (x0+1))] = v[1];
            lds.wavelet[dp * 1024 + (x0+2) * 32 + (yr ^ (x0+2))] = v[2];
            lds.wavelet[dp * 1024 + (x0+3) * 32 + (yr ^ (x0+3))] = v[3];
        }
        __syncthreads();

        int pl  = tid / 32;
        int pos = tid % 32;
        float line[32];

        // Inverse X
        for (int x = 0; x < 32; x++)
            line[x] = lds.wavelet[pl * 1024 + x * 32 + (pos ^ x)];
        us79_inverse_reg32(line);
        for (int x = 0; x < 32; x++)
            lds.wavelet[pl * 1024 + x * 32 + (pos ^ x)] = line[x];
        __syncthreads();

        // Inverse Y
        for (int y = 0; y < 32; y++)
            line[y] = lds.wavelet[pl * 1024 + pos * 32 + (y ^ pos)];
        us79_inverse_reg32(line);
        for (int y = 0; y < 32; y++)
            lds.wavelet[pl * 1024 + pos * 32 + (y ^ pos)] = line[y];
        __syncthreads();

        for (int dp = 0; dp < BATCH; dp++) {
            wrli_float4_vec v;
            int x0 = xg * 4;
            v[0] = lds.wavelet[dp * 1024 + (x0+0) * 32 + (yr ^ (x0+0))];
            v[1] = lds.wavelet[dp * 1024 + (x0+1) * 32 + (yr ^ (x0+1))];
            v[2] = lds.wavelet[dp * 1024 + (x0+2) * 32 + (yr ^ (x0+2))];
            v[3] = lds.wavelet[dp * 1024 + (x0+3) * 32 + (yr ^ (x0+3))];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    // ---- Phase 3: Inverse Z in registers ----
    us79_inverse_f4_scalar_tmp(regs, PLANES);

    // ---- Phase 4: Store to global (buffer instructions) ----
    float* block_base = output + (size_t)blockIdx.z * 32 * ldimxy;
    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    uint32_t byte_off = (gx + gy * ldimx) * (uint32_t)sizeof(float);

    #pragma unroll
    for (int p = 0; p < PLANES; p++) {
        auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
            block_base + (long)p * ldimxy,
            0, -1, 0x00027000);
        auto v = __builtin_bit_cast(__attribute__((__vector_size__(16))) int, regs[p]);
        __builtin_amdgcn_raw_buffer_store_b128(v, rsrc, byte_off, 0, SLC);
    }
}

inline hipError_t hipWaveletRLEInverseFusedFixedStride(
    const unsigned char* compressed,
    const size_t* block_sizes,
    float* output,
    float inv_scale,
    int nx, int ny, int nz,
    int ldimx, int ldimxy)
{
    dim3 grid((nx + 31) / 32, (ny + 31) / 32, (nz + 31) / 32);
    waveletRLEInverseFusedKernel<<<grid, dim3(256)>>>(
        compressed, block_sizes, nullptr,
        output, inv_scale, ldimx, ldimxy, 0);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Segment-aligned inverse: no per-z-line metadata.
// Thread 0 decodes the segment-aligned RLE stream into LDS (raster order),
// then all threads do the inverse Y/X/Z transform.
// ---------------------------------------------------------------------------

#include "hipSegmentedRLE.h"

__launch_bounds__(256, 2)
__global__ void waveletSegRLEInverseFusedKernel(
    const unsigned char* __restrict__ input,
    const size_t* __restrict__ block_sizes,
    const size_t* __restrict__ block_offsets,
    float* __restrict__ output,
    float inv_scale,
    int ldimx, int ldimxy,
    int header_size)
{
    constexpr int PLANES = 32;
    constexpr int BATCH  = 8;
    constexpr int SLC    = 2;

    __shared__ float lds_wavelet[BATCH * 1024];

    int tid = threadIdx.x;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    float actual_inv_scale;
    const unsigned char* block_in;
    if (header_size != 0) {
        const int* hdr = (const int*)input;
        int num_blocks = hdr[0];
        int num_mulfacs = hdr[1];
        const float* mulfacs = (const float*)(input + 8 + 8 * num_blocks);
        float mf = (num_mulfacs == 1) ? mulfacs[0] : mulfacs[bid];
        actual_inv_scale = 1.0f / mf;
        const unsigned char* data_base = input + 8 + 8 * num_blocks + 4 * num_mulfacs;
        block_in = data_base + ((const size_t*)(input + 8))[bid];
    } else {
        actual_inv_scale = inv_scale;
        const unsigned char* data_base = input;
        if (block_offsets != nullptr)
            block_in = data_base + block_offsets[bid];
        else
            block_in = data_base + (size_t)bid * 4 * WRLE_LDS_BYTES;
    }

    int xg = tid % 8;
    int yr = tid / 8;

    wrli_float4_vec regs[PLANES];

    int stream_pos = 0;
    int pending_zeros = 0;

    for (int pb = 0; pb < PLANES; pb += BATCH) {
        // Thread 0: decode this batch from segment-aligned stream → LDS
        if (tid == 0)
            seg_rle_decode_batch(block_in, actual_inv_scale, lds_wavelet,
                                 BATCH, SEG_RLE_SEG_SIZE, &stream_pos, &pending_zeros);
        __syncthreads();

        // Inverse X transform
        {
            int pl  = tid / 32;
            int pos = tid % 32;
            float line[32];
            for (int x = 0; x < 32; x++)
                line[x] = lds_wavelet[pl * 1024 + x * 32 + (pos ^ x)];
            us79_inverse_reg32(line);
            for (int x = 0; x < 32; x++)
                lds_wavelet[pl * 1024 + x * 32 + (pos ^ x)] = line[x];
        }
        __syncthreads();

        // Inverse Y transform
        {
            int pl  = tid / 32;
            int pos = tid % 32;
            float line[32];
            for (int y = 0; y < 32; y++)
                line[y] = lds_wavelet[pl * 1024 + pos * 32 + (y ^ pos)];
            us79_inverse_reg32(line);
            for (int y = 0; y < 32; y++)
                lds_wavelet[pl * 1024 + pos * 32 + (y ^ pos)] = line[y];
        }
        __syncthreads();

        // Read back to registers as z-lines
        for (int dp = 0; dp < BATCH; dp++) {
            wrli_float4_vec v;
            int x0 = xg * 4;
            v[0] = lds_wavelet[dp * 1024 + (x0+0) * 32 + (yr ^ (x0+0))];
            v[1] = lds_wavelet[dp * 1024 + (x0+1) * 32 + (yr ^ (x0+1))];
            v[2] = lds_wavelet[dp * 1024 + (x0+2) * 32 + (yr ^ (x0+2))];
            v[3] = lds_wavelet[dp * 1024 + (x0+3) * 32 + (yr ^ (x0+3))];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    // Inverse Z transform in registers
    us79_inverse_f4_scalar_tmp(regs, PLANES);

    // Store to global
    float* block_base_out = output + (size_t)blockIdx.z * 32 * ldimxy;
    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    uint32_t byte_off = (gx + gy * ldimx) * (uint32_t)sizeof(float);

    #pragma unroll
    for (int p = 0; p < PLANES; p++) {
        auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
            block_base_out + (long)p * ldimxy,
            0, -1, 0x00027000);
        auto v = __builtin_bit_cast(__attribute__((__vector_size__(16))) int, regs[p]);
        __builtin_amdgcn_raw_buffer_store_b128(v, rsrc, byte_off, 0, SLC);
    }
}

#endif // HIPWAVELET_RLE_INVERSE_H
