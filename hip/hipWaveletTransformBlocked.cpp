// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// DEPRECATED: Development/benchmark file, not used by the production API.
// Contains known int overflow issues for large lateral dims. Will be removed.
//
// GPU wavelet transform for 32x32x32 blocks.
// 2D thread block (8x32 = 256 threads), float4 vectorised loads/stores,
// inline-asm with 32-bit VGPR offset + SGPR base, non-temporal memory ops.

#include "hipWaveletTransformBlocked.h"
#include "ds79.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

using float4_vec = __attribute__((__vector_size__(4 * sizeof(float)))) float;

// ---------------------------------------------------------------------------
// Inline asm helpers — batched non-temporal load/store with auto-incrementing
// 32-bit VGPR offset.  A single VGPR carries the offset across all batches;
// v_add_u32 advances it by the SGPR stride after each dwordx4 access.
// Early-clobber (&) on load outputs prevents aliasing with the offset VGPR.
// ---------------------------------------------------------------------------

__device__ __forceinline__
void global_load_x4_nt_inc(float4_vec& d0, float4_vec& d1,
                            float4_vec& d2, float4_vec& d3,
                            int& off, int stride, const float* base)
{
    asm volatile(
        "global_load_dwordx4 %0, %4, %5 nt\n\t"
        "v_add_u32_e32 %4, %6, %4\n\t"
        "global_load_dwordx4 %1, %4, %5 nt\n\t"
        "v_add_u32_e32 %4, %6, %4\n\t"
        "global_load_dwordx4 %2, %4, %5 nt\n\t"
        "v_add_u32_e32 %4, %6, %4\n\t"
        "global_load_dwordx4 %3, %4, %5 nt\n\t"
        "v_add_u32_e32 %4, %6, %4\n\t"
        "s_waitcnt vmcnt(0)"
        : "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3), "+v"(off)
        : "s"(base), "s"(stride)
        : "memory"
    );
}

__device__ __forceinline__
void global_store_x4_nt_inc(float4_vec v0, float4_vec v1,
                             float4_vec v2, float4_vec v3,
                             int& off, int stride, float* base)
{
    asm volatile(
        "global_store_dwordx4 %0, %1, %5 nt\n\t"
        "v_add_u32_e32 %0, %6, %0\n\t"
        "global_store_dwordx4 %0, %2, %5 nt\n\t"
        "v_add_u32_e32 %0, %6, %0\n\t"
        "global_store_dwordx4 %0, %3, %5 nt\n\t"
        "v_add_u32_e32 %0, %6, %0\n\t"
        "global_store_dwordx4 %0, %4, %5 nt\n\t"
        "v_add_u32_e32 %0, %6, %0"
        : "+v"(off)
        : "v"(v0), "v"(v1), "v"(v2), "v"(v3), "s"(base), "s"(stride)
        : "memory"
    );
}

__device__ __forceinline__
void global_load_x2_nt_inc(float4_vec& d0, float4_vec& d1,
                            int& off, int stride, const float* base)
{
    asm volatile(
        "global_load_dwordx4 %0, %2, %3 nt\n\t"
        "v_add_u32_e32 %2, %4, %2\n\t"
        "global_load_dwordx4 %1, %2, %3 nt\n\t"
        "v_add_u32_e32 %2, %4, %2\n\t"
        "s_waitcnt vmcnt(0)"
        : "=&v"(d0), "=&v"(d1), "+v"(off)
        : "s"(base), "s"(stride)
        : "memory"
    );
}

__device__ __forceinline__
void global_store_x2_nt_inc(float4_vec v0, float4_vec v1,
                             int& off, int stride, float* base)
{
    asm volatile(
        "global_store_dwordx4 %0, %1, %3 nt\n\t"
        "v_add_u32_e32 %0, %4, %0\n\t"
        "global_store_dwordx4 %0, %2, %3 nt\n\t"
        "v_add_u32_e32 %0, %4, %0"
        : "+v"(off)
        : "v"(v0), "v"(v1), "s"(base), "s"(stride)
        : "memory"
    );
}

// ---------------------------------------------------------------------------
// 2D batched kernel (8x32 thread block).
// Out-of-place: reads from input, writes to output.
// Nplanes stored in LDS, remaining (32-Nplanes) in VGPRs.
// Store-order swap: register planes written first to free VGPRs before the
// LDS→VGPR→global writeback of the shared-memory planes.
// ---------------------------------------------------------------------------
template<int Nplanes, int Batch>
__launch_bounds__(256, 4)
__global__ void waveletBlockedForwardKernel2DBatched(
    const float* __restrict__ input,
    float* __restrict__ output,
    int ldimx,
    int ldimxy)
{
    static_assert(Batch == 2 || Batch == 4, "Inline asm supports Batch=2 or Batch=4");
    static_assert(Nplanes % Batch == 0, "Nplanes must be divisible by Batch");
    static_assert((32 - Nplanes) % Batch == 0, "num_reg_planes must be divisible by Batch");

    __shared__ float shared_data[Nplanes * 32 * 32];

    int local_y = threadIdx.y;
    int local_x_base = threadIdx.x * 4;

    int global_x_base = blockIdx.x * 32 + local_x_base;
    int global_y = blockIdx.y * 32 + local_y;
    int shared_base_yx = local_y * 32 + local_x_base;
    int global_base_yx = global_x_base + global_y * ldimx;

    constexpr int num_reg_planes = 32 - Nplanes;
    float4_vec reg_data[num_reg_planes];

    int byte_stride = ldimxy * (int)sizeof(float);
    int byte_off = (global_base_yx + blockIdx.z * 32 * ldimxy) * (int)sizeof(float);

    #pragma unroll 1
    for (int b = 0; b < Nplanes; b += Batch) {
        if constexpr (Batch == 4) {
            float4_vec tmp0, tmp1, tmp2, tmp3;
            global_load_x4_nt_inc(tmp0, tmp1, tmp2, tmp3, byte_off, byte_stride, input);
            int sb0 = (b+0) * 32 * 32 + shared_base_yx;
            shared_data[sb0+0] = tmp0[0]; shared_data[sb0+1] = tmp0[1];
            shared_data[sb0+2] = tmp0[2]; shared_data[sb0+3] = tmp0[3];
            int sb1 = (b+1) * 32 * 32 + shared_base_yx;
            shared_data[sb1+0] = tmp1[0]; shared_data[sb1+1] = tmp1[1];
            shared_data[sb1+2] = tmp1[2]; shared_data[sb1+3] = tmp1[3];
            int sb2 = (b+2) * 32 * 32 + shared_base_yx;
            shared_data[sb2+0] = tmp2[0]; shared_data[sb2+1] = tmp2[1];
            shared_data[sb2+2] = tmp2[2]; shared_data[sb2+3] = tmp2[3];
            int sb3 = (b+3) * 32 * 32 + shared_base_yx;
            shared_data[sb3+0] = tmp3[0]; shared_data[sb3+1] = tmp3[1];
            shared_data[sb3+2] = tmp3[2]; shared_data[sb3+3] = tmp3[3];
        } else {
            float4_vec tmp0, tmp1;
            global_load_x2_nt_inc(tmp0, tmp1, byte_off, byte_stride, input);
            int sb0 = (b+0) * 32 * 32 + shared_base_yx;
            shared_data[sb0+0] = tmp0[0]; shared_data[sb0+1] = tmp0[1];
            shared_data[sb0+2] = tmp0[2]; shared_data[sb0+3] = tmp0[3];
            int sb1 = (b+1) * 32 * 32 + shared_base_yx;
            shared_data[sb1+0] = tmp1[0]; shared_data[sb1+1] = tmp1[1];
            shared_data[sb1+2] = tmp1[2]; shared_data[sb1+3] = tmp1[3];
        }
    }

    #pragma unroll
    for (int b = 0; b < num_reg_planes; b += Batch) {
        if constexpr (Batch == 4) {
            global_load_x4_nt_inc(reg_data[b+0], reg_data[b+1],
                                  reg_data[b+2], reg_data[b+3],
                                  byte_off, byte_stride, input);
        } else {
            global_load_x2_nt_inc(reg_data[b+0], reg_data[b+1],
                                  byte_off, byte_stride, input);
        }
    }

    // -----------------------------------------------------------------------
    // 3D wavelet transform plan (Ds79, separable: Y -> X -> Z)
    //
    // LDS: stride 32 (no padding), Occ 2.
    //   8 planes * 32 * 32 * 4 B = 32 KB -> 64 KB / 32 KB = 2 workgroups.
    //   X-direction reads have 32-way bank conflicts (stride 32 = 32 banks),
    //   accepted for now; revisit after profiling.
    //   Y-direction reads are conflict-free (consecutive columns).
    //
    // Thread mapping for XY transforms:
    //   tid = threadIdx.x + threadIdx.y * 8   (0..255)
    //   plane = tid / 32   (0..7, which of the 8 planes in this pass)
    //   col_or_row = tid % 32   (0..31)
    //
    // Phase 1 -- XY transforms (4 passes x 8 planes):
    //   a) Load 8 planes from global -> LDS.
    //   b) Y-transform: 32 threads/plane, 1 thread/column.
    //      Read 32 y-values at stride 32, transform (ds79_forward), write back.
    //   c) X-transform: 32 threads/plane, 1 thread/row.
    //      Read 32 contiguous x-values, transform (ds79_forward), write back.
    //   d) Move 8 transformed planes from LDS -> registers.
    //
    // Phase 2 -- Z-transform (registers only, no LDS):
    //   32 planes x float4 = 128 VGPRs per thread.
    //   Gather 32 z-values per x-component, transform (ds79_forward),
    //   scatter back.
    //
    // Phase 3 -- Store all 32 planes from registers -> global.
    //
    // ds79_forward (ds79.h) operates on 32 contiguous values in registers,
    // reused for all 3 directions.
    // -----------------------------------------------------------------------

    __syncthreads();

    // Store register planes first (frees VGPRs before LDS writeback)
    byte_off = (global_base_yx + (blockIdx.z * 32 + Nplanes) * ldimxy) * (int)sizeof(float);

    #pragma unroll
    for (int b = 0; b < num_reg_planes; b += Batch) {
        if constexpr (Batch == 4) {
            global_store_x4_nt_inc(reg_data[b+0], reg_data[b+1],
                                   reg_data[b+2], reg_data[b+3],
                                   byte_off, byte_stride, output);
        } else {
            global_store_x2_nt_inc(reg_data[b+0], reg_data[b+1],
                                   byte_off, byte_stride, output);
        }
    }

    // Store LDS planes
    byte_off = (global_base_yx + blockIdx.z * 32 * ldimxy) * (int)sizeof(float);

    #pragma unroll 1
    for (int b = 0; b < Nplanes; b += Batch) {
        if constexpr (Batch == 4) {
            float4_vec tmp0, tmp1, tmp2, tmp3;
            int sb0 = (b+0) * 32 * 32 + shared_base_yx;
            tmp0[0] = shared_data[sb0+0]; tmp0[1] = shared_data[sb0+1];
            tmp0[2] = shared_data[sb0+2]; tmp0[3] = shared_data[sb0+3];
            int sb1 = (b+1) * 32 * 32 + shared_base_yx;
            tmp1[0] = shared_data[sb1+0]; tmp1[1] = shared_data[sb1+1];
            tmp1[2] = shared_data[sb1+2]; tmp1[3] = shared_data[sb1+3];
            int sb2 = (b+2) * 32 * 32 + shared_base_yx;
            tmp2[0] = shared_data[sb2+0]; tmp2[1] = shared_data[sb2+1];
            tmp2[2] = shared_data[sb2+2]; tmp2[3] = shared_data[sb2+3];
            int sb3 = (b+3) * 32 * 32 + shared_base_yx;
            tmp3[0] = shared_data[sb3+0]; tmp3[1] = shared_data[sb3+1];
            tmp3[2] = shared_data[sb3+2]; tmp3[3] = shared_data[sb3+3];
            global_store_x4_nt_inc(tmp0, tmp1, tmp2, tmp3,
                                   byte_off, byte_stride, output);
        } else {
            float4_vec tmp0, tmp1;
            int sb0 = (b+0) * 32 * 32 + shared_base_yx;
            tmp0[0] = shared_data[sb0+0]; tmp0[1] = shared_data[sb0+1];
            tmp0[2] = shared_data[sb0+2]; tmp0[3] = shared_data[sb0+3];
            int sb1 = (b+1) * 32 * 32 + shared_base_yx;
            tmp1[0] = shared_data[sb1+0]; tmp1[1] = shared_data[sb1+1];
            tmp1[2] = shared_data[sb1+2]; tmp1[3] = shared_data[sb1+3];
            global_store_x2_nt_inc(tmp0, tmp1, byte_off, byte_stride, output);
        }
    }
}

// ---------------------------------------------------------------------------
// All-register kernel: no shared memory, all 32 planes in VGPRs.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletBlockedForwardKernel1D_AllReg(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");

    constexpr int PLANES = 32;

    int tid = threadIdx.x;
    int local_y = tid / 8;
    int local_x_base = (tid % 8) * 4;

    int gx = blockIdx.x * 32 + local_x_base;
    int gy = blockIdx.y * 32 + local_y;
    int gz = blockIdx.z * 32;

    float4_vec reg_data[PLANES];

    int byte_stride = ldimxy * (int)sizeof(float);
    int byte_off = (gx + gy * ldimx + gz * ldimxy) * (int)sizeof(float);

    #pragma unroll
    for (int p = 0; p < PLANES; p += 4) {
        global_load_x4_nt_inc(reg_data[p+0], reg_data[p+1],
                              reg_data[p+2], reg_data[p+3],
                              byte_off, byte_stride, data);
    }

    // TODO: wavelet transform

    byte_off = (gx + gy * ldimx + gz * ldimxy) * (int)sizeof(float);

    #pragma unroll
    for (int p = 0; p < PLANES; p += 4) {
        global_store_x4_nt_inc(reg_data[p+0], reg_data[p+1],
                               reg_data[p+2], reg_data[p+3],
                               byte_off, byte_stride, data);
    }
}

// ---------------------------------------------------------------------------
// 3D wavelet kernel — Y-direction only (first step).
// Processes all 32 planes through LDS in 4 passes of 8 planes.
// Thread mapping: tid/32 = plane (0..7), tid%32 = column (0..31).
// ---------------------------------------------------------------------------
__launch_bounds__(256, 2)
__global__ void waveletForward3D_Y(
    const float* __restrict__ input,
    float* __restrict__ output,
    int ldimx,
    int ldimxy)
{
    __shared__ float lds[8 * 32 * 32];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int plane_in_pass = tid / 32;
    int col = tid % 32;

    int gx_base = blockIdx.x * 32;
    int gy_base = blockIdx.y * 32;
    int gz_base = blockIdx.z * 32;

    for (int pass = 0; pass < 4; ++pass) {
        int z0 = pass * 8;

        // Load 8 planes from global -> LDS.
        // Each thread loads one float per (plane, row) combination.
        // 256 threads = 8 planes x 32 rows; loop over 32 columns.
        for (int x = 0; x < 32; ++x) {
            int gz = gz_base + z0 + plane_in_pass;
            int gy = gy_base + col;
            int gx = gx_base + x;
            lds[plane_in_pass * 1024 + col * 32 + x] =
                input[gx + gy * ldimx + gz * ldimxy];
        }

        __syncthreads();

        // Y-transform: each thread reads column `col` of plane `plane_in_pass`.
        // 32 y-values at stride 32 -> contiguous in registers.
        float vals[32];
        for (int row = 0; row < 32; ++row)
            vals[row] = lds[plane_in_pass * 1024 + row * 32 + col];

        ds79_forward(vals, 32);

        for (int row = 0; row < 32; ++row)
            lds[plane_in_pass * 1024 + row * 32 + col] = vals[row];

        __syncthreads();

        // Store 8 planes from LDS -> global.
        for (int x = 0; x < 32; ++x) {
            int gz = gz_base + z0 + plane_in_pass;
            int gy = gy_base + col;
            int gx = gx_base + x;
            output[gx + gy * ldimx + gz * ldimxy] =
                lds[plane_in_pass * 1024 + col * 32 + x];
        }

        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

template<int Nplanes, int Batch>
hipError_t hipWaveletTransform3DBlockedForward2DImpl(
    const float* input,
    float* output,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(8, 32);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletBlockedForwardKernel2DBatched<Nplanes, Batch>),
                       grid, block, 0, 0, input, output, ldimx, ldimxy);
    return hipGetLastError();
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransform3DBlockedAllReg(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletBlockedForwardKernel1D_AllReg<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

hipError_t hipWaveletTransform3DForwardY(
    const float* input,
    float* output,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(8, 32);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL(waveletForward3D_Y,
                       grid, block, 0, 0, input, output, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

// 2D (8x32) batched kernel
template __global__ void waveletBlockedForwardKernel2DBatched<4, 4>(const float* __restrict__, float* __restrict__, int, int);
template hipError_t hipWaveletTransform3DBlockedForward2DImpl<4, 4>(const float*, float*, int, int, int, int, int, int, int, int);
template __global__ void waveletBlockedForwardKernel2DBatched<6, 2>(const float* __restrict__, float* __restrict__, int, int);
template hipError_t hipWaveletTransform3DBlockedForward2DImpl<6, 2>(const float*, float*, int, int, int, int, int, int, int, int);
template __global__ void waveletBlockedForwardKernel2DBatched<8, 4>(const float* __restrict__, float* __restrict__, int, int);
template hipError_t hipWaveletTransform3DBlockedForward2DImpl<8, 4>(const float*, float*, int, int, int, int, int, int, int, int);
template __global__ void waveletBlockedForwardKernel2DBatched<12, 4>(const float* __restrict__, float* __restrict__, int, int);
template hipError_t hipWaveletTransform3DBlockedForward2DImpl<12, 4>(const float*, float*, int, int, int, int, int, int, int, int);
template __global__ void waveletBlockedForwardKernel2DBatched<16, 4>(const float* __restrict__, float* __restrict__, int, int);
template hipError_t hipWaveletTransform3DBlockedForward2DImpl<16, 4>(const float*, float*, int, int, int, int, int, int, int, int);

// All-register kernel
template __global__ void waveletBlockedForwardKernel1D_AllReg<256, 3>(float* __restrict__, int, int);
template hipError_t hipWaveletTransform3DBlockedAllReg<256, 3>(float*, int, int, int, int, int, int, int, int);
