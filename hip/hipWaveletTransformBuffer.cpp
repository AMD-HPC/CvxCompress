// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// DEPRECATED: Development/benchmark file, not used by the production API.
// Contains known int overflow issues for large lateral dims. Will be removed.
//
// GPU wavelet transform for 32x32x32 blocks — buffer instruction variant.
//
// 1D thread block (256 threads), float4 vectorised loads/stores.
// Uses __builtin_amdgcn_raw_buffer_{load,store}_b128 for SGPR-based
// addressing with no inline asm.  The compiler auto-schedules vmcnt.
//
// Thread mapping (256 threads cover a 32×32 XY tile):
//   tid % 8  → x-group  (each loads 4 consecutive floats via dwordx4)
//   tid / 8  → y-row    (32 rows)
// Each thread loads 32 z-planes × float4 = 128 VGPRs of payload data.
//
// Address decomposition:
//   buffer descriptor base (SGPR) = data + blockIdx.z * 32 * ldimxy
//   voffset (VGPR)                = (gx + gy * ldimx) * sizeof(float)
//   soffset (SGPR)                = p * ldimxy * sizeof(float)

#include "hipWaveletTransformBuffer.h"
#include <hip/hip_runtime.h>

using float4_vec = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using uint4_vec  = __attribute__((__vector_size__(4 * sizeof(unsigned)))) unsigned;

#include "ds79.h"
#include "ds79_f4_reg32.inc"

// ds79 forward wavelet on float4_vec: applies the transform to all 4 lanes
// simultaneously.  Operates on regs[0..dim-1] in-place across the z-direction.
__device__ inline void ds79_forward_f4(float4_vec* data, int dim) {
    float4_vec tmp[32];
    for (int n = dim; n >= 2; n = n - n / 2) {
        for (int i = 0; i < n; ++i) tmp[i] = data[i];
        int nh = n / 2;
        int nl = n - nh;
        for (int ix = 0; ix < nl; ++ix) {
            int i0 = 2 * ix;
            int im1 = ds79_mirr(i0 - 1, n), ip1 = ds79_mirr(i0 + 1, n);
            int im2 = ds79_mirr(i0 - 2, n), ip2 = ds79_mirr(i0 + 2, n);
            int im3 = ds79_mirr(i0 - 3, n), ip3 = ds79_mirr(i0 + 3, n);
            int im4 = ds79_mirr(i0 - 4, n), ip4 = ds79_mirr(i0 + 4, n);
            float4_vec acc = DS79_AL0 * tmp[i0];
            acc += DS79_AL1 * (tmp[im1] + tmp[ip1]);
            acc += DS79_AL2 * (tmp[im2] + tmp[ip2]);
            acc += DS79_AL3 * (tmp[im3] + tmp[ip3]);
            acc += DS79_AL4 * (tmp[im4] + tmp[ip4]);
            data[ix] = acc;
        }
        for (int ix = 0; ix < nh; ++ix) {
            int i0 = 2 * ix + 1;
            int im1 = ds79_mirr(i0 - 1, n), ip1 = ds79_mirr(i0 + 1, n);
            int im2 = ds79_mirr(i0 - 2, n), ip2 = ds79_mirr(i0 + 2, n);
            int im3 = ds79_mirr(i0 - 3, n), ip3 = ds79_mirr(i0 + 3, n);
            float4_vec acc = DS79_AH0 * tmp[i0];
            acc += DS79_AH1 * (tmp[im1] + tmp[ip1]);
            acc += DS79_AH2 * (tmp[im2] + tmp[ip2]);
            acc += DS79_AH3 * (tmp[im3] + tmp[ip3]);
            data[nl + ix] = acc;
        }
    }
}

template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletBufferKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;
    constexpr int SLC = 2;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        block_base, 0, -1, 0x00027000);

    int byte_off    = (gx + gy * ldimx) * (int)sizeof(float);
    int byte_stride = ldimxy * (int)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, p * byte_stride, SLC));

    // Negate to prevent DCE; TODO: replace with wavelet transform
    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = -regs[p];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        __builtin_amdgcn_raw_buffer_store_b128(
            __builtin_bit_cast(uint4_vec, regs[p]),
            rsrc, byte_off, p * byte_stride, SLC);
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
hipError_t hipWaveletTransformBuffer(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletBufferKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Baseline: standard float4 pointer loads/stores, 64-bit addressing.
// No asm, no buffer intrinsics.  Each plane address costs 2 VGPRs.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletBaselineKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;
    int gz  = blockIdx.z * 32;

    size_t base_idx = gx + gy * ldimx + (size_t)gz * ldimxy;

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = __builtin_nontemporal_load(
            reinterpret_cast<const float4_vec*>(
                &data[base_idx + (size_t)p * ldimxy]));

    // Negate to prevent DCE; TODO: replace with wavelet transform
    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = -regs[p];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        __builtin_nontemporal_store(regs[p],
            reinterpret_cast<float4_vec*>(
                &data[base_idx + (size_t)p * ldimxy]));
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformBaseline(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletBaselineKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// saddr workaround: float4 nt loads/stores with uint8_t* + uint32_t offset.
// Tricks the compiler into emitting global_load/store with saddr (SGPR base +
// VGPR offset) without inline asm or buffer intrinsics.
// See: https://github.com/llvm/llvm-project/issues/55314
// ---------------------------------------------------------------------------
__device__ __forceinline__
float4_vec saddr_load_nt(const float* base, uint32_t byte_off) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(base);
    return __builtin_nontemporal_load(
        reinterpret_cast<const float4_vec*>(p + byte_off));
}

__device__ __forceinline__
void saddr_store_nt(float* base, uint32_t byte_off, float4_vec val) {
    uint8_t* p = reinterpret_cast<uint8_t*>(base);
    __builtin_nontemporal_store(val,
        reinterpret_cast<float4_vec*>(p + byte_off));
}

template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletSaddrKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    uint32_t xy_byte     = (uint32_t)(gx + gy * ldimx) * (uint32_t)sizeof(float);
    uint32_t byte_stride = (uint32_t)ldimxy * (uint32_t)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = saddr_load_nt(block_base, xy_byte + (uint32_t)p * byte_stride);

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = -regs[p];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        saddr_store_nt(block_base, xy_byte + (uint32_t)p * byte_stride, regs[p]);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddr(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletSaddrKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Scalar baseline (nontemporal): one float per load, 64-bit addressing, nt hints.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletScalarNTBaselineKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;
    int gz  = blockIdx.z * 32;

    size_t base_idx = gx + gy * ldimx + (size_t)gz * ldimxy;

    float regs[PLANES * 4];

    #pragma unroll
    for (int p = 0; p < PLANES; p++) {
        const float* ptr = &data[base_idx + (size_t)p * ldimxy];
        auto v = __builtin_nontemporal_load(
            reinterpret_cast<const float4_vec*>(ptr));
        regs[p * 4 + 0] = v[0];
        regs[p * 4 + 1] = v[1];
        regs[p * 4 + 2] = v[2];
        regs[p * 4 + 3] = v[3];
    }

    #pragma unroll
    for (int i = 0; i < PLANES * 4; i++)
        regs[i] = -regs[i];

    #pragma unroll
    for (int p = 0; p < PLANES; p++) {
        float* ptr = &data[base_idx + (size_t)p * ldimxy];
        float4_vec v = {regs[p*4+0], regs[p*4+1], regs[p*4+2], regs[p*4+3]};
        __builtin_nontemporal_store(v, reinterpret_cast<float4_vec*>(ptr));
    }
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformScalarNTBaseline(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletScalarNTBaselineKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Scalar baseline: one float per load, 64-bit addressing.
// Same thread mapping: each thread handles 4 x-values × 32 planes = 128 regs.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletScalarBaselineKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;
    int gz  = blockIdx.z * 32;

    size_t base_idx = gx + gy * ldimx + (size_t)gz * ldimxy;

    float regs[PLANES * 4];

    #pragma unroll
    for (int p = 0; p < PLANES; p++) {
        size_t plane_idx = base_idx + (size_t)p * ldimxy;
        regs[p * 4 + 0] = data[plane_idx + 0];
        regs[p * 4 + 1] = data[plane_idx + 1];
        regs[p * 4 + 2] = data[plane_idx + 2];
        regs[p * 4 + 3] = data[plane_idx + 3];
    }

    #pragma unroll
    for (int i = 0; i < PLANES * 4; i++)
        regs[i] = -regs[i];

    #pragma unroll
    for (int p = 0; p < PLANES; p++) {
        size_t plane_idx = base_idx + (size_t)p * ldimxy;
        data[plane_idx + 0] = regs[p * 4 + 0];
        data[plane_idx + 1] = regs[p * 4 + 1];
        data[plane_idx + 2] = regs[p * 4 + 2];
        data[plane_idx + 3] = regs[p * 4 + 3];
    }
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformScalarBaseline(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletScalarBaselineKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ds79_forward_f4_scalar_tmp is now in ds79.h (shared).

// ---------------------------------------------------------------------------
// Buffer kernel with Z-direction wavelet transform (float4_vec tmp).
// ds79_forward_f4: float4_vec tmp[32] = 128 VGPRs for tmp.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletBufferZKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;
    constexpr int SLC = 2;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        block_base, 0, -1, 0x00027000);

    int byte_off    = (gx + gy * ldimx) * (int)sizeof(float);
    int byte_stride = ldimxy * (int)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, p * byte_stride, SLC));

    ds79_forward_f4(regs, PLANES);

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        __builtin_amdgcn_raw_buffer_store_b128(
            __builtin_bit_cast(uint4_vec, regs[p]),
            rsrc, byte_off, p * byte_stride, SLC);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZ(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletBufferZKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Buffer kernel with Z-direction wavelet transform (scalar tmp).
// ds79_forward_f4_scalar_tmp: float tmp[32] = 32 VGPRs for tmp, reused
// across 4 components sequentially.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletBufferZScalarTmpKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;
    constexpr int SLC = 2;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        block_base, 0, -1, 0x00027000);

    int byte_off    = (gx + gy * ldimx) * (int)sizeof(float);
    int byte_stride = ldimxy * (int)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, p * byte_stride, SLC));

    ds79_forward_f4_scalar_tmp(regs, PLANES);

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        __builtin_amdgcn_raw_buffer_store_b128(
            __builtin_bit_cast(uint4_vec, regs[p]),
            rsrc, byte_off, p * byte_stride, SLC);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZScalarTmp(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletBufferZScalarTmpKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Buffer Z+Y kernel: Z-transform in registers, then Y-transform via LDS.
// 8 planes per LDS batch (256 threads = 8 planes × 32 x-positions).
// Simple LDS layout: LDS[plane][x][y], no padding, no XOR.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletBufferZYKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;
    constexpr int BATCH = 8;
    constexpr int SLC = 2;

    __shared__ float lds[BATCH * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        block_base, 0, -1, 0x00027000);

    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    int byte_off    = (gx + gy * ldimx) * (int)sizeof(float);
    int byte_stride = ldimxy * (int)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, p * byte_stride, SLC));

    ds79_forward_f4_scalar_tmp(regs, PLANES);

    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v = regs[pb + dp];
            lds[dp * 1024 + (xg * 4 + 0) * 32 + yr] = v[0];
            lds[dp * 1024 + (xg * 4 + 1) * 32 + yr] = v[1];
            lds[dp * 1024 + (xg * 4 + 2) * 32 + yr] = v[2];
            lds[dp * 1024 + (xg * 4 + 3) * 32 + yr] = v[3];
        }
        __syncthreads();

        int pl = tid / 32;
        int xp = tid % 32;

        float yline[32];
        for (int y = 0; y < 32; y++)
            yline[y] = lds[pl * 1024 + xp * 32 + y];

        ds79_forward(yline, 32);

        for (int y = 0; y < 32; y++)
            lds[pl * 1024 + xp * 32 + y] = yline[y];

        __syncthreads();

        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v;
            v[0] = lds[dp * 1024 + (xg * 4 + 0) * 32 + yr];
            v[1] = lds[dp * 1024 + (xg * 4 + 1) * 32 + yr];
            v[2] = lds[dp * 1024 + (xg * 4 + 2) * 32 + yr];
            v[3] = lds[dp * 1024 + (xg * 4 + 3) * 32 + yr];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        __builtin_amdgcn_raw_buffer_store_b128(
            __builtin_bit_cast(uint4_vec, regs[p]),
            rsrc, byte_off, p * byte_stride, SLC);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZY(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletBufferZYKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// saddr workaround with Z-direction wavelet transform (float4_vec tmp).
// Uses uint8_t* + uint32_t offset trick for SGPR base + VGPR offset addressing.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletSaddrZKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    uint32_t xy_byte     = (uint32_t)(gx + gy * ldimx) * (uint32_t)sizeof(float);
    uint32_t byte_stride = (uint32_t)ldimxy * (uint32_t)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = saddr_load_nt(block_base, xy_byte + (uint32_t)p * byte_stride);

    ds79_forward_f4(regs, PLANES);

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        saddr_store_nt(block_base, xy_byte + (uint32_t)p * byte_stride, regs[p]);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZ(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletSaddrZKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// saddr workaround with Z-direction wavelet transform (scalar tmp).
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletSaddrZScalarTmpKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    uint32_t xy_byte     = (uint32_t)(gx + gy * ldimx) * (uint32_t)sizeof(float);
    uint32_t byte_stride = (uint32_t)ldimxy * (uint32_t)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = saddr_load_nt(block_base, xy_byte + (uint32_t)p * byte_stride);

    ds79_forward_f4_scalar_tmp(regs, PLANES);

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        saddr_store_nt(block_base, xy_byte + (uint32_t)p * byte_stride, regs[p]);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZScalarTmp(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletSaddrZScalarTmpKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// saddr Z+Y kernel: Z-transform in registers, then Y-transform via LDS.
// 8 planes per LDS batch (256 threads = 8 planes × 32 x-positions).
// Simple LDS layout: LDS[plane][x][y], no padding, no XOR.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletSaddrZYKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;
    constexpr int BATCH = 8;

    __shared__ float lds[BATCH * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    uint32_t xy_byte     = (uint32_t)(gx + gy * ldimx) * (uint32_t)sizeof(float);
    uint32_t byte_stride = (uint32_t)ldimxy * (uint32_t)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = saddr_load_nt(block_base, xy_byte + (uint32_t)p * byte_stride);

    ds79_forward_f4_scalar_tmp(regs, PLANES);

    // Y-transform: 4 batches of 8 planes
    for (int pb = 0; pb < PLANES; pb += BATCH) {
        // Scatter to LDS: layout [plane_local][x_pos][y]
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v = regs[pb + dp];
            lds[dp * 1024 + (xg * 4 + 0) * 32 + yr] = v[0];
            lds[dp * 1024 + (xg * 4 + 1) * 32 + yr] = v[1];
            lds[dp * 1024 + (xg * 4 + 2) * 32 + yr] = v[2];
            lds[dp * 1024 + (xg * 4 + 3) * 32 + yr] = v[3];
        }
        __syncthreads();

        // Remap: tid/32 = plane (0..7), tid%32 = x_pos (0..31)
        int pl = tid / 32;
        int xp = tid % 32;

        float yline[32];
        for (int y = 0; y < 32; y++)
            yline[y] = lds[pl * 1024 + xp * 32 + y];

        ds79_forward(yline, 32);

        for (int y = 0; y < 32; y++)
            lds[pl * 1024 + xp * 32 + y] = yline[y];

        __syncthreads();

        // Gather back to registers
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v;
            v[0] = lds[dp * 1024 + (xg * 4 + 0) * 32 + yr];
            v[1] = lds[dp * 1024 + (xg * 4 + 1) * 32 + yr];
            v[2] = lds[dp * 1024 + (xg * 4 + 2) * 32 + yr];
            v[3] = lds[dp * 1024 + (xg * 4 + 3) * 32 + yr];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        saddr_store_nt(block_base, xy_byte + (uint32_t)p * byte_stride, regs[p]);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZY(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletSaddrZYKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Buffer Z+Y kernel with XOR bank-conflict avoidance.
// LDS index: x * 32 + (y ^ x) instead of x * 32 + y.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletBufferZYXorKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;
    constexpr int BATCH = 8;
    constexpr int SLC = 2;

    __shared__ float lds[BATCH * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        block_base, 0, -1, 0x00027000);

    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    int byte_off    = (gx + gy * ldimx) * (int)sizeof(float);
    int byte_stride = ldimxy * (int)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, p * byte_stride, SLC));

    ds79_forward_f4_scalar_tmp(regs, PLANES);

    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v = regs[pb + dp];
            int x0 = xg * 4;
            lds[dp * 1024 + (x0 + 0) * 32 + (yr ^ (x0 + 0))] = v[0];
            lds[dp * 1024 + (x0 + 1) * 32 + (yr ^ (x0 + 1))] = v[1];
            lds[dp * 1024 + (x0 + 2) * 32 + (yr ^ (x0 + 2))] = v[2];
            lds[dp * 1024 + (x0 + 3) * 32 + (yr ^ (x0 + 3))] = v[3];
        }
        __syncthreads();

        int pl = tid / 32;
        int xp = tid % 32;

        float yline[32];
        for (int y = 0; y < 32; y++)
            yline[y] = lds[pl * 1024 + xp * 32 + (y ^ xp)];

        ds79_forward(yline, 32);

        for (int y = 0; y < 32; y++)
            lds[pl * 1024 + xp * 32 + (y ^ xp)] = yline[y];

        __syncthreads();

        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v;
            int x0 = xg * 4;
            v[0] = lds[dp * 1024 + (x0 + 0) * 32 + (yr ^ (x0 + 0))];
            v[1] = lds[dp * 1024 + (x0 + 1) * 32 + (yr ^ (x0 + 1))];
            v[2] = lds[dp * 1024 + (x0 + 2) * 32 + (yr ^ (x0 + 2))];
            v[3] = lds[dp * 1024 + (x0 + 3) * 32 + (yr ^ (x0 + 3))];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        __builtin_amdgcn_raw_buffer_store_b128(
            __builtin_bit_cast(uint4_vec, regs[p]),
            rsrc, byte_off, p * byte_stride, SLC);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZYXor(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletBufferZYXorKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// saddr Z+Y kernel with XOR bank-conflict avoidance.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletSaddrZYXorKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;
    constexpr int BATCH = 8;

    __shared__ float lds[BATCH * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    uint32_t xy_byte     = (uint32_t)(gx + gy * ldimx) * (uint32_t)sizeof(float);
    uint32_t byte_stride = (uint32_t)ldimxy * (uint32_t)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = saddr_load_nt(block_base, xy_byte + (uint32_t)p * byte_stride);

    ds79_forward_f4_scalar_tmp(regs, PLANES);

    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v = regs[pb + dp];
            int x0 = xg * 4;
            lds[dp * 1024 + (x0 + 0) * 32 + (yr ^ (x0 + 0))] = v[0];
            lds[dp * 1024 + (x0 + 1) * 32 + (yr ^ (x0 + 1))] = v[1];
            lds[dp * 1024 + (x0 + 2) * 32 + (yr ^ (x0 + 2))] = v[2];
            lds[dp * 1024 + (x0 + 3) * 32 + (yr ^ (x0 + 3))] = v[3];
        }
        __syncthreads();

        int pl = tid / 32;
        int xp = tid % 32;

        float yline[32];
        for (int y = 0; y < 32; y++)
            yline[y] = lds[pl * 1024 + xp * 32 + (y ^ xp)];

        ds79_forward(yline, 32);

        for (int y = 0; y < 32; y++)
            lds[pl * 1024 + xp * 32 + (y ^ xp)] = yline[y];

        __syncthreads();

        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v;
            int x0 = xg * 4;
            v[0] = lds[dp * 1024 + (x0 + 0) * 32 + (yr ^ (x0 + 0))];
            v[1] = lds[dp * 1024 + (x0 + 1) * 32 + (yr ^ (x0 + 1))];
            v[2] = lds[dp * 1024 + (x0 + 2) * 32 + (yr ^ (x0 + 2))];
            v[3] = lds[dp * 1024 + (x0 + 3) * 32 + (yr ^ (x0 + 3))];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        saddr_store_nt(block_base, xy_byte + (uint32_t)p * byte_stride, regs[p]);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZYXor(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletSaddrZYXorKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// Grouped XOR: permute groups of 4, keep elements within a group consecutive.
// Enables ds_read/write_b128 while reducing bank conflicts from 32-way to 4-way.
__device__ __forceinline__
int xor4_idx(int x, int y) {
    return x * 32 + ((y & 3) | (((y >> 2) ^ (x >> 2)) << 2));
}

// ---------------------------------------------------------------------------
// Buffer Z+Y kernel with grouped XOR (b128-friendly) bank-conflict reduction.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletBufferZYXor4Kernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;
    constexpr int BATCH = 8;
    constexpr int SLC = 2;

    __shared__ float lds[BATCH * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        block_base, 0, -1, 0x00027000);

    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    int byte_off    = (gx + gy * ldimx) * (int)sizeof(float);
    int byte_stride = ldimxy * (int)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, p * byte_stride, SLC));

    ds79_forward_f4_scalar_tmp(regs, PLANES);

    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v = regs[pb + dp];
            int x0 = xg * 4;
            lds[dp * 1024 + xor4_idx(x0 + 0, yr)] = v[0];
            lds[dp * 1024 + xor4_idx(x0 + 1, yr)] = v[1];
            lds[dp * 1024 + xor4_idx(x0 + 2, yr)] = v[2];
            lds[dp * 1024 + xor4_idx(x0 + 3, yr)] = v[3];
        }
        __syncthreads();

        int pl = tid / 32;
        int xp = tid % 32;

        float yline[32];
        for (int y = 0; y < 32; y++)
            yline[y] = lds[pl * 1024 + xor4_idx(xp, y)];

        ds79_forward(yline, 32);

        for (int y = 0; y < 32; y++)
            lds[pl * 1024 + xor4_idx(xp, y)] = yline[y];

        __syncthreads();

        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v;
            int x0 = xg * 4;
            v[0] = lds[dp * 1024 + xor4_idx(x0 + 0, yr)];
            v[1] = lds[dp * 1024 + xor4_idx(x0 + 1, yr)];
            v[2] = lds[dp * 1024 + xor4_idx(x0 + 2, yr)];
            v[3] = lds[dp * 1024 + xor4_idx(x0 + 3, yr)];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        __builtin_amdgcn_raw_buffer_store_b128(
            __builtin_bit_cast(uint4_vec, regs[p]),
            rsrc, byte_off, p * byte_stride, SLC);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZYXor4(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletBufferZYXor4Kernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// saddr Z+Y kernel with grouped XOR (b128-friendly) bank-conflict reduction.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletSaddrZYXor4Kernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;
    constexpr int BATCH = 8;

    __shared__ float lds[BATCH * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    uint32_t xy_byte     = (uint32_t)(gx + gy * ldimx) * (uint32_t)sizeof(float);
    uint32_t byte_stride = (uint32_t)ldimxy * (uint32_t)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = saddr_load_nt(block_base, xy_byte + (uint32_t)p * byte_stride);

    ds79_forward_f4_scalar_tmp(regs, PLANES);

    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v = regs[pb + dp];
            int x0 = xg * 4;
            lds[dp * 1024 + xor4_idx(x0 + 0, yr)] = v[0];
            lds[dp * 1024 + xor4_idx(x0 + 1, yr)] = v[1];
            lds[dp * 1024 + xor4_idx(x0 + 2, yr)] = v[2];
            lds[dp * 1024 + xor4_idx(x0 + 3, yr)] = v[3];
        }
        __syncthreads();

        int pl = tid / 32;
        int xp = tid % 32;

        float yline[32];
        for (int y = 0; y < 32; y++)
            yline[y] = lds[pl * 1024 + xor4_idx(xp, y)];

        ds79_forward(yline, 32);

        for (int y = 0; y < 32; y++)
            lds[pl * 1024 + xor4_idx(xp, y)] = yline[y];

        __syncthreads();

        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v;
            int x0 = xg * 4;
            v[0] = lds[dp * 1024 + xor4_idx(x0 + 0, yr)];
            v[1] = lds[dp * 1024 + xor4_idx(x0 + 1, yr)];
            v[2] = lds[dp * 1024 + xor4_idx(x0 + 2, yr)];
            v[3] = lds[dp * 1024 + xor4_idx(x0 + 3, yr)];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        saddr_store_nt(block_base, xy_byte + (uint32_t)p * byte_stride, regs[p]);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZYXor4(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletSaddrZYXor4Kernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Buffer Z+Y+X kernel: fused Y+X in LDS with per-element XOR.
// After Y writes back to LDS, X reads directly from same LDS (no reg
// round-trip). XOR mapping x*32 + (y^x) gives 0 bank conflicts for both
// Y-line and X-line access.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletBufferZYXKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;
    constexpr int BATCH = 8;
    constexpr int SLC = 2;

    __shared__ float lds[BATCH * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        block_base, 0, -1, 0x00027000);

    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    int byte_off    = (gx + gy * ldimx) * (int)sizeof(float);
    int byte_stride = ldimxy * (int)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, p * byte_stride, SLC));

    ds79_forward_f4_scalar_tmp(regs, PLANES);

    for (int pb = 0; pb < PLANES; pb += BATCH) {
        // Scatter to LDS: layout [plane][x][ y ^ x ]
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v = regs[pb + dp];
            int x0 = xg * 4;
            lds[dp * 1024 + (x0 + 0) * 32 + (yr ^ (x0 + 0))] = v[0];
            lds[dp * 1024 + (x0 + 1) * 32 + (yr ^ (x0 + 1))] = v[1];
            lds[dp * 1024 + (x0 + 2) * 32 + (yr ^ (x0 + 2))] = v[2];
            lds[dp * 1024 + (x0 + 3) * 32 + (yr ^ (x0 + 3))] = v[3];
        }
        __syncthreads();

        int pl  = tid / 32;
        int pos = tid % 32;

        // Y-transform: read y-line at x=pos
        float line[32];
        for (int y = 0; y < 32; y++)
            line[y] = lds[pl * 1024 + pos * 32 + (y ^ pos)];

        ds79_forward_reg32(line);

        for (int y = 0; y < 32; y++)
            lds[pl * 1024 + pos * 32 + (y ^ pos)] = line[y];

        __syncthreads();

        // X-transform: read x-line at y=pos from same LDS
        for (int x = 0; x < 32; x++)
            line[x] = lds[pl * 1024 + x * 32 + (pos ^ x)];

        ds79_forward_reg32(line);

        for (int x = 0; x < 32; x++)
            lds[pl * 1024 + x * 32 + (pos ^ x)] = line[x];

        __syncthreads();

        // Gather back to registers
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v;
            int x0 = xg * 4;
            v[0] = lds[dp * 1024 + (x0 + 0) * 32 + (yr ^ (x0 + 0))];
            v[1] = lds[dp * 1024 + (x0 + 1) * 32 + (yr ^ (x0 + 1))];
            v[2] = lds[dp * 1024 + (x0 + 2) * 32 + (yr ^ (x0 + 2))];
            v[3] = lds[dp * 1024 + (x0 + 3) * 32 + (yr ^ (x0 + 3))];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        __builtin_amdgcn_raw_buffer_store_b128(
            __builtin_bit_cast(uint4_vec, regs[p]),
            rsrc, byte_off, p * byte_stride, SLC);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZYX(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletBufferZYXKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// saddr Z+Y+X kernel: fused Y+X in LDS with per-element XOR.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletSaddrZYXKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int PLANES = 32;
    constexpr int BATCH = 8;

    __shared__ float lds[BATCH * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    uint32_t xy_byte     = (uint32_t)(gx + gy * ldimx) * (uint32_t)sizeof(float);
    uint32_t byte_stride = (uint32_t)ldimxy * (uint32_t)sizeof(float);

    float4_vec regs[PLANES];

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = saddr_load_nt(block_base, xy_byte + (uint32_t)p * byte_stride);

    ds79_forward_f4_scalar_tmp(regs, PLANES);

    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v = regs[pb + dp];
            int x0 = xg * 4;
            lds[dp * 1024 + (x0 + 0) * 32 + (yr ^ (x0 + 0))] = v[0];
            lds[dp * 1024 + (x0 + 1) * 32 + (yr ^ (x0 + 1))] = v[1];
            lds[dp * 1024 + (x0 + 2) * 32 + (yr ^ (x0 + 2))] = v[2];
            lds[dp * 1024 + (x0 + 3) * 32 + (yr ^ (x0 + 3))] = v[3];
        }
        __syncthreads();

        int pl  = tid / 32;
        int pos = tid % 32;

        float line[32];
        for (int y = 0; y < 32; y++)
            line[y] = lds[pl * 1024 + pos * 32 + (y ^ pos)];

        ds79_forward_reg32(line);

        for (int y = 0; y < 32; y++)
            lds[pl * 1024 + pos * 32 + (y ^ pos)] = line[y];

        __syncthreads();

        for (int x = 0; x < 32; x++)
            line[x] = lds[pl * 1024 + x * 32 + (pos ^ x)];

        ds79_forward_reg32(line);

        for (int x = 0; x < 32; x++)
            lds[pl * 1024 + x * 32 + (pos ^ x)] = line[x];

        __syncthreads();

        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v;
            int x0 = xg * 4;
            v[0] = lds[dp * 1024 + (x0 + 0) * 32 + (yr ^ (x0 + 0))];
            v[1] = lds[dp * 1024 + (x0 + 1) * 32 + (yr ^ (x0 + 1))];
            v[2] = lds[dp * 1024 + (x0 + 2) * 32 + (yr ^ (x0 + 2))];
            v[3] = lds[dp * 1024 + (x0 + 3) * 32 + (yr ^ (x0 + 3))];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        saddr_store_nt(block_base, xy_byte + (uint32_t)p * byte_stride, regs[p]);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZYX(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletSaddrZYXKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Pipelined XYZ helpers: scatter batch to LDS, XY transform, gather back.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void pipe_scatter_xor(float4_vec* batch, float* lds, int xg, int yr) {
    int x0 = xg * 4;
    #pragma unroll
    for (int dp = 0; dp < 8; dp++) {
        float4_vec v = batch[dp];
        lds[dp * 1024 + (x0 + 0) * 32 + (yr ^ (x0 + 0))] = v[0];
        lds[dp * 1024 + (x0 + 1) * 32 + (yr ^ (x0 + 1))] = v[1];
        lds[dp * 1024 + (x0 + 2) * 32 + (yr ^ (x0 + 2))] = v[2];
        lds[dp * 1024 + (x0 + 3) * 32 + (yr ^ (x0 + 3))] = v[3];
    }
}

__device__ __forceinline__
void pipe_xy_transform(float* lds, int tid) {
    int pl  = tid / 32;
    int pos = tid % 32;

    float line[32];
    for (int y = 0; y < 32; y++)
        line[y] = lds[pl * 1024 + pos * 32 + (y ^ pos)];
    ds79_forward_reg32(line);
    for (int y = 0; y < 32; y++)
        lds[pl * 1024 + pos * 32 + (y ^ pos)] = line[y];

    __syncthreads();

    for (int x = 0; x < 32; x++)
        line[x] = lds[pl * 1024 + x * 32 + (pos ^ x)];
    ds79_forward_reg32(line);
    for (int x = 0; x < 32; x++)
        lds[pl * 1024 + x * 32 + (pos ^ x)] = line[x];
}

__device__ __forceinline__
void pipe_gather_xor(float4_vec* batch, float* lds, int xg, int yr) {
    int x0 = xg * 4;
    #pragma unroll
    for (int dp = 0; dp < 8; dp++) {
        float4_vec v;
        v[0] = lds[dp * 1024 + (x0 + 0) * 32 + (yr ^ (x0 + 0))];
        v[1] = lds[dp * 1024 + (x0 + 1) * 32 + (yr ^ (x0 + 1))];
        v[2] = lds[dp * 1024 + (x0 + 2) * 32 + (yr ^ (x0 + 2))];
        v[3] = lds[dp * 1024 + (x0 + 3) * 32 + (yr ^ (x0 + 3))];
        batch[dp] = v;
    }
}

// ---------------------------------------------------------------------------
// Pipelined XYZ kernel (buffer): overlaps global loads with LDS XY transforms.
// Uses separate b0..b3 arrays so the compiler can track per-batch liveness.
// After XY, copies to flat regs[32] for Z-transform.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletBufferPipeXYZKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int SLC = 2;

    __shared__ float lds[8 * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        block_base, 0, -1, 0x00027000);

    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    int byte_off    = (gx + gy * ldimx) * (int)sizeof(float);
    int byte_stride = ldimxy * (int)sizeof(float);

    float4_vec regs[32];

    // Load batch 0 into regs[0..7]
    #pragma unroll
    for (int i = 0; i < 8; i++)
        regs[i] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, i * byte_stride, SLC));

    // Batch 0: scatter regs[0..7] → LDS, prefetch regs[8..15], XY, gather → regs[0..7]
    pipe_scatter_xor(&regs[0], lds, xg, yr);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; i++)
        regs[8 + i] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, (8 + i) * byte_stride, SLC));
    pipe_xy_transform(lds, tid);
    __syncthreads();
    pipe_gather_xor(&regs[0], lds, xg, yr);
    __syncthreads();

    // Batch 1: scatter regs[8..15] → LDS, prefetch regs[16..23], XY, gather → regs[8..15]
    pipe_scatter_xor(&regs[8], lds, xg, yr);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; i++)
        regs[16 + i] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, (16 + i) * byte_stride, SLC));
    pipe_xy_transform(lds, tid);
    __syncthreads();
    pipe_gather_xor(&regs[8], lds, xg, yr);
    __syncthreads();

    // Batch 2: scatter regs[16..23] → LDS, prefetch regs[24..31], XY, gather → regs[16..23]
    pipe_scatter_xor(&regs[16], lds, xg, yr);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; i++)
        regs[24 + i] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, (24 + i) * byte_stride, SLC));
    pipe_xy_transform(lds, tid);
    __syncthreads();
    pipe_gather_xor(&regs[16], lds, xg, yr);
    __syncthreads();

    // Batch 3: scatter regs[24..31] → LDS, XY, gather → regs[24..31] (no prefetch)
    pipe_scatter_xor(&regs[24], lds, xg, yr);
    __syncthreads();
    pipe_xy_transform(lds, tid);
    __syncthreads();
    pipe_gather_xor(&regs[24], lds, xg, yr);

    ds79_forward_f4_scalar_tmp(regs, 32);

    #pragma unroll
    for (int p = 0; p < 32; p++)
        __builtin_amdgcn_raw_buffer_store_b128(
            __builtin_bit_cast(uint4_vec, regs[p]),
            rsrc, byte_off, p * byte_stride, SLC);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferPipeXYZ(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletBufferPipeXYZKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Pipelined XYZ kernel (saddr): same split-batch approach with saddr loads.
// ---------------------------------------------------------------------------
template<int T, int TargetOcc>
__launch_bounds__(T, TargetOcc)
__global__ void waveletSaddrPipeXYZKernel(
    float* __restrict__ data,
    int ldimx,
    int ldimxy)
{
    static_assert(T == 256, "T must be 256");
    constexpr int SLC = 2;

    __shared__ float lds[8 * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    uint32_t xy_byte     = (uint32_t)(gx + gy * ldimx) * (uint32_t)sizeof(float);
    uint32_t byte_stride = (uint32_t)ldimxy * (uint32_t)sizeof(float);

    float4_vec regs[32];

    #pragma unroll
    for (int i = 0; i < 8; i++)
        regs[i] = saddr_load_nt(block_base, xy_byte + (uint32_t)i * byte_stride);

    // Batch 0
    pipe_scatter_xor(&regs[0], lds, xg, yr);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; i++)
        regs[8 + i] = saddr_load_nt(block_base, xy_byte + (uint32_t)(8 + i) * byte_stride);
    pipe_xy_transform(lds, tid);
    __syncthreads();
    pipe_gather_xor(&regs[0], lds, xg, yr);
    __syncthreads();

    // Batch 1
    pipe_scatter_xor(&regs[8], lds, xg, yr);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; i++)
        regs[16 + i] = saddr_load_nt(block_base, xy_byte + (uint32_t)(16 + i) * byte_stride);
    pipe_xy_transform(lds, tid);
    __syncthreads();
    pipe_gather_xor(&regs[8], lds, xg, yr);
    __syncthreads();

    // Batch 2
    pipe_scatter_xor(&regs[16], lds, xg, yr);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; i++)
        regs[24 + i] = saddr_load_nt(block_base, xy_byte + (uint32_t)(24 + i) * byte_stride);
    pipe_xy_transform(lds, tid);
    __syncthreads();
    pipe_gather_xor(&regs[16], lds, xg, yr);
    __syncthreads();

    // Batch 3
    pipe_scatter_xor(&regs[24], lds, xg, yr);
    __syncthreads();
    pipe_xy_transform(lds, tid);
    __syncthreads();
    pipe_gather_xor(&regs[24], lds, xg, yr);

    ds79_forward_f4_scalar_tmp(regs, 32);

    #pragma unroll
    for (int p = 0; p < 32; p++)
        saddr_store_nt(block_base, xy_byte + (uint32_t)p * byte_stride, regs[p]);
}

template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrPipeXYZ(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy)
{
    dim3 block(T);
    dim3 grid(nx / bx, ny / by, nz / bz);
    hipLaunchKernelGGL((waveletSaddrPipeXYZKernel<T, TargetOcc>),
                       grid, block, 0, 0, data, ldimx, ldimxy);
    return hipGetLastError();
}

// Explicit instantiations
template __global__ void waveletBufferKernel<256, 3>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformBuffer<256, 3>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletBaselineKernel<256, 3>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformBaseline<256, 3>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletScalarBaselineKernel<256, 3>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformScalarBaseline<256, 3>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletScalarNTBaselineKernel<256, 3>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformScalarNTBaseline<256, 3>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletSaddrKernel<256, 3>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformSaddr<256, 3>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletBufferZKernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformBufferZ<256, 1>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletBufferZScalarTmpKernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformBufferZScalarTmp<256, 1>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletSaddrZKernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformSaddrZ<256, 1>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletSaddrZScalarTmpKernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformSaddrZScalarTmp<256, 1>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletBufferZYKernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformBufferZY<256, 1>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletSaddrZYKernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformSaddrZY<256, 1>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletBufferZYXorKernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformBufferZYXor<256, 1>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletSaddrZYXorKernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformSaddrZYXor<256, 1>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletBufferZYXor4Kernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformBufferZYXor4<256, 1>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletSaddrZYXor4Kernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformSaddrZYXor4<256, 1>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletBufferZYXKernel<256, 2>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformBufferZYX<256, 2>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletSaddrZYXKernel<256, 2>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformSaddrZYX<256, 2>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletBufferPipeXYZKernel<256, 2>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformBufferPipeXYZ<256, 2>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletSaddrPipeXYZKernel<256, 2>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformSaddrPipeXYZ<256, 2>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletBufferPipeXYZKernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformBufferPipeXYZ<256, 1>(float*, int, int, int, int, int, int, int, int);

template __global__ void waveletSaddrPipeXYZKernel<256, 1>(float* __restrict__, int, int);
template hipError_t hipWaveletTransformSaddrPipeXYZ<256, 1>(float*, int, int, int, int, int, int, int, int);
