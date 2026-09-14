// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIPWAVELET_BITMAP_H
#define HIPWAVELET_BITMAP_H

// PROTOTYPE: significance-map (bitmap) + packed-value split of the fused
// wavelet+quantize encode.  This is kernel 1 of a two-kernel design:
//
//   kernel 1 (this file): wavelet ZYX + quantize -> per block:
//       [ 4096 B bitmap : 1024 x uint32, one significance word per z-line ]
//       [ packed int32 nonzero quantized values, in (x_off,tid,z) order    ]
//     block_sizes[bid] = 4096 + nnz(block)*4
//     per-z-line nnz    = __popc(bitmap word)  (no separate metadata needed)
//
//   kernel 2 (coding, separate): consumes bitmap + packed values and emits
//     the final coded stream (fixed-width / width-class / bit-plane).
//
// Phases 1-3 load the field, apply the Z/Y/X transform, and quantize it before
// building the significance bitmap and packed-value stream.

#include <hip/hip_runtime.h>
#include <rocprim/block/block_scan.hpp>
#include "ds79.h"
#include "hipPlaneIO.h"

using hipcvx_wavelet_float4 = ds79_float4_vec;

// One significance word (32 bits, one per z) per z-line; 4*256 z-lines/block.
static constexpr int  WBMP_BITMAP_WORDS = 1024;
static constexpr int  WBMP_BITMAP_BYTES = WBMP_BITMAP_WORDS * 4;      // 4096
// Worst case (all 32768 coefficients nonzero) value region.
static constexpr int  WBMP_MAX_VAL_BYTES = 32768 * 4;                 // 131072
static constexpr long WBMP_SLOT_BYTES     = WBMP_BITMAP_BYTES + WBMP_MAX_VAL_BYTES; // 135168

__host__ __device__ __forceinline__ int hipcvx_quantize_i32(float value)
{
#if defined(__HIP_DEVICE_COMPILE__)
    if (__builtin_isnan(value)) return 0;
    // Clamp with one V_MED3_F32 before the conversion.  The upper endpoint is
    // the largest float below 2^31, so every converted value is representable.
    float clamped = __builtin_amdgcn_fmed3f(
        value, -2147483648.0f, 2147483520.0f);
    return (int)clamped;
#else
    if (!(value == value)) return 0;  // deterministic NaN handling
    if (value >= 2147483520.0f) return 2147483520;
    if (value <= -2147483648.0f) return (-2147483647 - 1);
    return (int)value;
#endif
}

__host__ __device__ __forceinline__ unsigned hipcvx_abs_i32(int value)
{
    return value < 0 ? 0u - (unsigned)value : (unsigned)value;
}

__launch_bounds__(256, 2)
__global__ void hipcvx_waveletBitmapFusedKernel(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    size_t* __restrict__ block_sizes,
    float scale,
    int ldimx, int ldimxy,
    const double* __restrict__ d_rms,
    float* __restrict__ d_mulfac_out)
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
    int xg  = tid % 8;
    int yr  = tid / 8;

    float mulfac;
    if (d_rms != nullptr) {
        float rms = (float)*d_rms;
        float product = rms * scale;
        mulfac = (product > 0.0f && __builtin_isfinite(1.0f / product))
                 ? (1.0f / product) : 1.0f;
    } else {
        mulfac = scale;
    }
    if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        if (d_mulfac_out) *d_mulfac_out = mulfac;
    }

    const float* block_base = input + (size_t)blockIdx.z * 32 * ldimxy;

    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    size_t byte_off = ((size_t)gy * ldimx + gx) * sizeof(float);

    // ---- Phase 1: Load 32 planes from global ----
    hipcvx_wavelet_float4 regs[PLANES];
    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = hipPlaneLoadNT<hipcvx_wavelet_float4>(
            block_base + (size_t)p * ldimxy, byte_off);

    // ---- Phase 2: Z-transform in registers ----
    ds79_forward_f4_scalar_tmp(regs, PLANES);

    // ---- Phase 3: Y+X transform in LDS (batches of 8) ----
    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            hipcvx_wavelet_float4 v = regs[pb + dp];
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
        for (int y = 0; y < 32; y++)
            line[y] = lds.wavelet[pl * 1024 + pos * 32 + (y ^ pos)];
        ds79_forward_reg32(line);
        for (int y = 0; y < 32; y++)
            lds.wavelet[pl * 1024 + pos * 32 + (y ^ pos)] = line[y];
        __syncthreads();

        for (int x = 0; x < 32; x++)
            line[x] = lds.wavelet[pl * 1024 + x * 32 + (pos ^ x)];
        ds79_forward_reg32(line);
        for (int x = 0; x < 32; x++)
            lds.wavelet[pl * 1024 + x * 32 + (pos ^ x)] = line[x];
        __syncthreads();

        for (int dp = 0; dp < BATCH; dp++) {
            hipcvx_wavelet_float4 v;
            int x0 = xg * 4;
            v[0] = lds.wavelet[dp * 1024 + (x0+0) * 32 + (yr ^ (x0+0))];
            v[1] = lds.wavelet[dp * 1024 + (x0+1) * 32 + (yr ^ (x0+1))];
            v[2] = lds.wavelet[dp * 1024 + (x0+2) * 32 + (yr ^ (x0+2))];
            v[3] = lds.wavelet[dp * 1024 + (x0+3) * 32 + (yr ^ (x0+3))];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    // ---- Phase 4: Quantize -> bitmap + packed nonzero values ----
    // Block layout: [4096B bitmap] [packed int32 values]
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    unsigned char* block_out = output + (long)bid * WBMP_SLOT_BYTES;
    uint32_t* bitmap_out = reinterpret_cast<uint32_t*>(block_out);
    int32_t*  val_out    = reinterpret_cast<int32_t*>(block_out + WBMP_BITMAP_BYTES);

    int block_val_base = 0;
    for (int x_off = 0; x_off < 4; ++x_off) {
        // Build 32-bit significance mask for this z-line.
        uint32_t mask = 0;
        #pragma unroll
        for (int z = 0; z < 32; ++z) {
            int ival = hipcvx_quantize_i32(mulfac * regs[z][x_off]);
            if (ival != 0) mask |= (1u << z);
        }
        int nnz = __popc(mask);
        bitmap_out[x_off * 256 + tid] = mask;

        // Value-stream offset = prefix sum of per-z-line nnz across the block.
        int my_off, pass_total;
        BlockScan().exclusive_scan(nnz, my_off, 0, pass_total, lds.scan);
        __syncthreads();

        // Scatter this z-line's nonzero values into the packed region, in
        // z order.  rank(z) = popcount of set bits below z.
        int base = block_val_base + my_off;
        #pragma unroll
        for (int z = 0; z < 32; ++z) {
            if (mask & (1u << z)) {
                int ival = hipcvx_quantize_i32(mulfac * regs[z][x_off]);
                int rank = __popc(mask & ((1u << z) - 1));
                val_out[base + rank] = ival;
            }
        }

        block_val_base += pass_total;
        __syncthreads();
    }

    if (tid == 0)
        block_sizes[bid] = (size_t)WBMP_BITMAP_BYTES + (size_t)block_val_base * 4;
}

// Minimum signed byte width to hold [-maxabs, maxabs].
__host__ __device__ __forceinline__ int wbmp_width_bytes(unsigned maxabs) {
    if (maxabs <= 0x7f)      return 1;
    if (maxabs <= 0x7fff)    return 2;
    if (maxabs <= 0x7fffff)  return 3;
    return 4;
}

static constexpr int  WBMP_WTAB_BYTES     = 256;   // 2 bits * 1024 lines

// Wave64 block scan used by the parallel octree significance coder.
namespace wbmp_opt {

static constexpr int WBMP_OPT_WARP   = 64;
static constexpr int WBMP_OPT_NWARPS = 16;

template <int CTRL, int RMASK>
__device__ __forceinline__ int dpp_add(int x) {
    return x + __builtin_amdgcn_update_dpp(0, x, CTRL, RMASK, 0xf, false);
}

__device__ __forceinline__ int winc(int v) {
    v = dpp_add<0x111, 0xf>(v);
    v = dpp_add<0x112, 0xf>(v);
    v = dpp_add<0x114, 0xf>(v);
    v = dpp_add<0x118, 0xf>(v);
    v = dpp_add<0x142, 0xa>(v);
    v = dpp_add<0x143, 0xc>(v);
    return v;
}

__device__ __forceinline__ int block_exscan(int v, int& total, int* warp_part) {
    const int tid = threadIdx.x;
    const int lane = tid & (WBMP_OPT_WARP - 1);
    const int warp = tid >> 6;
    const int wincl = winc(v);
    const int wsum = __builtin_amdgcn_readlane(wincl, WBMP_OPT_WARP - 1);
    if (lane == WBMP_OPT_WARP - 1) warp_part[warp] = wsum;
    __syncthreads();
    if (warp == 0) {
        const int part = (lane < WBMP_OPT_NWARPS) ? warp_part[lane] : 0;
        const int incl = winc(part);
        if (lane < WBMP_OPT_NWARPS) warp_part[lane] = incl;
    }
    __syncthreads();
    total = warp_part[WBMP_OPT_NWARPS - 1];
    const int wprefix = (warp == 0) ? 0 : warp_part[warp - 1];
    return wprefix + wincl - v;
}

__device__ __forceinline__ void block_exscan2(
    int v0, int v1,
    int& ex0, int& ex1,
    int& total0, int& total1,
    int* warp_part)
{
    const int tid = threadIdx.x;
    const int lane = tid & (WBMP_OPT_WARP - 1);
    const int warp = tid >> 6;
    const int w0incl = winc(v0);
    const int w1incl = winc(v1);
    const int w0sum = __builtin_amdgcn_readlane(w0incl, WBMP_OPT_WARP - 1);
    const int w1sum = __builtin_amdgcn_readlane(w1incl, WBMP_OPT_WARP - 1);
    if (lane == WBMP_OPT_WARP - 1) {
        warp_part[warp] = w0sum;
        warp_part[warp + WBMP_OPT_NWARPS] = w1sum;
    }
    __syncthreads();
    if (warp == 0) {
        const int s0 = (lane < WBMP_OPT_NWARPS) ? warp_part[lane] : 0;
        const int s1 = (lane < WBMP_OPT_NWARPS)
                     ? warp_part[lane + WBMP_OPT_NWARPS] : 0;
        const int s0incl = winc(s0);
        const int s1incl = winc(s1);
        if (lane < WBMP_OPT_NWARPS) {
            warp_part[lane] = s0incl;
            warp_part[lane + WBMP_OPT_NWARPS] = s1incl;
        }
    }
    __syncthreads();
    total0 = warp_part[WBMP_OPT_NWARPS - 1];
    total1 = warp_part[2 * WBMP_OPT_NWARPS - 1];
    const int w0prefix = (warp == 0) ? 0 : warp_part[warp - 1];
    const int w1prefix = (warp == 0)
                       ? 0 : warp_part[warp + WBMP_OPT_NWARPS - 1];
    ex0 = w0prefix + w0incl - v0;
    ex1 = w1prefix + w1incl - v1;
}

}  // namespace wbmp_opt

// Launch helper for the fused transform and bitmap pass. output must have
// nblocks * WBMP_SLOT_BYTES bytes; block_sizes has nblocks entries.
inline hipError_t hipWaveletBitmapFused(
    const float* input,
    unsigned char* output,
    size_t* block_sizes,
    float scale,
    int nx, int ny, int nz,
    int ldimx, int ldimxy,
    const double* d_rms = nullptr,
    float* d_mulfac_out = nullptr,
    hipStream_t stream = 0)
{
    dim3 grid((nx + 31) / 32, (ny + 31) / 32, (nz + 31) / 32);
    hipcvx_waveletBitmapFusedKernel<<<grid, dim3(256), 0, stream>>>(
        input, output, block_sizes, scale, ldimx, ldimxy,
        d_rms, d_mulfac_out);
    return hipGetLastError();
}

#endif // HIPWAVELET_BITMAP_H
