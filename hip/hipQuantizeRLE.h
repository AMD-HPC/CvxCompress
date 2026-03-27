// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIPQUANTIZE_RLE_H
#define HIPQUANTIZE_RLE_H

// DEPRECATED: Development/benchmark file, not used by the production API.
// Superseded by the fused wavelet+RLE kernel in hipWaveletRLE.h.
// Contains known int overflow issues for large lateral dims. Will be removed.
//
// GPU z-line quantize + RLE encode kernel (two-pass compacted, zero scratch).
//
// Thread block: 256 threads (1D), same layout as the buffer wavelet kernel.
//   tid % 8  -> x-group (loads 4 consecutive x via float4)
//   tid / 8  -> y-row (32 rows)
// Each thread loads 32 float4 values via buffer instructions, then encodes
// 4 z-lines independently using the CvxCompress escape code format.
//
// Per x_off (processes one of 4 z-lines at a time):
//   1. Dry encode (ALU only) → byte count per thread.
//   2. Blelloch prefix sum on byte counts in LDS → compacted offsets.
//   3. Wet encode directly to compacted LDS positions (safe writes only).
//   4. Coalesced dword store from LDS to global.
//
// No scratch memory, no LDS slots, no register readback.
// 32 KB LDS → occupancy 2 on gfx942.
//
// Output: one contiguous encoded byte stream per block.
// block_sizes[block_id] = total encoded bytes for the block.
// Grid: dim3(nbx, nby, nbz) — one thread block per 32x32x32 wavelet block.

#include "Run_Length_Escape_Codes.hxx"
#include <hip/hip_runtime.h>
#include <rocprim/block/block_scan.hpp>

using qrle_float4_vec = __attribute__((__vector_size__(4 * sizeof(float)))) float;

static constexpr int QRLE_LDS_BYTES = 32768;

// WRITE=false: count-only (pure ALU, dst unused).
// WRITE=true, SAFE=false: writes all bytes including trailing junk (for padded slots).
// WRITE=true, SAFE=true: writes only valid bytes (no overflow past bp).
template<bool WRITE, bool SAFE = false>
__device__ __forceinline__
int qrle_zline(const qrle_float4_vec* planes, int x_off, float scale,
               unsigned char* dst)
{
    int bp = 0, rle = 0;
    #pragma unroll
    for (int z = 0; z < 32; ++z) {
        float fval = scale * planes[z][x_off];
        int ival = (int)fval;
        if (ival == 0) { ++rle; } else {
            if constexpr (!SAFE) {
                bool r1 = (rle == 1), rs = (rle >= 2) & (rle < 256);
                int rb = (r1 ? 1 : (rs ? 2 : 4)) & -(rle > 0);
                if constexpr (WRITE) {
                    dst[bp]   = r1 ? (unsigned char)0 : rs ? (unsigned char)(RLESC1&0xFF) : (unsigned char)(RLESC3&0xFF);
                    dst[bp+1] = rs ? (unsigned char)rle : (unsigned char)(rle&0xFF);
                    dst[bp+2] = (unsigned char)((rle>>8)&0xFF);
                    dst[bp+3] = (unsigned char)((rle>>16)&0xFF);
                }
                bp += rb;
            } else {
                if (rle == 1) {
                    dst[bp] = 0;
                    bp += 1;
                } else if (rle >= 2 && rle < 256) {
                    dst[bp] = (unsigned char)(RLESC1 & 0xFF);
                    dst[bp+1] = (unsigned char)rle;
                    bp += 2;
                } else if (rle >= 256) {
                    dst[bp] = (unsigned char)(RLESC3 & 0xFF);
                    dst[bp+1] = (unsigned char)(rle & 0xFF);
                    dst[bp+2] = (unsigned char)((rle >> 8) & 0xFF);
                    dst[bp+3] = (unsigned char)((rle >> 16) & 0xFF);
                    bp += 4;
                }
            }
            rle = 0;

            bool ib = (ival>VLESC2)&(ival<RLESC3), i16 = (ival>=-32768)&(ival<=32767);
            bool i24 = (ival>=-8388608)&(ival<=8388607), f32 = !ib&!i16&!i24;
            if constexpr (!SAFE) {
                if constexpr (WRITE) {
                    unsigned u; __builtin_memcpy(&u, &fval, 4);
                    dst[bp] = ib ? (unsigned char)(signed char)ival : i16 ? (unsigned char)(signed char)VLESC2
                            : i24 ? (unsigned char)(signed char)VLESC3 : (unsigned char)(signed char)VLESC4;
                    unsigned pay = f32 ? u : (unsigned)ival;
                    dst[bp+1]=(unsigned char)(pay&0xFF); dst[bp+2]=(unsigned char)((pay>>8)&0xFF);
                    dst[bp+3]=(unsigned char)((pay>>16)&0xFF); dst[bp+4]=(unsigned char)((pay>>24)&0xFF);
                }
            } else {
                unsigned u; __builtin_memcpy(&u, &fval, 4);
                unsigned pay = f32 ? u : (unsigned)ival;
                if (ib) {
                    dst[bp] = (unsigned char)(signed char)ival;
                } else if (i16) {
                    dst[bp] = (unsigned char)(signed char)VLESC2;
                    dst[bp+1] = (unsigned char)(pay & 0xFF);
                    dst[bp+2] = (unsigned char)((pay >> 8) & 0xFF);
                } else if (i24) {
                    dst[bp] = (unsigned char)(signed char)VLESC3;
                    dst[bp+1] = (unsigned char)(pay & 0xFF);
                    dst[bp+2] = (unsigned char)((pay >> 8) & 0xFF);
                    dst[bp+3] = (unsigned char)((pay >> 16) & 0xFF);
                } else {
                    dst[bp] = (unsigned char)(signed char)VLESC4;
                    dst[bp+1] = (unsigned char)(pay & 0xFF);
                    dst[bp+2] = (unsigned char)((pay >> 8) & 0xFF);
                    dst[bp+3] = (unsigned char)((pay >> 16) & 0xFF);
                    dst[bp+4] = (unsigned char)((pay >> 24) & 0xFF);
                }
            }
            bp += ib ? 1 : i16 ? 3 : i24 ? 4 : 5;
        }
    }
    if constexpr (!SAFE) {
        { bool r1=(rle==1),rs=(rle>=2)&(rle<256); int rb=(r1?1:(rs?2:4))&-(rle>0);
          if constexpr (WRITE) {
              dst[bp]=r1?(unsigned char)0:rs?(unsigned char)(RLESC1&0xFF):(unsigned char)(RLESC3&0xFF);
              dst[bp+1]=rs?(unsigned char)rle:(unsigned char)(rle&0xFF);
              dst[bp+2]=(unsigned char)((rle>>8)&0xFF); dst[bp+3]=(unsigned char)((rle>>16)&0xFF);
          }
          bp+=rb; }
    } else {
        if (rle == 1) {
            dst[bp] = 0;
            bp += 1;
        } else if (rle >= 2 && rle < 256) {
            dst[bp] = (unsigned char)(RLESC1 & 0xFF);
            dst[bp+1] = (unsigned char)rle;
            bp += 2;
        } else if (rle >= 256) {
            dst[bp] = (unsigned char)(RLESC3 & 0xFF);
            dst[bp+1] = (unsigned char)(rle & 0xFF);
            dst[bp+2] = (unsigned char)((rle >> 8) & 0xFF);
            dst[bp+3] = (unsigned char)((rle >> 16) & 0xFF);
            bp += 4;
        }
    }
    return bp;
}

__launch_bounds__(256, 2)
__global__ void quantizeRLEKernel(
    float* __restrict__ input,
    unsigned char* __restrict__ output,
    int* __restrict__ block_sizes,
    float scale,
    int ldimx, int ldimxy)
{
    constexpr int SLC = 2;
    constexpr int NTHREADS = 256;
    using BlockScan = rocprim::block_scan<int, NTHREADS>;

    __shared__ union {
        typename BlockScan::storage_type scan;
        unsigned char compact[QRLE_LDS_BYTES];
    } lds;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;

    float* bb = input + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(bb, 0, -1, 0x00027000);
    int boff = (gx + gy * ldimx) * (int)sizeof(float);
    int bstr = ldimxy * (int)sizeof(float);

    qrle_float4_vec planes[32];
    #pragma unroll
    for (int p = 0; p < 32; p++)
        planes[p] = __builtin_bit_cast(qrle_float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(rsrc, boff, p * bstr, SLC));

    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    unsigned char* block_out = output + (long)bid * 4 * QRLE_LDS_BYTES;
    int block_total = 0;

    for (int x_off = 0; x_off < 4; ++x_off) {

        // 1. Dry encode: ALU only, get byte count
        int my_count = qrle_zline<false>(planes, x_off, scale, nullptr);

        // 2. Exclusive prefix sum (DPP-based via rocprim)
        int my_offset, pass_total;
        BlockScan().exclusive_scan(my_count, my_offset, 0, pass_total, lds.scan);
        __syncthreads();

        // 3. Wet encode: write directly to compacted LDS position (safe, no overflow)
        qrle_zline<true, true>(planes, x_off, scale, lds.compact + my_offset);
        __syncthreads();

        // 4. Coalesced dword writes from LDS to global
        int aligned_total = (pass_total + 3) & ~3;
        for (int off = tid * 4; off < aligned_total; off += NTHREADS * 4) {
            unsigned val = 0;
            if (off < pass_total) {
                val  = (unsigned)lds.compact[off];
                if (off+1 < pass_total) val |= (unsigned)lds.compact[off+1] << 8;
                if (off+2 < pass_total) val |= (unsigned)lds.compact[off+2] << 16;
                if (off+3 < pass_total) val |= (unsigned)lds.compact[off+3] << 24;
            }
            if (off < aligned_total) {
                unsigned* dst32 = (unsigned*)(block_out + block_total + off);
                *dst32 = val;
            }
        }
        __syncthreads();

        block_total += pass_total;
    }

    if (tid == 0)
        block_sizes[bid] = block_total;
}

inline hipError_t hipQuantizeRLEEncode(
    float* input,
    unsigned char* output,
    int* block_sizes,
    float scale,
    int nx, int ny, int nz,
    int ldimx, int ldimxy)
{
    dim3 grid((nx + 31) / 32, (ny + 31) / 32, (nz + 31) / 32);
    quantizeRLEKernel<<<grid, dim3(256)>>>(
        input, output, block_sizes, scale, ldimx, ldimxy);
    return hipGetLastError();
}

#endif // HIPQUANTIZE_RLE_H
