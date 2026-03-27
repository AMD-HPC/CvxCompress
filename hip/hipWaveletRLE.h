// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIPWAVELET_RLE_H
#define HIPWAVELET_RLE_H

// Fused wavelet ZYX + quantize + RLE encode kernel.
//
// Single kernel: global read → Z-transform (regs) → Y+X-transform (LDS) →
//                quantize+RLE encode (two-pass, LDS) → coalesced global write.
//
// Eliminates the intermediate global write+read between wavelet and RLE.
// 32 KB LDS reused across phases: wavelet Y/X, then RLE scan+compact.
// Zero scratch. Buffer instructions for loads. Occupancy 2 on gfx942.

#include <hip/hip_runtime.h>
#include <rocprim/block/block_scan.hpp>
#include <rocprim/device/device_scan.hpp>
#include "ds79.h"
#include "Run_Length_Escape_Codes.hxx"

using wrle_float4_vec = ds79_float4_vec;
using wrle_uint4_vec  = __attribute__((__vector_size__(4 * sizeof(unsigned)))) unsigned;

static constexpr int WRLE_LDS_BYTES = 32768;

// RLE z-line encode (same as hipQuantizeRLE.h).
// WRITE=false: count-only. WRITE=true, SAFE=true: no trailing junk.
template<bool WRITE>
__device__ __forceinline__
int wrle_zline(const wrle_float4_vec* planes, int x_off, float scale,
               unsigned char* dst)
{
    int bp = 0, rle = 0;
    #pragma unroll
    for (int z = 0; z < 32; ++z) {
        float fval = scale * planes[z][x_off];
        int ival = (int)fval;
        if (ival == 0) { ++rle; } else {
            if constexpr (!WRITE) {
                bool r1 = (rle == 1), rs = (rle >= 2) & (rle < 256);
                int rb = (r1 ? 1 : (rs ? 2 : 4)) & -(rle > 0);
                bp += rb;
            } else {
                if (rle == 1) {
                    dst[bp] = 0; bp += 1;
                } else if (rle >= 2 && rle < 256) {
                    dst[bp] = (unsigned char)(RLESC1 & 0xFF);
                    dst[bp+1] = (unsigned char)rle; bp += 2;
                } else if (rle >= 256) {
                    dst[bp] = (unsigned char)(RLESC3 & 0xFF);
                    dst[bp+1] = (unsigned char)(rle & 0xFF);
                    dst[bp+2] = (unsigned char)((rle >> 8) & 0xFF);
                    dst[bp+3] = (unsigned char)((rle >> 16) & 0xFF); bp += 4;
                }
            }
            rle = 0;
            bool ib = (ival>VLESC2)&(ival<RLESC3), i16 = (ival>=-32768)&(ival<=32767);
            bool i24 = (ival>=-8388608)&(ival<=8388607), f32 = !ib&!i16&!i24;
            if constexpr (!WRITE) {
                bp += ib ? 1 : i16 ? 3 : i24 ? 4 : 5;
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
                bp += ib ? 1 : i16 ? 3 : i24 ? 4 : 5;
            }
        }
    }
    if constexpr (!WRITE) {
        bool r1=(rle==1),rs=(rle>=2)&(rle<256);
        int rb=(r1?1:(rs?2:4))&-(rle>0);
        bp+=rb;
    } else {
        if (rle == 1) {
            dst[bp] = 0; bp += 1;
        } else if (rle >= 2 && rle < 256) {
            dst[bp] = (unsigned char)(RLESC1 & 0xFF);
            dst[bp+1] = (unsigned char)rle; bp += 2;
        } else if (rle >= 256) {
            dst[bp] = (unsigned char)(RLESC3 & 0xFF);
            dst[bp+1] = (unsigned char)(rle & 0xFF);
            dst[bp+2] = (unsigned char)((rle >> 8) & 0xFF);
            dst[bp+3] = (unsigned char)((rle >> 16) & 0xFF); bp += 4;
        }
    }
    return bp;
}

// Per-block z-line metadata: 4 passes × 256 z-lines × 1 byte = 1024 bytes.
// Stores the encoded byte count for each z-line, enabling parallel decoding.
static constexpr int WRLE_META_PER_BLOCK = 1024;

// Self-contained compressed stream header:
//   [4B num_blocks] [4B num_mulfacs] [8*num_blocks block_offsets] [4*num_mulfacs mulfacs]
// block_offsets[i] = byte offset of block i in the data region (after header).
inline int hipCompressHeaderSize(int num_blocks, int num_mulfacs)
{
    return 8 + 8 * num_blocks + 4 * num_mulfacs;
}

__launch_bounds__(256, 2)
__global__ void waveletRLEFusedKernel(
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
        unsigned char compact[WRLE_LDS_BYTES];
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
        if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            if (d_mulfac_out) *d_mulfac_out = mulfac;
        }
    } else {
        mulfac = scale;
    }

    const float* block_base = input + (size_t)blockIdx.z * 32 * ldimxy;

    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    uint32_t byte_off = (gx + gy * ldimx) * (uint32_t)sizeof(float);

    // ---- Phase 1: Load 32 planes from global ----
    wrle_float4_vec regs[PLANES];
    #pragma unroll
    for (int p = 0; p < PLANES; p++) {
        auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<float*>(block_base + (long)p * ldimxy),
            0, -1, 0x00027000);
        regs[p] = __builtin_bit_cast(wrle_float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, 0, SLC));
    }

    // ---- Phase 2: Z-transform in registers ----
    ds79_forward_f4_scalar_tmp(regs, PLANES);

    // ---- Phase 3: Y+X transform in LDS (batches of 8) ----
    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            wrle_float4_vec v = regs[pb + dp];
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
            wrle_float4_vec v;
            int x0 = xg * 4;
            v[0] = lds.wavelet[dp * 1024 + (x0+0) * 32 + (yr ^ (x0+0))];
            v[1] = lds.wavelet[dp * 1024 + (x0+1) * 32 + (yr ^ (x0+1))];
            v[2] = lds.wavelet[dp * 1024 + (x0+2) * 32 + (yr ^ (x0+2))];
            v[3] = lds.wavelet[dp * 1024 + (x0+3) * 32 + (yr ^ (x0+3))];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    // ---- Phase 4: Quantize + RLE encode (two-pass, compacted) ----
    // Block layout: [1024B zline_meta] [RLE data]
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    unsigned char* block_out = output + (long)bid * 4 * WRLE_LDS_BYTES;
    unsigned char* meta_out = block_out;
    unsigned char* rle_out  = block_out + WRLE_META_PER_BLOCK;
    int block_total = 0;

    for (int x_off = 0; x_off < 4; ++x_off) {
        int my_count = wrle_zline<false>(regs, x_off, mulfac, nullptr);

        meta_out[x_off * 256 + tid] = (unsigned char)my_count;

        int my_offset, pass_total;
        BlockScan().exclusive_scan(my_count, my_offset, 0, pass_total, lds.scan);
        __syncthreads();

        wrle_zline<true>(regs, x_off, mulfac, lds.compact + my_offset);
        __syncthreads();

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
                unsigned* dst32 = (unsigned*)(rle_out + block_total + off);
                *dst32 = val;
            }
        }
        __syncthreads();
        block_total += pass_total;
    }

    if (tid == 0)
        block_sizes[bid] = WRLE_META_PER_BLOCK + block_total;
}

inline hipError_t hipWaveletRLEFused(
    const float* input,
    unsigned char* output,
    size_t* block_sizes,
    float scale,
    int nx, int ny, int nz,
    int ldimx, int ldimxy)
{
    dim3 grid((nx + 31) / 32, (ny + 31) / 32, (nz + 31) / 32);
    waveletRLEFusedKernel<<<grid, dim3(256)>>>(
        input, output, block_sizes, scale, ldimx, ldimxy,
        nullptr, nullptr);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// saddr variant: uses global_load_dwordx4 with SGPR base + VGPR offset
// instead of buffer instructions.  Everything else is identical.
// ---------------------------------------------------------------------------

__device__ __forceinline__
wrle_float4_vec wrle_saddr_load_nt(const float* base, uint32_t byte_off) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(base);
    return __builtin_nontemporal_load(
        reinterpret_cast<const wrle_float4_vec*>(p + byte_off));
}

__launch_bounds__(256, 2)
__global__ void waveletRLEFusedSaddrKernel(
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
    constexpr int NTHREADS = 256;
    using BlockScan = rocprim::block_scan<int, NTHREADS>;

    __shared__ union {
        float wavelet[BATCH * 1024];
        typename BlockScan::storage_type scan;
        unsigned char compact[WRLE_LDS_BYTES];
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
        if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            if (d_mulfac_out) *d_mulfac_out = mulfac;
        }
    } else {
        mulfac = scale;
    }

    const float* block_base = input + (size_t)blockIdx.z * 32 * ldimxy;
    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    uint32_t xy_byte     = (uint32_t)(gx + gy * ldimx) * (uint32_t)sizeof(float);
    uint32_t byte_stride = (uint32_t)ldimxy * (uint32_t)sizeof(float);

    // ---- Phase 1: Load 32 planes from global (saddr) ----
    wrle_float4_vec regs[PLANES];
    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = wrle_saddr_load_nt(block_base, xy_byte + (uint32_t)p * byte_stride);

    // ---- Phase 2: Z-transform in registers ----
    ds79_forward_f4_scalar_tmp(regs, PLANES);

    // ---- Phase 3: Y+X transform in LDS (batches of 8) ----
    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            wrle_float4_vec v = regs[pb + dp];
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
            wrle_float4_vec v;
            int x0 = xg * 4;
            v[0] = lds.wavelet[dp * 1024 + (x0+0) * 32 + (yr ^ (x0+0))];
            v[1] = lds.wavelet[dp * 1024 + (x0+1) * 32 + (yr ^ (x0+1))];
            v[2] = lds.wavelet[dp * 1024 + (x0+2) * 32 + (yr ^ (x0+2))];
            v[3] = lds.wavelet[dp * 1024 + (x0+3) * 32 + (yr ^ (x0+3))];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    // ---- Phase 4: Quantize + RLE encode (two-pass, compacted) ----
    // Block layout: [1024B zline_meta] [RLE data]
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    unsigned char* block_out = output + (long)bid * 4 * WRLE_LDS_BYTES;
    unsigned char* meta_out = block_out;
    unsigned char* rle_out  = block_out + WRLE_META_PER_BLOCK;
    int block_total = 0;

    for (int x_off = 0; x_off < 4; ++x_off) {
        int my_count = wrle_zline<false>(regs, x_off, mulfac, nullptr);

        meta_out[x_off * 256 + tid] = (unsigned char)my_count;

        int my_offset, pass_total;
        BlockScan().exclusive_scan(my_count, my_offset, 0, pass_total, lds.scan);
        __syncthreads();

        wrle_zline<true>(regs, x_off, mulfac, lds.compact + my_offset);
        __syncthreads();

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
                unsigned* dst32 = (unsigned*)(rle_out + block_total + off);
                *dst32 = val;
            }
        }
        __syncthreads();
        block_total += pass_total;
    }

    if (tid == 0)
        block_sizes[bid] = WRLE_META_PER_BLOCK + block_total;
}

inline hipError_t hipWaveletRLEFusedSaddr(
    const float* input,
    unsigned char* output,
    size_t* block_sizes,
    float scale,
    int nx, int ny, int nz,
    int ldimx, int ldimxy)
{
    dim3 grid((nx + 31) / 32, (ny + 31) / 32, (nz + 31) / 32);
    waveletRLEFusedSaddrKernel<<<grid, dim3(256)>>>(
        input, output, block_sizes, scale, ldimx, ldimxy,
        nullptr, nullptr);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Output compaction: copies variable-length blocks from fixed-stride layout
// to a tightly packed buffer using precomputed prefix-sum offsets.
// ---------------------------------------------------------------------------

// Output compaction with optional header writing.
// If hdr != nullptr, each block writes its offset to the header, and block 0
// writes the header constants ({num_blocks, num_mulfacs}) and mulfacs.
// dst points to the payload region (after header).
__global__ void wrleCompactKernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    const size_t* __restrict__ block_sizes,
    const size_t* __restrict__ offsets,
    unsigned char* __restrict__ hdr,
    int num_blocks,
    int num_mulfacs,
    const float* __restrict__ d_mulfac)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    size_t size = block_sizes[bid];
    size_t dst_off = offsets[bid];
    size_t src_off = (size_t)bid * 4 * WRLE_LDS_BYTES;

    if (hdr != nullptr && tid == 0) {
        ((size_t*)(hdr + 8))[bid] = offsets[bid];
        if (bid == 0) {
            ((int*)hdr)[0] = num_blocks;
            ((int*)hdr)[1] = num_mulfacs;
            float* mf_dst = (float*)(hdr + 8 + 8 * num_blocks);
            for (int i = 0; i < num_mulfacs; ++i)
                mf_dst[i] = d_mulfac[i];
        }
    }

    for (size_t i = tid * 4; i < size; i += blockDim.x * 4) {
        unsigned val;
        __builtin_memcpy(&val, src + src_off + i, 4);
        size_t remain = size - i;
        if (remain >= 4) {
            __builtin_memcpy(dst + dst_off + i, &val, 4);
        } else {
            for (size_t b = 0; b < remain; ++b)
                dst[dst_off + i + b] = (unsigned char)(val >> (b * 8));
        }
    }
}

// Combined fused encode + compaction.
//
// Workflow: encode (fixed-stride) → prefix sum → compact copy.
//
// Buffers (all device pointers):
//   scratch         — nblocks * 4 * WRLE_LDS_BYTES bytes (fixed-stride encoded output)
//   compact_out     — destination for tightly packed compressed data (at most same size as scratch)
//   block_sizes     — [nblocks] int, filled with per-block compressed sizes
//   offsets         — [nblocks] int, filled with exclusive prefix sum of block_sizes
//   scan_temp       — rocprim temp storage (query size with hipWaveletRLECompactScanTempSize)
//
// After return and stream sync, total compressed bytes = offsets[nblocks-1] + block_sizes[nblocks-1].
inline hipError_t hipWaveletRLEFusedCompact(
    const float* input,
    unsigned char* scratch,
    unsigned char* compact_out,
    size_t* block_sizes,
    size_t* offsets,
    void* scan_temp,
    size_t scan_temp_bytes,
    float scale,
    int nx, int ny, int nz,
    int ldimx, int ldimxy,
    hipStream_t stream = 0)
{
    int nblocks = (nx / 32) * (ny / 32) * (nz / 32);

    dim3 grid((nx + 31) / 32, (ny + 31) / 32, (nz + 31) / 32);
    waveletRLEFusedKernel<<<grid, dim3(256), 0, stream>>>(
        input, scratch, block_sizes, scale, ldimx, ldimxy,
        nullptr, nullptr);

    hipError_t err = rocprim::exclusive_scan(
        scan_temp, scan_temp_bytes,
        block_sizes, offsets, (size_t)0, (size_t)nblocks,
        rocprim::plus<size_t>(), stream);
    if (err != hipSuccess) return err;

    wrleCompactKernel<<<nblocks, 256, 0, stream>>>(
        scratch, compact_out, block_sizes, offsets,
        nullptr, 0, 0, nullptr);

    return hipGetLastError();
}

// Query scan temp storage bytes needed for the compaction prefix sum.
inline hipError_t hipWaveletRLECompactScanTempSize(int nblocks, size_t* scan_temp_bytes)
{
    return rocprim::exclusive_scan(
        nullptr, *scan_temp_bytes,
        (size_t*)nullptr, (size_t*)nullptr, (size_t)0, (size_t)nblocks,
        rocprim::plus<size_t>());
}

// ---------------------------------------------------------------------------
// Segment-aligned variant: eliminates 1024B per-block z-line metadata.
// Encodes wavelet coefficients in raster order (x, y, plane) with null-padded
// segment boundaries.  Thread 0 does sequential RLE encode from LDS after
// the Y/X transform.  Other threads idle during encode.
// ---------------------------------------------------------------------------

#include "hipSegmentedRLE.h"

__launch_bounds__(256, 2)
__global__ void waveletSegRLEFusedKernel(
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

    __shared__ float lds_wavelet[BATCH * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float mulfac;
    if (d_rms != nullptr) {
        float rms = (float)*d_rms;
        float product = rms * scale;
        mulfac = (product > 0.0f && __builtin_isfinite(1.0f / product))
                 ? (1.0f / product) : 1.0f;
        if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            if (d_mulfac_out) *d_mulfac_out = mulfac;
        }
    } else {
        mulfac = scale;
    }

    const float* block_base = input + (size_t)blockIdx.z * 32 * ldimxy;

    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    uint32_t byte_off = (gx + gy * ldimx) * (uint32_t)sizeof(float);

    // ---- Phase 1: Load 32 planes from global ----
    wrle_float4_vec regs[PLANES];
    #pragma unroll
    for (int p = 0; p < PLANES; p++) {
        auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<float*>(block_base + (long)p * ldimxy),
            0, -1, 0x00027000);
        regs[p] = __builtin_bit_cast(wrle_float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(
                rsrc, byte_off, 0, SLC));
    }

    // ---- Phase 2: Z-transform in registers ----
    ds79_forward_f4_scalar_tmp(regs, PLANES);

    // ---- Phase 3 + 4: Y/X transform in LDS, then encode ----
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    unsigned char* block_out = output + (long)bid * 4 * WRLE_LDS_BYTES;
    int bp = 0, rle = 0;

    for (int pb = 0; pb < PLANES; pb += BATCH) {
        // Write registers to LDS for Y/X transform
        for (int dp = 0; dp < BATCH; dp++) {
            wrle_float4_vec v = regs[pb + dp];
            int x0 = xg * 4;
            lds_wavelet[dp * 1024 + (x0+0) * 32 + (yr ^ (x0+0))] = v[0];
            lds_wavelet[dp * 1024 + (x0+1) * 32 + (yr ^ (x0+1))] = v[1];
            lds_wavelet[dp * 1024 + (x0+2) * 32 + (yr ^ (x0+2))] = v[2];
            lds_wavelet[dp * 1024 + (x0+3) * 32 + (yr ^ (x0+3))] = v[3];
        }
        __syncthreads();

        // Y transform
        {
            int pl  = tid / 32;
            int pos = tid % 32;
            float line[32];
            for (int y = 0; y < 32; y++)
                line[y] = lds_wavelet[pl * 1024 + pos * 32 + (y ^ pos)];
            ds79_forward_reg32(line);
            for (int y = 0; y < 32; y++)
                lds_wavelet[pl * 1024 + pos * 32 + (y ^ pos)] = line[y];
        }
        __syncthreads();

        // X transform
        {
            int pl  = tid / 32;
            int pos = tid % 32;
            float line[32];
            for (int x = 0; x < 32; x++)
                line[x] = lds_wavelet[pl * 1024 + x * 32 + (pos ^ x)];
            ds79_forward_reg32(line);
            for (int x = 0; x < 32; x++)
                lds_wavelet[pl * 1024 + x * 32 + (pos ^ x)] = line[x];
        }
        __syncthreads();

        // Thread 0: encode this batch from LDS
        if (tid == 0)
            seg_rle_encode_batch(lds_wavelet, mulfac, block_out,
                                 BATCH, SEG_RLE_SEG_SIZE, &bp, &rle);
        __syncthreads();
    }

    // Finalize: flush remaining zeros, pad to segment boundary
    if (tid == 0) {
        seg_rle_finalize(block_out, &bp, &rle, SEG_RLE_SEG_SIZE);
        block_sizes[bid] = bp;
    }
}

#endif // HIPWAVELET_RLE_H
