// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIPWAVELET_RLE_2D_H
#define HIPWAVELET_RLE_2D_H

// Fused 2D wavelet + quantize + RLE encode/decode kernels.
//
// Each workgroup processes up to 32 x-adjacent 32x32 tiles in 4 batches of 8.
// Per batch:  load 8 tiles → Y-transform (LDS) → transposed readback →
//             X-transform (registers) → x-line RLE encode (scan+compact).
//
// Each thread encodes one 32-element x-row as a single RLE line.
// Per-block layout: [32B x-line metadata] [RLE data].
// 32 KB LDS reused across phases. Occupancy 2 on gfx942.

#include <hip/hip_runtime.h>
#include <rocprim/block/block_scan.hpp>
#include <rocprim/device/device_scan.hpp>
#include "ds79.h"
#include "Run_Length_Escape_Codes.hxx"
#include "hipRLEDecode.h"

static constexpr int WRLE2D_TILES_PER_WG = 32;
static constexpr int WRLE2D_BATCH        = 8;
static constexpr int WRLE2D_META_BYTES   = 32;
static constexpr int WRLE2D_SLOT_BYTES   = 8192;

// RLE encode 32 scalar values from a float[32] array.
// WRITE=false: count-only. WRITE=true: writes encoded bytes.
template<bool WRITE>
__device__ __forceinline__
int wrle_xline(const float* values, float scale, unsigned char* dst)
{
    int bp = 0, rle = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        float fval = scale * values[i];
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

// ---------------------------------------------------------------------------
// Forward: 2D wavelet + quantize + RLE encode.
// Grid: (ceil(nbx/32), nby) where nbx = nx/32, nby = ny/32.
// ---------------------------------------------------------------------------
__launch_bounds__(256, 2)
__global__ void waveletRLE2DFusedKernel(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    size_t* __restrict__ block_sizes,
    float scale,
    int ldimx,
    int nbx,
    const double* __restrict__ d_rms,
    float* __restrict__ d_mulfac_out)
{
    constexpr int NTHREADS = 256;
    constexpr int SLC      = 2;
    using BlockScan = rocprim::block_scan<int, NTHREADS>;

    __shared__ union {
        float wavelet[WRLE2D_BATCH * 1024];
        struct {
            typename BlockScan::storage_type scan;
            int block_base_off[WRLE2D_BATCH];
        };
        unsigned char compact[WRLE_LDS_BYTES];
    } lds;

    int tid = threadIdx.x;

    float mulfac;
    if (d_rms != nullptr) {
        float rms = (float)*d_rms;
        float product = rms * scale;
        mulfac = (product > 0.0f && __builtin_isfinite(1.0f / product))
                 ? (1.0f / product) : 1.0f;
        if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            if (d_mulfac_out) *d_mulfac_out = mulfac;
        }
    } else {
        mulfac = scale;
    }

    int wg_tile_x0 = blockIdx.x * WRLE2D_TILES_PER_WG;

    for (int batch = 0; batch < 4; ++batch) {
        int batch_tile_x0 = wg_tile_x0 + batch * WRLE2D_BATCH;

        // ---- Load 8 tiles → LDS ----
        int xg = tid % 8, yr = tid / 8;
        for (int b = 0; b < WRLE2D_BATCH; ++b) {
            int tile_bx = batch_tile_x0 + b;
            int gx = tile_bx * 32 + xg * 4;
            int gy = blockIdx.y * 32 + yr;
            int x0 = xg * 4;
            if (tile_bx < nbx) {
                uint32_t byte_off = (gx + gy * ldimx) * (uint32_t)sizeof(float);
                auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
                    const_cast<float*>(input), 0, -1, 0x00027000);
                auto v = __builtin_bit_cast(
                    __attribute__((__vector_size__(4 * sizeof(float)))) float,
                    __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_off, 0, SLC));
                lds.wavelet[b * 1024 + (x0+0) * 32 + (yr ^ (x0+0))] = v[0];
                lds.wavelet[b * 1024 + (x0+1) * 32 + (yr ^ (x0+1))] = v[1];
                lds.wavelet[b * 1024 + (x0+2) * 32 + (yr ^ (x0+2))] = v[2];
                lds.wavelet[b * 1024 + (x0+3) * 32 + (yr ^ (x0+3))] = v[3];
            } else {
                lds.wavelet[b * 1024 + (x0+0) * 32 + (yr ^ (x0+0))] = 0.0f;
                lds.wavelet[b * 1024 + (x0+1) * 32 + (yr ^ (x0+1))] = 0.0f;
                lds.wavelet[b * 1024 + (x0+2) * 32 + (yr ^ (x0+2))] = 0.0f;
                lds.wavelet[b * 1024 + (x0+3) * 32 + (yr ^ (x0+3))] = 0.0f;
            }
        }
        __syncthreads();

        // ---- Y-transform in LDS ----
        {
            int blk = tid / 32, pos = tid % 32;
            float line[32];
            for (int y = 0; y < 32; y++)
                line[y] = lds.wavelet[blk * 1024 + pos * 32 + (y ^ pos)];
            ds79_forward_reg32(line);
            for (int y = 0; y < 32; y++)
                lds.wavelet[blk * 1024 + pos * 32 + (y ^ pos)] = line[y];
        }
        __syncthreads();

        // ---- Transposed readback + X-transform in registers ----
        int blk = tid / 32;
        int row = tid % 32;
        int tile_bx = batch_tile_x0 + blk;
        bool active = (tile_bx < nbx);

        float xline[32];
        if (active) {
            for (int x = 0; x < 32; x++)
                xline[x] = lds.wavelet[blk * 1024 + x * 32 + (row ^ x)];
            ds79_forward_reg32(xline);
        } else {
            for (int x = 0; x < 32; x++)
                xline[x] = 0.0f;
        }
        __syncthreads();

        // ---- RLE encode ----
        int my_count = active ? wrle_xline<false>(xline, mulfac, nullptr) : 0;

        int global_bid = tile_bx + blockIdx.y * nbx;
        // Write per-line metadata to global scratch
        if (active) {
            unsigned char* slot = output + (size_t)global_bid * WRLE2D_SLOT_BYTES;
            slot[row] = (unsigned char)my_count;
        }

        int my_offset, pass_total;
        BlockScan().exclusive_scan(my_count, my_offset, 0, pass_total, lds.scan);
        __syncthreads();

        // Store per-block base offsets (inside union, will be clobbered by compact)
        if (tid % 32 == 0)
            lds.block_base_off[tid / 32] = my_offset;
        __syncthreads();

        // Cache block boundaries in registers before compact clobbers them
        int blk_start = lds.block_base_off[blk];
        bool next_blk_active = (blk < WRLE2D_BATCH - 1)
                             && (batch_tile_x0 + blk + 1 < nbx);
        int blk_end = next_blk_active
                    ? lds.block_base_off[blk + 1] : pass_total;
        int blk_rle_bytes = blk_end - blk_start;
        __syncthreads();

        // Write encoded data to LDS compact buffer (clobbers block_base_off)
        if (active)
            wrle_xline<true>(xline, mulfac, lds.compact + my_offset);
        __syncthreads();

        // Copy per-block data from LDS → global scratch slot
        if (active) {
            unsigned char* slot = output + (size_t)global_bid * WRLE2D_SLOT_BYTES;
            for (int i = row; i < blk_rle_bytes; i += 32) {
                slot[WRLE2D_META_BYTES + i] = lds.compact[blk_start + i];
            }
            if (row == 0)
                block_sizes[global_bid] = WRLE2D_META_BYTES + blk_rle_bytes;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Inverse: RLE decode + dequantize + inverse wavelet 2D.
// Grid: (ceil(nbx/32), nby).
//
// Phase-separated design with 2 macro-passes of 2 batches each.
// Halves persistent register array (decoded[2][32] = 64 VGPRs vs 128)
// to reduce scratch spilling while keeping phase-separation benefits.
// ---------------------------------------------------------------------------
__launch_bounds__(256, 2)
__global__ void waveletRLE2DInverseFusedKernel(
    const unsigned char* __restrict__ input,
    const size_t* __restrict__ block_sizes,
    const size_t* __restrict__ block_offsets,
    float* __restrict__ output,
    float inv_scale,
    int ldimx,
    int nbx,
    int header_size)
{
    constexpr int NTHREADS = 256;
    constexpr int SLC      = 2;
    constexpr int MACRO_BATCHES = 2;
    using BlockScan = rocprim::block_scan<int, NTHREADS>;
    using f4vec = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    __shared__ union {
        float wavelet[WRLE2D_BATCH * 1024];
        struct {
            typename BlockScan::storage_type scan;
            int block_base_off[WRLE2D_BATCH];
        };
    } lds;

    int tid = threadIdx.x;
    int wg_tile_x0 = blockIdx.x * WRLE2D_TILES_PER_WG;
    int blk = tid / 32;
    int row = tid % 32;

    float actual_inv_scale;
    const unsigned char* data_base;

    if (header_size != 0) {
        const int* hdr = (const int*)input;
        int num_blocks = hdr[0];
        int num_mulfacs = hdr[1];
        const float* mulfacs = (const float*)(input + 8 + 8 * num_blocks);
        actual_inv_scale = 1.0f / mulfacs[0];
        data_base = input + 8 + 8 * num_blocks + 4 * num_mulfacs;
    } else {
        actual_inv_scale = inv_scale;
        data_base = input;
    }

    for (int mp = 0; mp < 4 / MACRO_BATCHES; ++mp) {

        float decoded[MACRO_BATCHES][32];

        // ---- Decode + X-transform (MACRO_BATCHES passes) ----
        for (int b = 0; b < MACRO_BATCHES; ++b) {
            int batch = mp * MACRO_BATCHES + b;
            int batch_tile_x0 = wg_tile_x0 + batch * WRLE2D_BATCH;
            int tile_bx = batch_tile_x0 + blk;
            int global_bid = tile_bx + blockIdx.y * nbx;
            bool active = (tile_bx < nbx);

            int my_bytes = 0;
            const unsigned char* rle_data = nullptr;

            if (active) {
                const unsigned char* block_in;
                if (header_size != 0) {
                    block_in = data_base + ((const size_t*)(input + 8))[global_bid];
                } else if (block_offsets != nullptr) {
                    block_in = data_base + block_offsets[global_bid];
                } else {
                    block_in = data_base + (size_t)global_bid * WRLE2D_SLOT_BYTES;
                }
                rle_data = block_in + WRLE2D_META_BYTES;
                my_bytes = (int)block_in[row];
            }

            int my_offset, pass_total;
            BlockScan().exclusive_scan(my_bytes, my_offset, 0, pass_total, lds.scan);
            __syncthreads();

            if (tid % 32 == 0)
                lds.block_base_off[tid / 32] = my_offset;
            __syncthreads();

            if (active) {
                int blk_base = lds.block_base_off[blk];
                wrle_zline_decode(rle_data + my_offset - blk_base,
                                  my_bytes, actual_inv_scale, decoded[b]);
                us79_inverse_reg32(decoded[b]);
            } else {
                #pragma unroll
                for (int x = 0; x < 32; x++) decoded[b][x] = 0.0f;
            }
        }

        __syncthreads();

        // ---- Y-transform + store (MACRO_BATCHES passes) ----
        for (int b = 0; b < MACRO_BATCHES; ++b) {
            int batch = mp * MACRO_BATCHES + b;
            int batch_tile_x0 = wg_tile_x0 + batch * WRLE2D_BATCH;

            for (int x = 0; x < 32; x++)
                lds.wavelet[blk * 1024 + x * 32 + (row ^ x)] = decoded[b][x];
            __syncthreads();

            {
                int pl  = tid / 32;
                int pos = tid % 32;
                float line[32];
                for (int y = 0; y < 32; y++)
                    line[y] = lds.wavelet[pl * 1024 + pos * 32 + (y ^ pos)];
                us79_inverse_reg32(line);
                for (int y = 0; y < 32; y++)
                    lds.wavelet[pl * 1024 + pos * 32 + (y ^ pos)] = line[y];
            }
            __syncthreads();

            f4vec store_regs[WRLE2D_BATCH];
            {
                int xg = tid % 8, yr = tid / 8;
                int x0 = xg * 4;
                #pragma unroll
                for (int s = 0; s < WRLE2D_BATCH; ++s) {
                    store_regs[s][0] = lds.wavelet[s * 1024 + (x0+0) * 32 + (yr ^ (x0+0))];
                    store_regs[s][1] = lds.wavelet[s * 1024 + (x0+1) * 32 + (yr ^ (x0+1))];
                    store_regs[s][2] = lds.wavelet[s * 1024 + (x0+2) * 32 + (yr ^ (x0+2))];
                    store_regs[s][3] = lds.wavelet[s * 1024 + (x0+3) * 32 + (yr ^ (x0+3))];
                }
            }
            __syncthreads();

            {
                int xg = tid % 8, yr = tid / 8;
                #pragma unroll
                for (int s = 0; s < WRLE2D_BATCH; ++s) {
                    int tb = batch_tile_x0 + s;
                    if (tb >= nbx) continue;
                    int gx = tb * 32 + xg * 4;
                    int gy = blockIdx.y * 32 + yr;
                    uint32_t byte_off = (gx + gy * ldimx) * (uint32_t)sizeof(float);
                    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
                        output, 0, -1, 0x00027000);
                    auto vi = __builtin_bit_cast(
                        __attribute__((__vector_size__(4 * sizeof(int)))) int, store_regs[s]);
                    __builtin_amdgcn_raw_buffer_store_b128(vi, rsrc, byte_off, 0, SLC);
                }
            }
            __syncthreads();
        }
    }
}

// Output compaction for 2D blocks. Same as wrleCompactKernel but with
// configurable slot stride.
__global__ void wrle2DCompactKernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    const size_t* __restrict__ block_sizes,
    const size_t* __restrict__ offsets,
    unsigned char* __restrict__ hdr,
    int num_blocks,
    int num_mulfacs,
    const float* __restrict__ d_mulfac,
    size_t slot_stride)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    size_t size = block_sizes[bid];
    size_t dst_off = offsets[bid];
    size_t src_off = (size_t)bid * slot_stride;

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

inline int hipCompressHeaderSize2D(int num_blocks, int num_mulfacs)
{
    return 8 + 8 * num_blocks + 4 * num_mulfacs;
}

inline hipError_t hipWaveletRLE2DCompactScanTempSize(int nblocks, size_t* scan_temp_bytes)
{
    return rocprim::exclusive_scan(
        nullptr, *scan_temp_bytes,
        (size_t*)nullptr, (size_t*)nullptr, (size_t)0, (size_t)nblocks,
        rocprim::plus<size_t>());
}

#endif // HIPWAVELET_RLE_2D_H
