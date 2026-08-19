// Copyright (C) 2025 Advanced Micro Devices, Inc.
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
// Phases 1-3 (load, Z-transform, Y+X transform) are identical to
// waveletRLEFusedKernel in hipWaveletRLE.h, so the quantized nonzero set and
// values are bit-for-bit identical to the RLE path (used for validation).

#include <hip/hip_runtime.h>
#include <rocprim/block/block_scan.hpp>
#include "ds79.h"
#include "hipWaveletRLE.h"   // wrle_float4_vec, WRLE_LDS_BYTES, ds79 helpers

// One significance word (32 bits, one per z) per z-line; 4*256 z-lines/block.
static constexpr int  WBMP_BITMAP_WORDS = 1024;
static constexpr int  WBMP_BITMAP_BYTES = WBMP_BITMAP_WORDS * 4;      // 4096
// Worst case (all 32768 coefficients nonzero) value region.
static constexpr int  WBMP_MAX_VAL_BYTES = 32768 * 4;                 // 131072
static constexpr long WBMP_SLOT_BYTES     = WBMP_BITMAP_BYTES + WBMP_MAX_VAL_BYTES; // 135168

__launch_bounds__(256, 2)
__global__ void waveletBitmapFusedKernel(
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
            __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_off, 0, SLC));
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
            int ival = (int)(mulfac * regs[z][x_off]);
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
                int ival = (int)(mulfac * regs[z][x_off]);
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

// Coded block stride: [4096B bitmap][4B width header][nnz * W bytes], W<=4.
static constexpr long WBMP_CODE_SLOT_BYTES =
    WBMP_BITMAP_BYTES + 8 + WBMP_MAX_VAL_BYTES;   // 135176

// Minimum signed byte width to hold [-maxabs, maxabs].
__host__ __device__ __forceinline__ int wbmp_width_bytes(int maxabs) {
    if (maxabs <= 0x7f)      return 1;
    if (maxabs <= 0x7fff)    return 2;
    if (maxabs <= 0x7fffff)  return 3;
    return 4;
}

// ---------------------------------------------------------------------------
// Kernel 2 (coding): per-block fixed-width packing of the value stream.
// Reads kernel-1 output [bitmap + packed int32]; writes [bitmap + W + W-byte
// values].  One workgroup per block.  Bitmap is copied verbatim.
//   coded block bytes = 4096 + 4 + nnz*W
// ---------------------------------------------------------------------------
__launch_bounds__(256, 4)
__global__ void waveletBitmapCodeKernel(
    const unsigned char* __restrict__ scratch1,
    const size_t* __restrict__ block_sizes1,
    unsigned char* __restrict__ out,
    size_t* __restrict__ block_sizes2)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    const unsigned char* blk_in = scratch1 + (long)bid * WBMP_SLOT_BYTES;
    const uint32_t* bmp_in  = reinterpret_cast<const uint32_t*>(blk_in);
    const int32_t*  vals_in = reinterpret_cast<const int32_t*>(blk_in + WBMP_BITMAP_BYTES);
    int nnz = (int)((block_sizes1[bid] - WBMP_BITMAP_BYTES) / 4);

    unsigned char* blk_out = out + (long)bid * WBMP_CODE_SLOT_BYTES;
    uint32_t* bmp_out = reinterpret_cast<uint32_t*>(blk_out);

    __shared__ int s_max;
    if (tid == 0) s_max = 0;
    __syncthreads();

    int local = 0;
    for (int i = tid; i < nnz; i += 256) {
        int v = vals_in[i];
        int a = v < 0 ? -v : v;
        if (a > local) local = a;
    }
    atomicMax(&s_max, local);

    // Copy bitmap verbatim (coalesced) while the reduction settles.
    for (int i = tid; i < WBMP_BITMAP_WORDS; i += 256)
        bmp_out[i] = bmp_in[i];
    __syncthreads();

    int W = wbmp_width_bytes(s_max);

    if (tid == 0) {
        reinterpret_cast<int*>(blk_out + WBMP_BITMAP_BYTES)[0] = W;
        block_sizes2[bid] = (size_t)WBMP_BITMAP_BYTES + 4 + (size_t)nnz * W;
    }

    unsigned char* vout = blk_out + WBMP_BITMAP_BYTES + 4;
    for (int i = tid; i < nnz; i += 256) {
        unsigned uv = (unsigned)vals_in[i];
        long base = (long)i * W;
        #pragma unroll
        for (int b = 0; b < 4; ++b)
            if (b < W) vout[base + b] = (unsigned char)(uv >> (8 * b));
    }
}

inline hipError_t hipWaveletBitmapCode(
    const unsigned char* scratch1,
    const size_t* block_sizes1,
    unsigned char* out,
    size_t* block_sizes2,
    int nblocks,
    hipStream_t stream = 0)
{
    waveletBitmapCodeKernel<<<nblocks, dim3(256), 0, stream>>>(
        scratch1, block_sizes1, out, block_sizes2);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Kernel 2 (coding), per-LINE width: one width per z-line instead of per
// block, eliminating the block-wide padding waste.  Width table is 2 bits
// per z-line (code = W-1), 1024 lines -> 256 B/block.
// Coded block: [4096B bitmap][256B width table][packed variable-width values]
//   bytes = 4096 + 256 + sum_line(nnz_line * W_line)
// Each thread persistently owns 4 z-lines (x_off=0..3, fixed tid), so per-line
// nnz/W/offsets stay in registers; two block-scans give input(nnz) and output
// (nnz*W) prefix sums.
// ---------------------------------------------------------------------------
static constexpr int  WBMP_WTAB_BYTES     = 256;   // 2 bits * 1024 lines
static constexpr long WBMP_PL_SLOT_BYTES  =
    WBMP_BITMAP_BYTES + WBMP_WTAB_BYTES + WBMP_MAX_VAL_BYTES;

__device__ __forceinline__ int wbmp_line_width(const int32_t* v, int n) {
    int mx = 0;
    for (int k = 0; k < n; ++k) { int a = v[k]; a = a < 0 ? -a : a; if (a > mx) mx = a; }
    return wbmp_width_bytes(mx);
}

__launch_bounds__(256, 4)
__global__ void waveletBitmapCodePerLineKernel(
    const unsigned char* __restrict__ scratch1,
    const size_t* __restrict__ block_sizes1,
    unsigned char* __restrict__ out,
    size_t* __restrict__ block_sizes2)
{
    constexpr int NTHREADS = 256;
    using BlockScan = rocprim::block_scan<int, NTHREADS>;
    __shared__ typename BlockScan::storage_type scan;

    int bid = blockIdx.x, tid = threadIdx.x;
    const unsigned char* blk_in = scratch1 + (long)bid * WBMP_SLOT_BYTES;
    const uint32_t* bmp_in  = reinterpret_cast<const uint32_t*>(blk_in);
    const int32_t*  vals_in = reinterpret_cast<const int32_t*>(blk_in + WBMP_BITMAP_BYTES);

    unsigned char* blk_out = out + (long)bid * WBMP_PL_SLOT_BYTES;
    uint32_t* bmp_out = reinterpret_cast<uint32_t*>(blk_out);
    uint32_t* wtab    = reinterpret_cast<uint32_t*>(blk_out + WBMP_BITMAP_BYTES);
    unsigned char* vout = blk_out + WBMP_BITMAP_BYTES + WBMP_WTAB_BYTES;

    for (int i = tid; i < WBMP_BITMAP_WORDS; i += NTHREADS) bmp_out[i] = bmp_in[i];
    for (int i = tid; i < WBMP_WTAB_BYTES / 4; i += NTHREADS) wtab[i] = 0;
    __syncthreads();

    int nnz[4], W[4], in_off[4], out_off[4];
    int in_base = 0, out_base = 0;

    // Phase 1: per-line nnz, input offsets (scan over nnz), per-line width.
    for (int x = 0; x < 4; ++x) {
        int nz = __popc(bmp_in[x * 256 + tid]);
        nnz[x] = nz;
        int off, tot;
        BlockScan().exclusive_scan(nz, off, 0, tot, scan);
        __syncthreads();
        in_off[x] = in_base + off;
        in_base += tot;
        W[x] = nz ? wbmp_line_width(vals_in + in_off[x], nz) : 1;
    }
    // Phase 2: output byte offsets (scan over nnz*W).
    for (int x = 0; x < 4; ++x) {
        int ob = nnz[x] * W[x];
        int off, tot;
        BlockScan().exclusive_scan(ob, off, 0, tot, scan);
        __syncthreads();
        out_off[x] = out_base + off;
        out_base += tot;
    }
    // Phase 3: write 2-bit width table + pack values at per-line width.
    for (int x = 0; x < 4; ++x) {
        int L = x * 256 + tid;
        if (nnz[x] > 0)
            atomicOr(&wtab[L >> 4], (uint32_t)(W[x] - 1) << ((L & 15) * 2));
        long base = out_off[x];
        for (int k = 0; k < nnz[x]; ++k) {
            unsigned uv = (unsigned)vals_in[in_off[x] + k];
            long p = base + (long)k * W[x];
            #pragma unroll
            for (int b = 0; b < 4; ++b)
                if (b < W[x]) vout[p + b] = (unsigned char)(uv >> (8 * b));
        }
    }
    if (tid == 0)
        block_sizes2[bid] = (size_t)WBMP_BITMAP_BYTES + WBMP_WTAB_BYTES + (size_t)out_base;
}

inline hipError_t hipWaveletBitmapCodePerLine(
    const unsigned char* scratch1,
    const size_t* block_sizes1,
    unsigned char* out,
    size_t* block_sizes2,
    int nblocks,
    hipStream_t stream = 0)
{
    waveletBitmapCodePerLineKernel<<<nblocks, dim3(256), 0, stream>>>(
        scratch1, block_sizes1, out, block_sizes2);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Kernel 2 (coding), TWO-LEVEL occupancy + per-line width.  Replaces the flat
// 4096 B/block significance bitmap with a 128 B line-occupancy mask (1 bit per
// z-line) followed by only the nonempty lines' data.  All region bases are
// recoverable at decode from popcount(occupancy).
// Coded block:
//   [128B occupancy][4B * n_ne masks][2b * n_ne widths][per-line values]
//   n_ne  = popcount(occupancy)
//   bytes = 128 + 4*n_ne + ceil(2*n_ne/8) + sum_line(nnz*W)
// ---------------------------------------------------------------------------
static constexpr int  WBMP_OCC_BYTES     = 128;   // 1024 bits, 1 per z-line
static constexpr long WBMP_TL_SLOT_BYTES =
    WBMP_OCC_BYTES + WBMP_BITMAP_BYTES + WBMP_WTAB_BYTES + WBMP_MAX_VAL_BYTES;

__launch_bounds__(256, 4)
__global__ void waveletBitmapCodeTwoLevelKernel(
    const unsigned char* __restrict__ scratch1,
    const size_t* __restrict__ block_sizes1,
    unsigned char* __restrict__ out,
    size_t* __restrict__ block_sizes2)
{
    constexpr int NTHREADS = 256;
    using BlockScan = rocprim::block_scan<int, NTHREADS>;
    __shared__ typename BlockScan::storage_type scan;

    int bid = blockIdx.x, tid = threadIdx.x;
    const unsigned char* blk_in = scratch1 + (long)bid * WBMP_SLOT_BYTES;
    const uint32_t* bmp_in  = reinterpret_cast<const uint32_t*>(blk_in);
    const int32_t*  vals_in = reinterpret_cast<const int32_t*>(blk_in + WBMP_BITMAP_BYTES);

    unsigned char* blk_out = out + (long)bid * WBMP_TL_SLOT_BYTES;
    uint32_t* occ = reinterpret_cast<uint32_t*>(blk_out);   // 32 words

    for (int i = tid; i < WBMP_OCC_BYTES / 4; i += NTHREADS) occ[i] = 0;
    __syncthreads();

    int mask[4], nnz[4], W[4], occb[4], in_off[4], occ_rank[4], val_off[4];

    // Phase 1: mask, nnz, occupancy, input offsets (scan nnz), per-line width.
    int in_base = 0;
    for (int x = 0; x < 4; ++x) {
        uint32_t m = bmp_in[x * 256 + tid];
        mask[x] = (int)m;
        int nz = __popc(m);
        nnz[x] = nz;
        occb[x] = m ? 1 : 0;
        int off, tot;
        BlockScan().exclusive_scan(nz, off, 0, tot, scan);
        __syncthreads();
        in_off[x] = in_base + off;
        in_base += tot;
        W[x] = nz ? wbmp_line_width(vals_in + in_off[x], nz) : 1;
    }
    // Phase 2a: nonempty rank (scan occupancy) -> total_ne.
    int ne_base = 0;
    for (int x = 0; x < 4; ++x) {
        int off, tot;
        BlockScan().exclusive_scan(occb[x], off, 0, tot, scan);
        __syncthreads();
        occ_rank[x] = ne_base + off;
        ne_base += tot;
    }
    int total_ne = ne_base;
    // Phase 2b: value byte offsets (scan nnz*W).
    int vb_base = 0;
    for (int x = 0; x < 4; ++x) {
        int ob = nnz[x] * W[x];
        int off, tot;
        BlockScan().exclusive_scan(ob, off, 0, tot, scan);
        __syncthreads();
        val_off[x] = vb_base + off;
        vb_base += tot;
    }
    int total_val = vb_base;

    long masks_base = WBMP_OCC_BYTES;                          // 128, 4-aligned
    long wtab_base  = masks_base + 4L * total_ne;              // 4-aligned
    long vals_base  = wtab_base + (2L * total_ne + 7) / 8;
    uint32_t* masks = reinterpret_cast<uint32_t*>(blk_out + masks_base);
    uint32_t* wtab  = reinterpret_cast<uint32_t*>(blk_out + wtab_base);

    // Zero the width-table words (2 bits per nonempty line).
    int wtab_words = (2 * total_ne + 31) / 32;
    for (int i = tid; i < wtab_words; i += NTHREADS) wtab[i] = 0;
    __syncthreads();

    // Phase 3: write occupancy, masks, widths, packed values (nonempty lines).
    for (int x = 0; x < 4; ++x) {
        if (!occb[x]) continue;
        int L = x * 256 + tid;
        atomicOr(&occ[L >> 5], 1u << (L & 31));
        int r = occ_rank[x];
        masks[r] = (uint32_t)mask[x];
        atomicOr(&wtab[r >> 4], (uint32_t)(W[x] - 1) << ((r & 15) * 2));
        long p = vals_base + val_off[x];
        for (int k = 0; k < nnz[x]; ++k) {
            unsigned uv = (unsigned)vals_in[in_off[x] + k];
            long q = p + (long)k * W[x];
            #pragma unroll
            for (int b = 0; b < 4; ++b)
                if (b < W[x]) blk_out[q + b] = (unsigned char)(uv >> (8 * b));
        }
    }
    if (tid == 0) block_sizes2[bid] = (size_t)(vals_base + total_val);
}

inline hipError_t hipWaveletBitmapCodeTwoLevel(
    const unsigned char* scratch1,
    const size_t* block_sizes1,
    unsigned char* out,
    size_t* block_sizes2,
    int nblocks,
    hipStream_t stream = 0)
{
    waveletBitmapCodeTwoLevelKernel<<<nblocks, dim3(256), 0, stream>>>(
        scratch1, block_sizes1, out, block_sizes2);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Kernel 2 (coding), TWO-LEVEL occupancy + per-line width -- OPTIMIZED.
//
// Byte-for-byte identical output to waveletBitmapCodeTwoLevelKernel (occupancy,
// masks, 2-bit width table and packed values in the same layout and order), so
// the existing decoder is unchanged. It is faster because of six levers found by
// the polyopt round-3 campaign (worker continue_1_s2, mean 1.59x on gfx950; the
// coding kernel `op` phase alone ~5x):
//   - one workgroup of 1024 threads, one z-line per thread (vs 256 threads x 4)
//   - wave64 exclusive scans as DPP row shifts, not ds_bpermute/LDS crossbar
//   - the two width-independent scans (input offset, nonempty rank) fused into
//     one dual block scan
//   - a branch-free per-width group packer (whole-dword stores, ragged tail)
//   - the packed values staged in an LDS image of the value region, then drained
//     to global with 16-byte non-temporal stores (one instruction per ~1 KB)
//   - occupancy written from a single ballot per warp
//
// LDS budget is WBMP_MAX_VAL_BYTES + ~1.2 KB (~132 KB), so this kernel targets
// gfx950 (CDNA4) and its larger LDS; the original kernel above remains the
// portable path for gfx90a and earlier. The unused tail of each output slot is
// left untouched exactly as in the original -- block_sizes2[bid] records the
// exact coded length and the decoder reads only [0, used).
// ---------------------------------------------------------------------------
namespace wbmp_opt {

typedef unsigned int v4u __attribute__((ext_vector_type(4)));

static constexpr int WBMP_OPT_THREADS = 1024;
static constexpr int WBMP_OPT_WARP    = 64;
static constexpr int WBMP_OPT_NWARPS  = WBMP_OPT_THREADS / WBMP_OPT_WARP;   // 16
// Worst case a coded block can pack is WBMP_MAX_VAL_BYTES; the extra 16 absorbs
// the offset that keeps an aligned 16-byte block of the image an aligned 16-byte
// block of the slot. No input can overflow it, so there is no fallback path.
static constexpr int WBMP_OPT_VBUF    = WBMP_MAX_VAL_BYTES + 16;

__device__ __forceinline__ void st32(unsigned char* p, unsigned u) {
    __builtin_memcpy(p, &u, 4);
}

// Wave64 inclusive add scan as six DPP row shifts. __shfl_up lowers to
// ds_bpermute_b32 on gfx9 -- an LDS crossbar round trip per step; the DPP form
// is six plain VALU adds with a modifier and touches no LDS.
template <int CTRL, int RMASK>
__device__ __forceinline__ int dpp_add(int x) {
    return x + __builtin_amdgcn_update_dpp(0, x, CTRL, RMASK, 0xf, false);
}

__device__ __forceinline__ int winc(int v) {
    v = dpp_add<0x111, 0xf>(v); /* row_shr:1  */
    v = dpp_add<0x112, 0xf>(v); /* row_shr:2  */
    v = dpp_add<0x114, 0xf>(v); /* row_shr:4  */
    v = dpp_add<0x118, 0xf>(v); /* row_shr:8  */
    v = dpp_add<0x142, 0xa>(v); /* row_bcast:15 -> rows 1,3 */
    v = dpp_add<0x143, 0xc>(v); /* row_bcast:31 -> rows 2,3 */
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
        const int s = (lane < WBMP_OPT_NWARPS) ? warp_part[lane] : 0;
        const int sincl = winc(s);
        if (lane < WBMP_OPT_NWARPS) warp_part[lane] = sincl;
    }
    __syncthreads();
    total = warp_part[WBMP_OPT_NWARPS - 1];
    const int wprefix = (warp == 0) ? 0 : warp_part[warp - 1];
    return wprefix + wincl - v;
}

__device__ __forceinline__ void block_exscan2(int v0, int v1, int& ex0, int& ex1, int& total0,
                                              int& total1, int* warp_part) {
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
        const int s1 = (lane < WBMP_OPT_NWARPS) ? warp_part[lane + WBMP_OPT_NWARPS] : 0;
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
    const int w1prefix = (warp == 0) ? 0 : warp_part[warp + WBMP_OPT_NWARPS - 1];
    ex0 = w0prefix + w0incl - v0;
    ex1 = w1prefix + w1incl - v1;
}

template <int W>
__device__ __forceinline__ void pack_line_w(unsigned char* out, long p, const int32_t* val,
                                            int base, int nz) {
    long q = p;
    int k = 0;
    if (W == 1) {
        for (; k + 4 <= nz; k += 4) {
            const unsigned a = (unsigned)__ldg(val + base + k);
            const unsigned b = (unsigned)__ldg(val + base + k + 1);
            const unsigned c = (unsigned)__ldg(val + base + k + 2);
            const unsigned d = (unsigned)__ldg(val + base + k + 3);
            st32(out + q, (a & 0xffu) | ((b & 0xffu) << 8) | ((c & 0xffu) << 16) | (d << 24));
            q += 4;
        }
    } else if (W == 2) {
        for (; k + 2 <= nz; k += 2) {
            const unsigned a = (unsigned)__ldg(val + base + k);
            const unsigned b = (unsigned)__ldg(val + base + k + 1);
            st32(out + q, (a & 0xffffu) | (b << 16));
            q += 4;
        }
    } else if (W == 3) {
        for (; k + 4 <= nz; k += 4) {
            const unsigned a = (unsigned)__ldg(val + base + k);
            const unsigned b = (unsigned)__ldg(val + base + k + 1);
            const unsigned c = (unsigned)__ldg(val + base + k + 2);
            const unsigned d = (unsigned)__ldg(val + base + k + 3);
            st32(out + q + 0, (a & 0xffffffu) | (b << 24));
            st32(out + q + 4, ((b >> 8) & 0xffffu) | (c << 16));
            st32(out + q + 8, ((c >> 16) & 0xffu) | (d << 8));
            q += 12;
        }
    } else {
        for (; k < nz; ++k) {
            st32(out + q, (unsigned)__ldg(val + base + k));
            q += 4;
        }
    }
    for (; k < nz; ++k) {
        const unsigned u = (unsigned)__ldg(val + base + k);
#pragma unroll
        for (int b = 0; b < W; ++b) out[q + b] = (unsigned char)(u >> (8 * b));
        q += W;
    }
}

__device__ __forceinline__ void pack_line(unsigned char* out, long p, const int32_t* val,
                                          int base, int nz, int W) {
    switch (W) {
        case 1: pack_line_w<1>(out, p, val, base, nz); break;
        case 2: pack_line_w<2>(out, p, val, base, nz); break;
        case 3: pack_line_w<3>(out, p, val, base, nz); break;
        default: pack_line_w<4>(out, p, val, base, nz); break;
    }
}

}  // namespace wbmp_opt

__launch_bounds__(wbmp_opt::WBMP_OPT_THREADS)
__global__ void waveletBitmapCodeTwoLevelOptKernel(
    const unsigned char* __restrict__ scratch1,
    const size_t* __restrict__ block_sizes1,
    unsigned char* __restrict__ out,
    size_t* __restrict__ block_sizes2)
{
// This kernel stages the whole per-block value payload in LDS
// (WBMP_OPT_VBUF ~= 131 KB), which fits only CDNA4 (gfx950, 160 KB LDS).  On
// every other target (gfx942/gfx90a: 64 KB LDS, and the host pass) it compiles
// to an empty stub so the translation unit builds portably; the host dispatch
// (hipCompress) only ever launches it on gfx950 and falls back to
// waveletBitmapCodeTwoLevelKernel elsewhere.  Both kernels emit byte-identical
// two-level streams, so the decoder is arch-independent.
#if defined(__gfx950__)
    using namespace wbmp_opt;
    (void)block_sizes1;  // recomputed from the bitmap, kept for signature parity
    __shared__ int warp_part[2 * WBMP_OPT_NWARPS];
    __shared__ unsigned char wtab8[WBMP_BITMAP_WORDS];
    __shared__ unsigned char vbuf[WBMP_OPT_VBUF];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & (WBMP_OPT_WARP - 1);
    const int warp = tid >> 6;

    const unsigned char* blk_in = scratch1 + (long)bid * WBMP_SLOT_BYTES;
    const uint32_t* bmp = reinterpret_cast<const uint32_t*>(blk_in);
    const int32_t*  val = reinterpret_cast<const int32_t*>(blk_in + WBMP_BITMAP_BYTES);
    unsigned char* blk_out = out + (long)bid * WBMP_TL_SLOT_BYTES;

    const uint32_t m = __ldg(bmp + tid);
    const int mask = (int)m;
    const int nz = __popc(m);
    const int occb = m ? 1 : 0;

    // First block scan carries both quantities that do not depend on the width:
    // the input offset (over nnz) and the nonempty rank (over occupancy).
    int tot_nz, tot_ne, in_off, occ_rank;
    block_exscan2(nz, occb, in_off, occ_rank, tot_nz, tot_ne, warp_part);
    (void)tot_nz;

    const long masks_base = WBMP_OCC_BYTES;
    const long wtab_base = masks_base + 4L * tot_ne;
    const long vals_base = wtab_base + (2L * tot_ne + 7) / 8;

    uint32_t* occ = reinterpret_cast<uint32_t*>(blk_out);
    uint32_t* masks = reinterpret_cast<uint32_t*>(blk_out + masks_base);

    const unsigned long long occ_ballot = __ballot(occb);
    if (lane == 31) occ[warp * 2] = (uint32_t)occ_ballot;
    if (lane == 63) occ[warp * 2 + 1] = (uint32_t)(occ_ballot >> 32);
    if (occb) masks[occ_rank] = (uint32_t)mask;

    int mx = 0;
#pragma unroll 4
    for (int k = 0; k < nz; ++k) {
        const int a = __ldg(val + in_off + k);
        mx |= (a < 0 ? -a : a);
    }
    const int W = nz ? wbmp_width_bytes(mx) : 1;

    int tot_val;
    const int val_off = block_exscan(nz * W, tot_val, warp_part);
    const long used = vals_base + tot_val;

    const long g0 = vals_base & ~15L;
    const int lo = (int)(vals_base - g0);

    if (occb) {
        wtab8[occ_rank] = (unsigned char)(W - 1);
        pack_line(vbuf, (long)lo + val_off, val, in_off, nz, W);
    }
    __syncthreads();

    // Drain the image: 16-byte aligned interior with non-temporal stores, the at
    // most fifteen ragged bytes at either end byte-wise.
    {
        const long hi = (long)lo + tot_val;
        const long abeg = ((long)lo + 15) & ~15L;
        const long aend = hi & ~15L;
        const long hend = abeg < hi ? abeg : hi;
        const long tbeg = abeg > aend ? abeg : aend;
        for (long i = (long)lo + tid; i < hend; i += WBMP_OPT_THREADS) blk_out[g0 + i] = vbuf[i];
        for (long i = abeg + 16L * tid; i + 16 <= aend; i += 16L * WBMP_OPT_THREADS) {
            v4u v = *reinterpret_cast<const v4u*>(vbuf + i);
            __builtin_nontemporal_store(v, reinterpret_cast<v4u*>(blk_out + g0 + i));
        }
        for (long i = tbeg + tid; i < hi; i += WBMP_OPT_THREADS) blk_out[g0 + i] = vbuf[i];
    }

    const long wtab_bytes = (2L * tot_ne + 7) / 8;
    const int full_words = (int)(wtab_bytes >> 2);
    uint32_t* wtab = reinterpret_cast<uint32_t*>(blk_out + wtab_base);
    for (int wi = tid; wi <= full_words; wi += WBMP_OPT_THREADS) {
        uint32_t w = 0;
        const int base = wi * 16;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (base + j < tot_ne) w |= (uint32_t)wtab8[base + j] << (j * 2);
        }
        if (wi < full_words) {
            wtab[wi] = w;
        } else {
            unsigned char* edge = blk_out + wtab_base + 4L * full_words;
            for (int j = 0; j < (int)(wtab_bytes & 3); ++j) edge[j] = (unsigned char)(w >> (8 * j));
        }
    }

    if (tid == 0) block_sizes2[bid] = (size_t)used;
#else
    (void)scratch1; (void)block_sizes1; (void)out; (void)block_sizes2;
#endif  // __gfx950__
}

inline hipError_t hipWaveletBitmapCodeTwoLevelOpt(
    const unsigned char* scratch1,
    const size_t* block_sizes1,
    unsigned char* out,
    size_t* block_sizes2,
    int nblocks,
    hipStream_t stream = 0)
{
    waveletBitmapCodeTwoLevelOptKernel<<<nblocks, dim3(wbmp_opt::WBMP_OPT_THREADS), 0, stream>>>(
        scratch1, block_sizes1, out, block_sizes2);
    return hipGetLastError();
}

// ===========================================================================
// TWO-LEVEL codec: compaction + full decode for the public hipCompress API.
// ===========================================================================
// Compaction for the two-level coder: copies each variable-length coded block
// from the fixed-stride (WBMP_TL_SLOT_BYTES) intermediate into a tightly packed
// payload using the exclusive-scan offsets, and writes the self-contained
// RLE-style header (block offsets + mulfac).  The two-level format is fully
// self-describing -- popcount(occupancy) recovers every region base -- so unlike
// the octree stream it needs no per-block significance-size table; the header is
// exactly hipCompressHeaderSize(nb, nmf).  dst points to the payload (after the
// header).  Mirrors woctCompactKernel / wrleCompactKernel.
__global__ void wtlCompactKernel(
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
    size_t src_off = (size_t)bid * WBMP_TL_SLOT_BYTES;

    if (hdr != nullptr && tid == 0) {
        ((size_t*)(hdr + 8))[bid] = offsets[bid];
        if (bid == 0) {
            ((int*)hdr)[0] = num_blocks;
            ((int*)hdr)[1] = num_mulfacs;
            float* mf_dst = (float*)(hdr + 8 + 8L * num_blocks);
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

// Two-level decode (stage A): reconstructs the kernel-1 scratch layout
// [4096B bitmap (L order)][packed int32 values] for one block from its coded
// bytes [128B occupancy][4B*n_ne masks][2b/nonempty-line widths][values].
// Exact inverse of the two-level coder's packing, so the output is byte-
// identical to the waveletBitmapFusedKernel (kernel-1) output and feeds the
// shared inverse-wavelet stage B unchanged.  One workgroup of 1024 threads per
// block, one z-line (L index) per thread.  Portable: only wave64 DPP block
// scans, no arch-specific LDS budget (~128 B shared), so it runs on gfx90a/
// gfx942/gfx950 alike regardless of which encoder produced the stream.
//   blk — pointer to the block's coded bytes (4-byte aligned)
//   out — this block's WBMP_SLOT_BYTES scratch slot
__device__ __forceinline__ void wtl_decode_block_to_scratch(
    const unsigned char* __restrict__ blk,
    unsigned char* __restrict__ out)
{
    using namespace wbmp_opt;
    const int tid = threadIdx.x;                       // L index in [0,1024)
    __shared__ int warp_part[2 * WBMP_OPT_NWARPS];

    const uint32_t* occ = reinterpret_cast<const uint32_t*>(blk);
    uint32_t* bmp_out = reinterpret_cast<uint32_t*>(out);
    int32_t*  val_out = reinterpret_cast<int32_t*>(out + WBMP_BITMAP_BYTES);

    // Occupancy bit for this line and its nonempty rank (exclusive scan).
    const int occb = (occ[tid >> 5] >> (tid & 31)) & 1;
    int tot_ne;
    const int occ_rank = block_exscan(occb, tot_ne, warp_part);
    __syncthreads();                                   // protect warp_part reuse

    const long masks_base = WBMP_OCC_BYTES;
    const long wtab_base  = masks_base + 4L * tot_ne;
    const long vals_base  = wtab_base + (2L * tot_ne + 7) / 8;
    const uint32_t* masks = reinterpret_cast<const uint32_t*>(blk + masks_base);
    const uint32_t* wtab  = reinterpret_cast<const uint32_t*>(blk + wtab_base);

    const uint32_t maskL = occb ? masks[occ_rank] : 0u;
    bmp_out[tid] = maskL;                              // bitmap in L order
    const int nz = __popc(maskL);
    const int W  = occb ? (int)((wtab[occ_rank >> 4] >> ((occ_rank & 15) * 2)) & 3u) + 1 : 1;

    // Scratch int32 offset (scan nnz) and coded value byte offset (scan nnz*W).
    int in_off, val_off, tot_nz, tot_val;
    block_exscan2(nz, nz * W, in_off, val_off, tot_nz, tot_val, warp_part);
    (void)tot_nz; (void)tot_val;

    if (occb) {
        const unsigned char* vp = blk + vals_base + val_off;
        const int sh = 32 - 8 * W;
        for (int k = 0; k < nz; ++k) {
            unsigned uv = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b)
                if (b < W) uv |= (unsigned)vp[(long)k * W + b] << (8 * b);
            val_out[in_off + k] = (int)(uv << sh) >> sh;   // sign-extend from W bytes
        }
    }
}

// API decode kernel: locates each block in the compacted stream via the header
// offset table, reconstructs the kernel-1 scratch layout, and (block 0)
// publishes inv_scale = 1/mulfac for stage B.  Header is the RLE-style layout
// [int nb][int nmf][size_t offsets[nb]][float mulfac[nmf]].
__launch_bounds__(wbmp_opt::WBMP_OPT_THREADS)
__global__ void waveletBitmapTwoLevelDecodeHdrKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ scratch1_out,
    float* __restrict__ inv_scale_out)
{
    const int bid = blockIdx.x, tid = threadIdx.x;
    const int* hdr = reinterpret_cast<const int*>(input);
    const int num_blocks  = hdr[0];
    const int num_mulfacs = hdr[1];
    const size_t* offsets = reinterpret_cast<const size_t*>(input + 8);
    const float*  mulfacs = reinterpret_cast<const float*>(input + 8 + 8L * num_blocks);
    const unsigned char* data_base = input + 8 + 8L * num_blocks + 4L * num_mulfacs;

    if (bid == 0 && tid == 0 && inv_scale_out)
        *inv_scale_out = 1.0f / mulfacs[0];

    wtl_decode_block_to_scratch(
        data_base + offsets[bid],
        scratch1_out + (long)bid * WBMP_SLOT_BYTES);
}

// Launch helper mirroring hipWaveletRLEFused.  output must have
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
    waveletBitmapFusedKernel<<<grid, dim3(256), 0, stream>>>(
        input, output, block_sizes, scale, ldimx, ldimxy,
        d_rms, d_mulfac_out);
    return hipGetLastError();
}

#endif // HIPWAVELET_BITMAP_H
