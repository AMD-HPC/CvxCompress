// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// 2D quadtree significance coder -- the 2D counterpart of the 3D octree
// (hipWaveletOctree.h).  A 32x32 wavelet block's significance set is encoded as
// a quadtree: 2x2 subdivision, depth 5 (node sides 32,16,8,4,2), 4 bits (one per
// child) per non-empty node, nibble-packed two nodes per byte, serialized in
// DFS pre-order.  A per-block 1-byte mode tag falls back to the flat 32-uint32
// row bitmap when the quadtree would be larger (dense blocks).
//
// Pipeline (mirrors the octree split; 2D blocks are tiny so the significance /
// value passes run one workgroup per block, mostly serial in thread 0):
//   encode:  k1 forward (2D wavelet + quantize -> int32 32x32 grid scratch)
//            k2 code     (grid -> [mode][significance][per-block PFOR values])
//            compact     (pack coded blocks + octree-style self-contained header)
//   decode:  stage A     (coded stream -> int32 grid scratch; publishes inv_scale)
//            stage B     (grid -> dequantize + inverse 2D wavelet -> volume)
//
// The forward/inverse transform kernels are byte-for-byte the 2D DS 7/9 transform
// of hipWaveletRLE2D.h with the RLE tail replaced by an int32-grid read/write, so
// the reconstructed field matches the RLE 2D path bit-for-bit at equal scale.

#ifndef HIPWAVELET_QUADTREE_2D_H
#define HIPWAVELET_QUADTREE_2D_H

#include <hip/hip_runtime.h>
#include "ds79.h"

// Tile batching (identical to hipWaveletRLE2D.h so the transform is unchanged).
static constexpr int WQT2D_TILES_PER_WG = 32;
static constexpr int WQT2D_BATCH        = 8;

// Per-block scratch: a 32x32 int32 grid of quantized wavelet coefficients.
static constexpr int  WQT2D_GRID_INTS  = 32 * 32;          // 1024
static constexpr long WQT2D_GRID_BYTES = WQT2D_GRID_INTS * 4L;   // 4096

// Coded-block regions.
static constexpr int WQT2D_HDR_BYTES  = 1;    // mode: 0 flat, 2 quadtree
static constexpr int WQT2D_FLAT_SIG   = 128;  // 32 uint32 row masks (little-endian)
// Worst-case (fully dense) quadtree node count 1+4+16+64+256 = 341, nibble-packed
// two nodes per byte -> ceil(341/2) = 171 bytes (transient; flat caps at 128).
static constexpr int WQT2D_MAX_SIG    = 171;
// Value region is per-block PFOR: [Wbyte][base nnz*W_lo][mask ceil(nnz/8) if
// W_lo<W_hi][patch n_exc*(W_hi-W_lo)].  The optimiser always has the W_lo==W_hi
// (no mask) fallback at cost nnz*W_hi <= 4096, so the chosen region never exceeds
// 1 + nnz*W_hi; +128 covers the mask headroom.
static constexpr int WQT2D_MAX_VAL    = 1 + WQT2D_GRID_INTS * 4 + (WQT2D_GRID_INTS + 7) / 8;

static constexpr long WQT2D_CODE_SLOT_RAW =
    WQT2D_HDR_BYTES + WQT2D_MAX_SIG + WQT2D_MAX_VAL;
static constexpr long WQT2D_CODE_SLOT_BYTES = (WQT2D_CODE_SLOT_RAW + 15) & ~15L;

static constexpr int WQT2D_CODE_THREADS = 64;  // one workgroup per block

// Minimum signed byte width to hold [-maxabs, maxabs].
__host__ __device__ __forceinline__ int wqt2d_width_bytes(unsigned maxabs) {
    if (maxabs <= 0x7fu)      return 1;
    if (maxabs <= 0x7fffu)    return 2;
    if (maxabs <= 0x7fffffu)  return 3;
    return 4;
}

// ---------------------------------------------------------------------------
// Serial quadtree significance core (host + device, byte-exact by construction).
// m[iy] is a 32-bit row mask: bit ix set iff cell (ix,iy) is significant.
// ---------------------------------------------------------------------------

// Any significant cell in the [ox,ox+cs) x [oy,oy+cs) subquad?
__host__ __device__ __forceinline__ bool
wqt2d_sub_any(const uint32_t* m, int ox, int oy, int cs) {
    uint32_t xmask = (cs >= 32) ? 0xFFFFFFFFu
                                : (uint32_t)((((uint32_t)1 << cs) - 1) << ox);
    for (int iy = oy; iy < oy + cs; ++iy)
        if (m[iy] & xmask) return true;
    return false;
}

// 4-bit child occupancy of the node at (px,py) with the given side.  Child
// index c = cx | cy<<1 ; child subquad side = side/2.
__host__ __device__ __forceinline__ int
wqt2d_child_occ(const uint32_t* m, int px, int py, int side) {
    int cs = side >> 1, b = 0;
    for (int c = 0; c < 4; ++c) {
        int cx = c & 1, cy = (c >> 1) & 1;
        if (wqt2d_sub_any(m, px + cx * cs, py + cy * cs, cs)) b |= (1 << c);
    }
    return b;
}

// Encode the 32x32 significance into a DFS-preorder quadtree nibble stream (one
// 4-bit occupancy per non-empty node, two nodes packed per byte: node 2k in the
// low nibble of byte k, node 2k+1 in the high nibble).  Returns bytes written
// (0 for an empty block).  Explicit depth-<=5 stack as parallel scalar arrays.
__host__ __device__ __forceinline__ int
wqt2d_encode(const uint32_t* m, unsigned char* out) {
    int rootb = wqt2d_child_occ(m, 0, 0, 32);
    if (rootb == 0) return 0;
    int px[6], py[6], side[6], bb[6], ch[6];
    int sp = 0, nc = 0;                         // nc: node index (nibble position)
#define WQT2D_PUT_NIB(v) do { int _v = (v) & 0xF; \
    if (nc & 1) out[nc >> 1] |= (unsigned char)(_v << 4); \
    else        out[nc >> 1]  = (unsigned char)_v;  ++nc; } while (0)
    WQT2D_PUT_NIB(rootb);
    px[0] = 0; py[0] = 0; side[0] = 32; bb[0] = rootb; ch[0] = 0; sp = 1;
    while (sp > 0) {
        int i = sp - 1;
        int cs = side[i] >> 1;
        int found = -1;
        if (cs > 1) {                          // size-2 nodes carry cells, no node children
            for (int c = ch[i]; c < 4; ++c)
                if (bb[i] & (1 << c)) { found = c; break; }
        }
        if (found >= 0) {
            ch[i] = found + 1;
            int cx = found & 1, cy = (found >> 1) & 1;
            int ox = px[i] + cx * cs, oy = py[i] + cy * cs;
            int b = wqt2d_child_occ(m, ox, oy, cs);
            WQT2D_PUT_NIB(b);
            px[sp] = ox; py[sp] = oy; side[sp] = cs; bb[sp] = b; ch[sp] = 0; ++sp;
        } else {
            --sp;
        }
    }
#undef WQT2D_PUT_NIB
    return (nc + 1) >> 1;                       // packed byte count
}

// Inverse of wqt2d_encode: reconstruct row masks m[0..31] (caller pre-zeros).
// Reads one nibble per node in the same DFS order (self-terminating by tree
// structure); sigbytes is only used to short-circuit the empty block.
__host__ __device__ __forceinline__ void
wqt2d_decode(const unsigned char* in, int sigbytes, uint32_t* m) {
    if (sigbytes == 0) return;
    int px[6], py[6], side[6], bb[6], ch[6];
    int sp = 0, nc = 0;                         // nc: node index (nibble position)
    int rootb = in[0] & 0xF; nc = 1;
    px[0] = 0; py[0] = 0; side[0] = 32; bb[0] = rootb; ch[0] = 0; sp = 1;
    while (sp > 0) {
        int i = sp - 1;
        int cs = side[i] >> 1;
        int found = -1;
        for (int c = ch[i]; c < 4; ++c)
            if (bb[i] & (1 << c)) { found = c; break; }
        if (found < 0) { --sp; continue; }
        ch[i] = found + 1;
        int cx = found & 1, cy = (found >> 1) & 1;
        int ox = px[i] + cx * cs, oy = py[i] + cy * cs;
        if (cs == 1) {                          // cell leaf of a size-2 node
            m[oy] |= (1u << ox);
        } else {
            int b = (nc & 1) ? (in[nc >> 1] >> 4) : (in[nc >> 1] & 0xF); ++nc;
            px[sp] = ox; py[sp] = oy; side[sp] = cs; bb[sp] = b; ch[sp] = 0; ++sp;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-block PFOR value coder (host + device, byte-exact by construction).  The
// significant coefficients (nonzeros of m[], enumerated row-major, x ascending)
// are stored as three fixed-width streams so decode has no variable-bit chain:
//
//   [Wbyte]  low nibble = W_lo (base width), high nibble = W_hi (max width)
//   [base]   low W_lo bytes of every nonzero (little-endian), nnz*W_lo bytes
//   [mask]   1 exception bit per nonzero (set iff width > W_lo); present only
//            when W_lo < W_hi; ceil(nnz/8) bytes
//   [patch]  high (W_hi-W_lo) bytes of each exception, n_exc*(W_hi-W_lo) bytes
//
// W_lo is chosen to minimise total bytes (incl. mask).  When W_lo==W_hi there
// are no exceptions, so mask and patch are omitted (the sparse-block fast path).
// nnz is not stored; the decoder recovers it from the significance map.
// ---------------------------------------------------------------------------

// Encode grid[] nonzeros (mask m[]) into out; returns bytes written (0 if empty).
__host__ __device__ __forceinline__ int
wqt2d_pfor_encode(const int* g, const uint32_t* m, unsigned char* out) {
    int cnt[5] = {0, 0, 0, 0, 0}, nnz = 0, Whi = 0;
    for (int r = 0; r < 32; ++r) { uint32_t mm = m[r]; if (!mm) continue;
        for (int c = 0; c < 32; ++c) if (mm & (1u << c)) {
            int v = g[r * 32 + c]; unsigned a = (unsigned)(v < 0 ? -v : v);
            int w = wqt2d_width_bytes(a); ++cnt[w]; ++nnz; if (w > Whi) Whi = w;
        }
    }
    if (nnz == 0) return 0;

    int maskbytes = (nnz + 7) >> 3, Wlo = Whi; long best = -1;
    for (int wl = 1; wl <= Whi; ++wl) {
        int nexc = 0; for (int w = wl + 1; w <= 4; ++w) nexc += cnt[w];
        long cost = (long)nnz * wl + (long)nexc * (Whi - wl) + (wl < Whi ? maskbytes : 0);
        if (best < 0 || cost < best) { best = cost; Wlo = wl; }
    }

    out[0] = (unsigned char)((Whi << 4) | Wlo);
    int base_off = 1, have_mask = (Wlo < Whi);
    int mask_off = base_off + nnz * Wlo;
    int patch_off = mask_off + (have_mask ? maskbytes : 0);
    if (have_mask) for (int i = 0; i < maskbytes; ++i) out[mask_off + i] = 0;

    int i = 0, ex = 0;
    for (int r = 0; r < 32; ++r) { uint32_t mm = m[r]; if (!mm) continue;
        for (int c = 0; c < 32; ++c) if (mm & (1u << c)) {
            int v = g[r * 32 + c]; unsigned uv = (unsigned)v;
            for (int b = 0; b < Wlo; ++b) out[base_off + (long)i * Wlo + b] = (unsigned char)(uv >> (8 * b));
            unsigned a = (unsigned)(v < 0 ? -v : v);
            if (wqt2d_width_bytes(a) > Wlo) {          // exception
                out[mask_off + (i >> 3)] |= (unsigned char)(1u << (i & 7));
                for (int b = 0; b < Whi - Wlo; ++b)
                    out[patch_off + (long)ex * (Whi - Wlo) + b] = (unsigned char)(uv >> (8 * (Wlo + b)));
                ++ex;
            }
            ++i;
        }
    }
    return patch_off + ex * (Whi - Wlo);
}

// Inverse of wqt2d_pfor_encode: fill grid g[] (caller pre-zeros) from the mask m[].
__host__ __device__ __forceinline__ void
wqt2d_pfor_decode(const unsigned char* in, const uint32_t* m, int* g) {
    int nnz = 0;
    for (int r = 0; r < 32; ++r) { uint32_t mm = m[r]; while (mm) { nnz += (int)(mm & 1u); mm >>= 1; } }
    if (nnz == 0) return;

    int Wlo = in[0] & 0xF, Whi = (in[0] >> 4) & 0xF;
    int base_off = 1, have_mask = (Wlo < Whi);
    int mask_off = base_off + nnz * Wlo;
    int patch_off = mask_off + (have_mask ? ((nnz + 7) >> 3) : 0);
    int shlo = 32 - 8 * Wlo, shhi = 32 - 8 * Whi;

    int i = 0, ex = 0;
    for (int r = 0; r < 32; ++r) { uint32_t mm = m[r]; if (!mm) continue;
        for (int c = 0; c < 32; ++c) if (mm & (1u << c)) {
            unsigned uv = 0;
            for (int b = 0; b < Wlo; ++b) uv |= (unsigned)in[base_off + (long)i * Wlo + b] << (8 * b);
            int exc = have_mask && ((in[mask_off + (i >> 3)] >> (i & 7)) & 1);
            if (exc) {
                for (int b = 0; b < Whi - Wlo; ++b)
                    uv |= (unsigned)in[patch_off + (long)ex * (Whi - Wlo) + b] << (8 * (Wlo + b));
                g[r * 32 + c] = (int)(uv << shhi) >> shhi;   // sign-extend from W_hi
                ++ex;
            } else {
                g[r * 32 + c] = (int)(uv << shlo) >> shlo;   // sign-extend from W_lo
            }
            ++i;
        }
    }
}

// ===========================================================================
// k1 forward: 2D DS 7/9 wavelet + quantize -> int32 32x32 grid scratch.
// Grid/tile layout identical to waveletRLE2DFusedKernel; only the tail differs.
// ===========================================================================
__launch_bounds__(256, 2)
__global__ void waveletQuadtree2DForwardKernel(
    const float* __restrict__ input,
    int* __restrict__ grid_out,
    float scale,
    int ldimx,
    int nbx,
    const double* __restrict__ d_rms,
    float* __restrict__ d_mulfac_out)
{
    constexpr int SLC = 2;
    __shared__ float wavelet[WQT2D_BATCH * 1024];

    int tid = threadIdx.x;

    float mulfac;
    if (d_rms != nullptr) {
        float rms = (float)*d_rms;
        float product = rms * scale;
        mulfac = (product > 0.0f && __builtin_isfinite(1.0f / product))
                 ? (1.0f / product) : 1.0f;
        if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && d_mulfac_out)
            *d_mulfac_out = mulfac;
    } else {
        mulfac = scale;
    }

    int wg_tile_x0 = blockIdx.x * WQT2D_TILES_PER_WG;

    for (int batch = 0; batch < 4; ++batch) {
        int batch_tile_x0 = wg_tile_x0 + batch * WQT2D_BATCH;

        // ---- Load 8 tiles → LDS ----
        int xg = tid % 8, yr = tid / 8;
        for (int b = 0; b < WQT2D_BATCH; ++b) {
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
                wavelet[b * 1024 + (x0+0) * 32 + (yr ^ (x0+0))] = v[0];
                wavelet[b * 1024 + (x0+1) * 32 + (yr ^ (x0+1))] = v[1];
                wavelet[b * 1024 + (x0+2) * 32 + (yr ^ (x0+2))] = v[2];
                wavelet[b * 1024 + (x0+3) * 32 + (yr ^ (x0+3))] = v[3];
            } else {
                wavelet[b * 1024 + (x0+0) * 32 + (yr ^ (x0+0))] = 0.0f;
                wavelet[b * 1024 + (x0+1) * 32 + (yr ^ (x0+1))] = 0.0f;
                wavelet[b * 1024 + (x0+2) * 32 + (yr ^ (x0+2))] = 0.0f;
                wavelet[b * 1024 + (x0+3) * 32 + (yr ^ (x0+3))] = 0.0f;
            }
        }
        __syncthreads();

        // ---- Y-transform in LDS ----
        {
            int blk = tid / 32, pos = tid % 32;
            float line[32];
            for (int y = 0; y < 32; y++)
                line[y] = wavelet[blk * 1024 + pos * 32 + (y ^ pos)];
            ds79_forward_reg32(line);
            for (int y = 0; y < 32; y++)
                wavelet[blk * 1024 + pos * 32 + (y ^ pos)] = line[y];
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
                xline[x] = wavelet[blk * 1024 + x * 32 + (row ^ x)];
            ds79_forward_reg32(xline);
        } else {
            for (int x = 0; x < 32; x++) xline[x] = 0.0f;
        }
        __syncthreads();

        // ---- Quantize → int32 grid ----
        if (active) {
            int global_bid = tile_bx + blockIdx.y * nbx;
            int* g = grid_out + (size_t)global_bid * WQT2D_GRID_INTS + row * 32;
            #pragma unroll
            for (int x = 0; x < 32; x++) g[x] = (int)(mulfac * xline[x]);
        }
        __syncthreads();
    }
}

// ===========================================================================
// k2 code: int32 grid -> [mode][significance][row-width table][packed values].
// One workgroup per block; the significance + value passes run in thread 0
// (a 32x32 block is tiny), everything is byte-addressed so no alignment padding
// is required.  block_sizes[bid] := total coded length; sig_sizes[bid] :=
// significance-region length (for the self-contained header, like the octree).
// ===========================================================================
__launch_bounds__(WQT2D_CODE_THREADS)
__global__ void waveletQuadtree2DCodeKernel(
    const int* __restrict__ grid,
    unsigned char* __restrict__ out,
    size_t* __restrict__ block_sizes,
    size_t* __restrict__ sig_sizes)
{
    int bid = blockIdx.x, tid = threadIdx.x;
    __shared__ int g[WQT2D_GRID_INTS];
    __shared__ uint32_t m[32];

    const int* gin = grid + (size_t)bid * WQT2D_GRID_INTS;
    for (int i = tid; i < WQT2D_GRID_INTS; i += blockDim.x) g[i] = gin[i];
    __syncthreads();
    for (int r = tid; r < 32; r += blockDim.x) {
        uint32_t mm = 0;
        for (int c = 0; c < 32; ++c) if (g[r * 32 + c] != 0) mm |= (1u << c);
        m[r] = mm;
    }
    __syncthreads();

    if (tid == 0) {
        unsigned char* blk = out + (size_t)bid * WQT2D_CODE_SLOT_BYTES;
        int ob = wqt2d_encode(m, blk + WQT2D_HDR_BYTES);
        int siglen;
        if (ob < WQT2D_FLAT_SIG) {          // quadtree wins (includes empty ob==0)
            blk[0] = 2;
            siglen = ob;
        } else {                            // flat 32 uint32 row masks (little-endian)
            unsigned char* f = blk + WQT2D_HDR_BYTES;
            for (int r = 0; r < 32; ++r) {
                uint32_t mm = m[r];
                #pragma unroll
                for (int b = 0; b < 4; ++b) f[r * 4 + b] = (unsigned char)(mm >> (8 * b));
            }
            blk[0] = 0;
            siglen = WQT2D_FLAT_SIG;
        }
        sig_sizes[bid] = (size_t)siglen;

        int vlen = wqt2d_pfor_encode(g, m, blk + WQT2D_HDR_BYTES + siglen);
        block_sizes[bid] = (size_t)(WQT2D_HDR_BYTES + siglen + vlen);
    }
}

// Compaction: copy variable-length coded blocks from the fixed-stride
// intermediate into a tightly packed payload and write the self-contained
// header [int nb][int nmf][size_t offsets[nb]][uint32 sig_sizes[nb]][float mf[nmf]]
// (== hipOctreeHeaderSize).  Mirrors woctCompactKernel with the 2D slot stride.
__global__ void wqt2dCompactKernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    const size_t* __restrict__ block_sizes,
    const size_t* __restrict__ offsets,
    const size_t* __restrict__ sig_sizes,
    unsigned char* __restrict__ hdr,
    int num_blocks,
    int num_mulfacs,
    const float* __restrict__ d_mulfac)
{
    int bid = blockIdx.x, tid = threadIdx.x;
    size_t size = block_sizes[bid];
    size_t dst_off = offsets[bid];
    size_t src_off = (size_t)bid * WQT2D_CODE_SLOT_BYTES;

    if (hdr != nullptr && tid == 0) {
        ((size_t*)(hdr + 8))[bid] = offsets[bid];
        ((uint32_t*)(hdr + 8 + 8L * num_blocks))[bid] = (uint32_t)sig_sizes[bid];
        if (bid == 0) {
            ((int*)hdr)[0] = num_blocks;
            ((int*)hdr)[1] = num_mulfacs;
            float* mf_dst = (float*)(hdr + 8 + 12L * num_blocks);
            for (int i = 0; i < num_mulfacs; ++i) mf_dst[i] = d_mulfac[i];
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

// ===========================================================================
// Decode stage A: compacted coded stream -> int32 grid scratch.  Locates each
// block via the header offset table, reads its significance length from the
// header, reconstructs the 32x32 quantized grid, and (block 0) publishes
// inv_scale = 1/mulfac (device) for stage B.
// ===========================================================================
__launch_bounds__(WQT2D_CODE_THREADS)
__global__ void waveletQuadtree2DDecodeHdrKernel(
    const unsigned char* __restrict__ input,
    int* __restrict__ grid_out,
    float* __restrict__ inv_scale_out)
{
    int bid = blockIdx.x, tid = threadIdx.x;
    const int* hdr = reinterpret_cast<const int*>(input);
    int num_blocks  = hdr[0];
    int num_mulfacs = hdr[1];
    const size_t*   offsets = reinterpret_cast<const size_t*>(input + 8);
    const uint32_t* sig_u32 = reinterpret_cast<const uint32_t*>(input + 8 + 8L * num_blocks);
    const float*    mulfacs = reinterpret_cast<const float*>(input + 8 + 12L * num_blocks);
    const unsigned char* data_base = input + 8 + 12L * num_blocks + 4L * num_mulfacs;

    if (bid == 0 && tid == 0 && inv_scale_out)
        *inv_scale_out = 1.0f / mulfacs[0];

    __shared__ int g[WQT2D_GRID_INTS];
    __shared__ uint32_t m[32];

    for (int i = tid; i < WQT2D_GRID_INTS; i += blockDim.x) g[i] = 0;
    for (int r = tid; r < 32; r += blockDim.x) m[r] = 0;
    __syncthreads();

    const unsigned char* blk = data_base + offsets[bid];
    int siglen = (int)sig_u32[bid];

    if (tid == 0) {
        int mode = blk[0];
        if (mode == 0) {
            const unsigned char* f = blk + WQT2D_HDR_BYTES;
            for (int r = 0; r < 32; ++r) {
                uint32_t mm = 0;
                #pragma unroll
                for (int b = 0; b < 4; ++b) mm |= (uint32_t)f[r * 4 + b] << (8 * b);
                m[r] = mm;
            }
        } else {
            wqt2d_decode(blk + WQT2D_HDR_BYTES, siglen, m);
        }

        wqt2d_pfor_decode(blk + WQT2D_HDR_BYTES + siglen, m, g);
    }
    __syncthreads();

    int* gout = grid_out + (size_t)bid * WQT2D_GRID_INTS;
    for (int i = tid; i < WQT2D_GRID_INTS; i += blockDim.x) gout[i] = g[i];
}

// ===========================================================================
// Decode stage B: int32 grid -> dequantize + inverse 2D DS 7/9 wavelet.
// Transform is byte-for-byte waveletRLE2DInverseFusedKernel with the RLE decode
// replaced by an int32-grid read scaled by inv_scale (read from device).
// ===========================================================================
__launch_bounds__(256, 2)
__global__ void waveletQuadtree2DInverseKernel(
    const int* __restrict__ grid,
    float* __restrict__ output,
    const float* __restrict__ inv_scale_dev,
    int ldimx,
    int nbx)
{
    constexpr int SLC = 2;
    constexpr int MACRO_BATCHES = 2;
    using f4vec = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    __shared__ float wavelet[WQT2D_BATCH * 1024];

    int tid = threadIdx.x;
    int wg_tile_x0 = blockIdx.x * WQT2D_TILES_PER_WG;
    int blk = tid / 32;
    int row = tid % 32;

    float inv_scale = inv_scale_dev ? *inv_scale_dev : 1.0f;

    for (int mp = 0; mp < 4 / MACRO_BATCHES; ++mp) {

        float decoded[MACRO_BATCHES][32];

        // ---- Grid read + X-inverse ----
        for (int b = 0; b < MACRO_BATCHES; ++b) {
            int batch = mp * MACRO_BATCHES + b;
            int batch_tile_x0 = wg_tile_x0 + batch * WQT2D_BATCH;
            int tile_bx = batch_tile_x0 + blk;
            int global_bid = tile_bx + blockIdx.y * nbx;
            bool active = (tile_bx < nbx);

            if (active) {
                const int* g = grid + (size_t)global_bid * WQT2D_GRID_INTS + row * 32;
                #pragma unroll
                for (int x = 0; x < 32; x++) decoded[b][x] = (float)g[x] * inv_scale;
                us79_inverse_reg32(decoded[b]);
            } else {
                #pragma unroll
                for (int x = 0; x < 32; x++) decoded[b][x] = 0.0f;
            }
        }
        __syncthreads();

        // ---- Y-inverse + store ----
        for (int b = 0; b < MACRO_BATCHES; ++b) {
            int batch = mp * MACRO_BATCHES + b;
            int batch_tile_x0 = wg_tile_x0 + batch * WQT2D_BATCH;

            for (int x = 0; x < 32; x++)
                wavelet[blk * 1024 + x * 32 + (row ^ x)] = decoded[b][x];
            __syncthreads();

            {
                int pl  = tid / 32;
                int pos = tid % 32;
                float line[32];
                for (int y = 0; y < 32; y++)
                    line[y] = wavelet[pl * 1024 + pos * 32 + (y ^ pos)];
                us79_inverse_reg32(line);
                for (int y = 0; y < 32; y++)
                    wavelet[pl * 1024 + pos * 32 + (y ^ pos)] = line[y];
            }
            __syncthreads();

            f4vec store_regs[WQT2D_BATCH];
            {
                int xg = tid % 8, yr = tid / 8, x0 = xg * 4;
                #pragma unroll
                for (int s = 0; s < WQT2D_BATCH; ++s) {
                    store_regs[s][0] = wavelet[s * 1024 + (x0+0) * 32 + (yr ^ (x0+0))];
                    store_regs[s][1] = wavelet[s * 1024 + (x0+1) * 32 + (yr ^ (x0+1))];
                    store_regs[s][2] = wavelet[s * 1024 + (x0+2) * 32 + (yr ^ (x0+2))];
                    store_regs[s][3] = wavelet[s * 1024 + (x0+3) * 32 + (yr ^ (x0+3))];
                }
            }
            __syncthreads();

            {
                int xg = tid % 8, yr = tid / 8;
                #pragma unroll
                for (int s = 0; s < WQT2D_BATCH; ++s) {
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

#endif // HIPWAVELET_QUADTREE_2D_H
