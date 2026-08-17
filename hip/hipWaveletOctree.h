// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// PROTOTYPE: octree significance coder (kernel-2 variant).  Replaces the flat
// 4096 B/block significance bitmap (or the two-level occupancy+masks) with a
// full 3D octree of the 32^3 significance set: quadtree in (x,y) x bisection in
// z, depth 5 (node sizes 32,16,8,4,2), 8 bits (one per child) per non-empty
// node, serialized in a fixed DFS pre-order.  A per-block 1-byte mode tag falls
// back to the flat bitmap when the octree would be larger (dense blocks).
//
// The encode/decode core is a single __host__ __device__ function with an
// explicit depth-5 stack, so the CPU reference and the GPU kernel produce
// byte-for-byte identical streams by construction.  Both operate on the 32x32
// z-line masks produced by waveletBitmapFusedKernel (kernel 1).
//
// Coded block layout:  [1B mode][significance][2b/nonempty-line width table]
//                      [per-line packed values]
//   mode 0 (flat)   : significance = 1024 uint32 masks in L order  (4096 B)
//   mode 2 (octree) : significance = variable-length octree stream
// The width table and value payload are identical to the two-level coder, so
// only the significance representation is new here.

#ifndef HIPWAVELET_OCTREE_H
#define HIPWAVELET_OCTREE_H

#include <hip/hip_runtime.h>
#include "hipWaveletBitmap.h"   // WBMP_* layout constants

// Worst-case (fully dense 32^3) octree node count = 4096+512+64+8+1 = 4681.
static constexpr int  WOCT_MAX_SIG_BYTES = 4681;
static constexpr int  WOCT_FLAT_SIG_BYTES = WBMP_BITMAP_BYTES;   // 4096
// 4-byte mode header keeps the significance region 4-aligned (the flat variant
// stores 1024 uint32 masks; unaligned uint32 access breaks on device).
static constexpr int  WOCT_HDR_BYTES = 4;
// Coded slot: header + max significance + width table + max packed values,
// rounded up to 16 B so every per-block slot (and thus the 4-aligned
// significance region) stays aligned for uint32 flat masks / vectorized stores.
static constexpr long WOCT_SLOT_RAW =
    WOCT_HDR_BYTES + (WOCT_MAX_SIG_BYTES > WOCT_FLAT_SIG_BYTES ? WOCT_MAX_SIG_BYTES : WOCT_FLAT_SIG_BYTES)
      + WBMP_WTAB_BYTES + WBMP_MAX_VAL_BYTES;
static constexpr long WOCT_SLOT_BYTES = (WOCT_SLOT_RAW + 15) & ~15L;

// Fused kernel-2 coded slot: [4B mode][significance][<=3B pad to 4-align the
// width table][2b/nonempty-line width table][per-line packed values].
//   sig    = octree stream (<=WOCT_MAX_SIG_BYTES) or flat masks (4096 B)
//   values = same per-line variable-width payload as the two-level coder
static constexpr long WOCT_CODE_SLOT_RAW =
    WOCT_HDR_BYTES + WOCT_MAX_SIG_BYTES + 3 + WBMP_WTAB_BYTES + WBMP_MAX_VAL_BYTES;
static constexpr long WOCT_CODE_SLOT_BYTES = (WOCT_CODE_SLOT_RAW + 15) & ~15L;

// Kernel-1 bitmap word order L=x_off*256+tid maps to spatial line (ix,iy):
//   tid = L & 255 ; xg = tid & 7 ; yr = tid >> 3 ; ix = xg*4 + (L>>8) ; iy = yr
__host__ __device__ __forceinline__ int woct_L_to_spatial(int L) {
    int x_off = L >> 8, tid = L & 255, xg = tid & 7, yr = tid >> 3;
    return yr * 32 + (xg * 4 + x_off);   // spatial index iy*32+ix
}

// True if any significant voxel lies in the [ox,ox+cs) x [oy,oy+cs) x [oz,oz+cs)
// subcube of the block, given the 32 spatial z-line masks m[iy*32+ix].
__host__ __device__ __forceinline__ bool
woct_subcube_any(const uint32_t* m, int ox, int oy, int oz, int cs) {
    uint32_t zmask = (cs >= 32) ? 0xFFFFFFFFu
                                : (uint32_t)((((uint64_t)1 << cs) - 1) << oz);
    for (int iy = oy; iy < oy + cs; ++iy)
        for (int ix = ox; ix < ox + cs; ++ix)
            if (m[iy * 32 + ix] & zmask) return true;
    return false;
}

// 8-bit child-occupancy of a node of the given size at (px,py,pz).  child index
// c = cx | cy<<1 | cz<<2 ; child subcube side = side/2.
__host__ __device__ __forceinline__ int
woct_child_occ(const uint32_t* m, int px, int py, int pz, int side) {
    int cs = side >> 1, b = 0;
    for (int c = 0; c < 8; ++c) {
        int cx = c & 1, cy = (c >> 1) & 1, cz = (c >> 2) & 1;
        if (woct_subcube_any(m, px + cx * cs, py + cy * cs, pz + cz * cs, cs)) b |= (1 << c);
    }
    return b;
}

// Explicit depth-<=5 DFS stack kept as parallel scalar arrays (no struct
// references) so the device compiler serializes it correctly.
// Encode the 32^3 significance (spatial z-line masks) into a DFS-preorder octree
// byte stream.  Returns the number of bytes written (0 for an empty block).
__host__ __device__ __forceinline__ int
woct_encode(const uint32_t* m, unsigned char* out) {
    int rootb = woct_child_occ(m, 0, 0, 0, 32);
    if (rootb == 0) return 0;                 // empty block -> zero-length stream
    int px[8], py[8], pz[8], side[8], bb[8], ch[8];
    int sp = 0, pos = 0;
    out[pos++] = (unsigned char)rootb;
    px[0]=0; py[0]=0; pz[0]=0; side[0]=32; bb[0]=rootb; ch[0]=0; sp = 1;
    while (sp > 0) {
        int i = sp - 1;
        int cs = side[i] >> 1;
        int found = -1;
        if (cs > 1) {                          // size-2 nodes carry voxels, no node children
            for (int c = ch[i]; c < 8; ++c)
                if (bb[i] & (1 << c)) { found = c; break; }
        }
        if (found >= 0) {
            ch[i] = found + 1;
            int cx = found & 1, cy = (found >> 1) & 1, cz = (found >> 2) & 1;
            int ox = px[i] + cx * cs, oy = py[i] + cy * cs, oz = pz[i] + cz * cs;
            int b = woct_child_occ(m, ox, oy, oz, cs);
            out[pos++] = (unsigned char)b;
            px[sp]=ox; py[sp]=oy; pz[sp]=oz; side[sp]=cs; bb[sp]=b; ch[sp]=0; ++sp;
        } else {
            --sp;
        }
    }
    return pos;
}

// Inverse of woct_encode: reconstruct the spatial z-line masks m[iy*32+ix]
// (which the caller must pre-zero).  sigbytes==0 means an empty block.
__host__ __device__ __forceinline__ void
woct_decode(const unsigned char* in, int sigbytes, uint32_t* m) {
    if (sigbytes == 0) return;
    int px[8], py[8], pz[8], side[8], bb[8], ch[8];
    int sp = 0, pos = 0;
    px[0]=0; py[0]=0; pz[0]=0; side[0]=32; bb[0]=in[pos++]; ch[0]=0; sp = 1;
    while (sp > 0) {
        int i = sp - 1;
        int cs = side[i] >> 1;
        int found = -1;
        for (int c = ch[i]; c < 8; ++c)
            if (bb[i] & (1 << c)) { found = c; break; }
        if (found < 0) { --sp; continue; }
        ch[i] = found + 1;
        int cx = found & 1, cy = (found >> 1) & 1, cz = (found >> 2) & 1;
        int ox = px[i] + cx * cs, oy = py[i] + cy * cs, oz = pz[i] + cz * cs;
        if (cs == 1) {                         // voxel leaf of a size-2 node
            m[oy * 32 + ox] |= (1u << oz);
        } else {
            px[sp]=ox; py[sp]=oy; pz[sp]=oz; side[sp]=cs; bb[sp]=in[pos++]; ch[sp]=0; ++sp;
        }
    }
}

// ===========================================================================
// Level-major (parallel-friendly) octree format.
//
// The DFS pre-order above is inherently serial to both emit and parse.  The
// level-major layout below encodes the *same node set* (so the byte count, and
// thus the compression ratio, is byte-for-byte identical to woct_encode) but in
// breadth-first order: all size-16 node bytes, then all size-8, size-4, size-2,
// each level's present nodes listed in canonical linear index (x fastest).
// That order is producible/consumable in parallel via per-level prefix sums,
// which the GPU kernels below exploit.  These host references define the
// canonical stream and serve as the byte-exact oracle for the parallel kernels.
//
// Node grids (occupancy):  g16[2^3], g8[4^3], g4[8^3], g2[16^3]; strides x=1,
// y=dim, z=dim^2.  voxel(x,y,z) = (m[y*32+x]>>z)&1.
// ---------------------------------------------------------------------------

// Occupancy grids for the four interior levels (1 = subcube contains a voxel).
struct WoctGrids { unsigned char g16[8], g8[64], g4[512], g2[4096]; int rootocc; };

static inline void woct_build_grids(const uint32_t* m, WoctGrids& g) {
    // g2: OR over each 2x2x2 voxel cube.
    for (int k = 0; k < 16; ++k) for (int j = 0; j < 16; ++j) for (int i = 0; i < 16; ++i) {
        int occ = 0;
        for (int c = 0; c < 8 && !occ; ++c) {
            int x = 2*i + (c&1), y = 2*j + ((c>>1)&1), z = 2*k + ((c>>2)&1);
            occ = (m[y*32+x] >> z) & 1u;
        }
        g.g2[k*256 + j*16 + i] = (unsigned char)occ;
    }
    // g4,g8,g16: OR over each 2x2x2 subcube of the finer grid.
    auto pool = [](const unsigned char* fine, int fdim, unsigned char* coarse, int cdim) {
        for (int k = 0; k < cdim; ++k) for (int j = 0; j < cdim; ++j) for (int i = 0; i < cdim; ++i) {
            int occ = 0;
            for (int c = 0; c < 8 && !occ; ++c) {
                int x = 2*i + (c&1), y = 2*j + ((c>>1)&1), z = 2*k + ((c>>2)&1);
                occ = fine[z*fdim*fdim + y*fdim + x];
            }
            coarse[k*cdim*cdim + j*cdim + i] = (unsigned char)occ;
        }
    };
    pool(g.g2, 16, g.g4, 8);
    pool(g.g4, 8,  g.g8, 4);
    pool(g.g8, 4,  g.g16, 2);
    g.rootocc = 0;
    for (int n = 0; n < 8; ++n) g.rootocc |= g.g16[n];
}

// 8-bit child-occupancy byte of node (Ix,Iy,Iz) whose children live in the
// finer grid `fine` of dimension fdim (child at 2I+c).
static inline int woct_byte_from_grid(const unsigned char* fine, int fdim,
                                      int Ix, int Iy, int Iz) {
    int b = 0;
    for (int c = 0; c < 8; ++c) {
        int x = 2*Ix + (c&1), y = 2*Iy + ((c>>1)&1), z = 2*Iz + ((c>>2)&1);
        if (fine[z*fdim*fdim + y*fdim + x]) b |= (1 << c);
    }
    return b;
}

// Level-major encode.  Returns bytes written (== woct_encode's count).
static inline int woct_encode_lm(const uint32_t* m, unsigned char* out) {
    WoctGrids g; woct_build_grids(m, g);
    if (!g.rootocc) return 0;
    int pos = 0;
    // root byte = size-16 children occupancy.
    out[pos++] = (unsigned char)woct_byte_from_grid(g.g16, 2, 0, 0, 0);
    // size-16 nodes -> bytes over g8.
    for (int n = 0; n < 8; ++n) if (g.g16[n]) {
        int Ix = n&1, Iy = (n>>1)&1, Iz = (n>>2)&1;
        out[pos++] = (unsigned char)woct_byte_from_grid(g.g8, 4, Ix, Iy, Iz);
    }
    // size-8 nodes -> bytes over g4.
    for (int n = 0; n < 64; ++n) if (g.g8[n]) {
        int Ix = n&3, Iy = (n>>2)&3, Iz = (n>>4)&3;
        out[pos++] = (unsigned char)woct_byte_from_grid(g.g4, 8, Ix, Iy, Iz);
    }
    // size-4 nodes -> bytes over g2.
    for (int n = 0; n < 512; ++n) if (g.g4[n]) {
        int Ix = n&7, Iy = (n>>3)&7, Iz = (n>>6)&7;
        out[pos++] = (unsigned char)woct_byte_from_grid(g.g2, 16, Ix, Iy, Iz);
    }
    // size-2 nodes -> voxel bytes (children are the voxels themselves).
    for (int n = 0; n < 4096; ++n) if (g.g2[n]) {
        int Ix = n&15, Iy = (n>>4)&15, Iz = (n>>8)&15, b = 0;
        for (int c = 0; c < 8; ++c) {
            int x = 2*Ix + (c&1), y = 2*Iy + ((c>>1)&1), z = 2*Iz + ((c>>2)&1);
            if ((m[y*32+x] >> z) & 1u) b |= (1 << c);
        }
        out[pos++] = (unsigned char)b;
    }
    return pos;
}

// Inverse of woct_encode_lm.  Caller pre-zeros m.
static inline void woct_decode_lm(const unsigned char* in, int sigbytes, uint32_t* m) {
    if (sigbytes == 0) return;
    unsigned char g16[8]={0}, g8[64]={0}, g4[512]={0}, g2[4096]={0};
    int pos = 0;
    // root -> size-16 present set.
    { int b = in[pos++]; for (int c = 0; c < 8; ++c) if (b & (1<<c)) g16[c] = 1; }
    // size-16 bytes -> size-8 present set.
    for (int n = 0; n < 8; ++n) if (g16[n]) {
        int Ix = n&1, Iy = (n>>1)&1, Iz = (n>>2)&1, b = in[pos++];
        for (int c = 0; c < 8; ++c) if (b & (1<<c)) {
            int x = 2*Ix+(c&1), y = 2*Iy+((c>>1)&1), z = 2*Iz+((c>>2)&1);
            g8[z*16 + y*4 + x] = 1;
        }
    }
    // size-8 bytes -> size-4 present set.
    for (int n = 0; n < 64; ++n) if (g8[n]) {
        int Ix = n&3, Iy = (n>>2)&3, Iz = (n>>4)&3, b = in[pos++];
        for (int c = 0; c < 8; ++c) if (b & (1<<c)) {
            int x = 2*Ix+(c&1), y = 2*Iy+((c>>1)&1), z = 2*Iz+((c>>2)&1);
            g4[z*64 + y*8 + x] = 1;
        }
    }
    // size-4 bytes -> size-2 present set.
    for (int n = 0; n < 512; ++n) if (g4[n]) {
        int Ix = n&7, Iy = (n>>3)&7, Iz = (n>>6)&7, b = in[pos++];
        for (int c = 0; c < 8; ++c) if (b & (1<<c)) {
            int x = 2*Ix+(c&1), y = 2*Iy+((c>>1)&1), z = 2*Iz+((c>>2)&1);
            g2[z*256 + y*16 + x] = 1;
        }
    }
    // size-2 bytes -> voxels.
    for (int n = 0; n < 4096; ++n) if (g2[n]) {
        int Ix = n&15, Iy = (n>>4)&15, Iz = (n>>8)&15, b = in[pos++];
        for (int c = 0; c < 8; ++c) if (b & (1<<c)) {
            int x = 2*Ix+(c&1), y = 2*Iy+((c>>1)&1), z = 2*Iz+((c>>2)&1);
            m[y*32+x] |= (1u << z);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: per-block octree significance encode with flat fallback.
// Reads kernel-1 output [4096B bitmap][packed int32]; writes
// out[bid*WOCT_SLOT_BYTES] = [1B mode][significance...] and records the
// significance length in sig_sizes[bid].  (Width table + values are appended by
// a separate pass reusing the two-level packer; omitted from this prototype,
// which validates the significance round-trip.)
//
// The volume gather is parallel; the DFS serialization is done by thread 0 so
// the stream is byte-identical to the host reference (prototype: not yet a
// parallel serializer).
// ---------------------------------------------------------------------------
__global__ void waveletOctreeSigEncodeKernel(
    const unsigned char* __restrict__ scratch1,
    unsigned char* __restrict__ out,
    size_t* __restrict__ sig_sizes)
{
    int bid = blockIdx.x, tid = threadIdx.x;
    const uint32_t* bmp = reinterpret_cast<const uint32_t*>(scratch1 + (long)bid * WBMP_SLOT_BYTES);

    __shared__ uint32_t masks_sp[1024];
    for (int L = tid; L < 1024; L += blockDim.x) masks_sp[woct_L_to_spatial(L)] = bmp[L];
    __syncthreads();

    if (tid == 0) {
        unsigned char* blk = out + (long)bid * WOCT_SLOT_BYTES;
        int ob = woct_encode(masks_sp, blk + WOCT_HDR_BYTES);
        if (ob < WOCT_FLAT_SIG_BYTES) {
            blk[0] = 2;                        // octree
            sig_sizes[bid] = (size_t)ob;
        } else {
            uint32_t* flat = reinterpret_cast<uint32_t*>(blk + WOCT_HDR_BYTES);
            for (int L = 0; L < 1024; ++L) flat[L] = bmp[L];   // flat masks in L order
            blk[0] = 0;                        // flat
            sig_sizes[bid] = (size_t)WOCT_FLAT_SIG_BYTES;
        }
    }
}

// GPU kernel: decode significance -> reconstruct 1024 z-line masks in L order
// into out_masks[bid*1024 + L] for byte-exact comparison against the bitmap.
__global__ void waveletOctreeSigDecodeKernel(
    const unsigned char* __restrict__ coded,
    const size_t* __restrict__ sig_sizes,
    uint32_t* __restrict__ out_masks)
{
    int bid = blockIdx.x, tid = threadIdx.x;
    const unsigned char* blk = coded + (long)bid * WOCT_SLOT_BYTES;

    __shared__ uint32_t masks_sp[1024];
    for (int L = tid; L < 1024; L += blockDim.x) masks_sp[L] = 0;
    __syncthreads();

    if (tid == 0) {
        int mode = blk[0];
        if (mode == 0) {
            const uint32_t* flat = reinterpret_cast<const uint32_t*>(blk + WOCT_HDR_BYTES);
            for (int L = 0; L < 1024; ++L) masks_sp[woct_L_to_spatial(L)] = flat[L];
        } else {
            woct_decode(blk + WOCT_HDR_BYTES, (int)sig_sizes[bid], masks_sp);
        }
    }
    __syncthreads();

    for (int L = tid; L < 1024; L += blockDim.x)
        out_masks[(long)bid * 1024 + L] = masks_sp[woct_L_to_spatial(L)];
}

// ===========================================================================
// Parallel level-major octree kernels (1024 threads/block).
//
// Both kernels build the occupancy pyramid (o16/o8/o4/o2) cooperatively in LDS,
// then use per-level inclusive prefix sums to place / locate each present node's
// byte in the level-major stream defined by woct_encode_lm.  The byte count (and
// thus CR) is identical to the DFS serial coder; only the emission order and the
// degree of parallelism differ.  WOCT_PAR_THREADS must be 1024.
// ===========================================================================
static constexpr int WOCT_PAR_THREADS = 1024;

// In-place inclusive Hillis-Steele scan of a[0..N) using 1024 threads (up to 4
// elements/thread; N<=4096).  Two syncs/step make it safe in place.
__device__ __forceinline__ void woct_incl_scan(int* a, int N) {
    int tid = threadIdx.x;
    for (int off = 1; off < N; off <<= 1) {
        int v0=0,v1=0,v2=0,v3=0;
        int i0=tid,i1=tid+1024,i2=tid+2048,i3=tid+3072;
        if (i0<N && i0>=off) v0=a[i0-off];
        if (i1<N && i1>=off) v1=a[i1-off];
        if (i2<N && i2>=off) v2=a[i2-off];
        if (i3<N && i3>=off) v3=a[i3-off];
        __syncthreads();
        if (i0<N) a[i0]+=v0;
        if (i1<N) a[i1]+=v1;
        if (i2<N) a[i2]+=v2;
        if (i3<N) a[i3]+=v3;
        __syncthreads();
    }
}

// Fast inclusive scan of occ[0..N)->r[0..N) (lidx order), N in {8,64,512,4096}.
// Uses a blocked layout (thread t owns the contiguous chunk [t*C, t*C+C), C=N/1024
// rounded up) so a single wave64 DPP block scan (wbmp_opt::block_exscan) replaces
// the ~log2(N) Hillis-Steele passes.  r[n] holds the inclusive prefix at n exactly
// as woct_incl_scan, so downstream (rank = r[n]-occ[n], count = total) is
// unchanged.  Returns the total (== popcount of occ).  warp_part is 2*16 ints.
__device__ __forceinline__ int
woct_incl_scan_fast(const int* occ, int* r, int N, int* warp_part) {
    int tid = threadIdx.x;
    int C = (N + 1023) >> 10;          // 1 (N<=1024) or 4 (N==4096)
    int base = tid * C;
    int l0=0,l1=0,l2=0,l3=0,lsum=0;
    if (C == 1) {
        if (base < N) { lsum = occ[base]; l0 = lsum; }
    } else {                           // C == 4
        if (base+0 < N) { lsum += occ[base+0]; l0 = lsum; }
        if (base+1 < N) { lsum += occ[base+1]; l1 = lsum; }
        if (base+2 < N) { lsum += occ[base+2]; l2 = lsum; }
        if (base+3 < N) { lsum += occ[base+3]; l3 = lsum; }
    }
    int total;
    int ex = wbmp_opt::block_exscan(lsum, total, warp_part);
    if (C == 1) {
        if (base < N) r[base] = ex + l0;
    } else {
        if (base+0 < N) r[base+0] = ex + l0;
        if (base+1 < N) r[base+1] = ex + l1;
        if (base+2 < N) r[base+2] = ex + l2;
        if (base+3 < N) r[base+3] = ex + l3;
    }
    __syncthreads();                   // protect warp_part reuse + publish r
    return total;
}

// Shared occupancy pyramid + rank buffers reused by both parallel kernels.
struct WoctShared {
    uint32_t masks_sp[1024];
    int o2[4096], o4[512], o8[64], o16[8];   // occupancy (0/1)
    int r2[4096], r4[512], r8[64], r16[8];   // inclusive scans of occupancy
};

// Cooperatively build o2/o4/o8/o16 from masks_sp.  Requires __syncthreads() by
// caller before use; issues its own syncs between levels.
__device__ __forceinline__ void woct_build_pyramid(WoctShared& s) {
    int tid = threadIdx.x;
    for (int n = tid; n < 4096; n += 1024) {
        int Ix=n&15, Iy=(n>>4)&15, Iz=(n>>8)&15, occ=0;
        for (int c=0;c<8 && !occ;++c){
            int x=2*Ix+(c&1), y=2*Iy+((c>>1)&1), z=2*Iz+((c>>2)&1);
            occ = (s.masks_sp[y*32+x] >> z) & 1u;
        }
        s.o2[n]=occ;
    }
    __syncthreads();
    for (int n = tid; n < 512; n += 1024) {   // o4 from o2 (df=16)
        int Ix=n&7, Iy=(n>>3)&7, Iz=(n>>6)&7, occ=0;
        for (int c=0;c<8 && !occ;++c){
            int x=2*Ix+(c&1), y=2*Iy+((c>>1)&1), z=2*Iz+((c>>2)&1);
            occ = s.o2[z*256 + y*16 + x];
        }
        s.o4[n]=occ;
    }
    __syncthreads();
    for (int n = tid; n < 64; n += 1024) {    // o8 from o4 (df=8)
        int Ix=n&3, Iy=(n>>2)&3, Iz=(n>>4)&3, occ=0;
        for (int c=0;c<8 && !occ;++c){
            int x=2*Ix+(c&1), y=2*Iy+((c>>1)&1), z=2*Iz+((c>>2)&1);
            occ = s.o4[z*64 + y*8 + x];
        }
        s.o8[n]=occ;
    }
    __syncthreads();
    for (int n = tid; n < 8; n += 1024) {     // o16 from o8 (df=4)
        int Ix=n&1, Iy=(n>>1)&1, Iz=(n>>2)&1, occ=0;
        for (int c=0;c<8 && !occ;++c){
            int x=2*Ix+(c&1), y=2*Iy+((c>>1)&1), z=2*Iz+((c>>2)&1);
            occ = s.o8[z*16 + y*4 + x];
        }
        s.o16[n]=occ;
    }
    __syncthreads();
}

// Parallel level-major encode: byte-for-byte identical stream to woct_encode_lm.
__global__ void waveletOctreeSigEncodeParKernel(
    const unsigned char* __restrict__ scratch1,
    unsigned char* __restrict__ out,
    size_t* __restrict__ sig_sizes)
{
    int bid = blockIdx.x, tid = threadIdx.x;
    const uint32_t* bmp = reinterpret_cast<const uint32_t*>(scratch1 + (long)bid * WBMP_SLOT_BYTES);
    unsigned char* blk = out + (long)bid * WOCT_SLOT_BYTES;

    __shared__ WoctShared s;
    __shared__ int n16,n8,n4,n2;
    for (int L = tid; L < 1024; L += 1024) s.masks_sp[woct_L_to_spatial(L)] = bmp[L];
    __syncthreads();

    woct_build_pyramid(s);

    // Root occupancy: any size-16 node present.
    int rootocc = 0;
    for (int i=0;i<8;++i) rootocc |= s.o16[i];
    if (!rootocc) {                            // empty block -> zero-length stream
        if (tid==0){ blk[0]=2; sig_sizes[bid]=0; }
        return;
    }

    // Per-level inclusive scans -> counts.
    for (int i=tid;i<8;   i+=1024) s.r16[i]=s.o16[i];
    for (int i=tid;i<64;  i+=1024) s.r8[i] =s.o8[i];
    for (int i=tid;i<512; i+=1024) s.r4[i] =s.o4[i];
    for (int i=tid;i<4096;i+=1024) s.r2[i] =s.o2[i];
    __syncthreads();
    woct_incl_scan(s.r16,8);
    woct_incl_scan(s.r8,64);
    woct_incl_scan(s.r4,512);
    woct_incl_scan(s.r2,4096);
    if (tid==0){ n16=s.r16[7]; n8=s.r8[63]; n4=s.r4[511]; n2=s.r2[4095]; }
    __syncthreads();

    int oct_total = 1 + n16 + n8 + n4 + n2;
    if (oct_total >= WOCT_FLAT_SIG_BYTES) {    // dense -> flat fallback
        uint32_t* flat = reinterpret_cast<uint32_t*>(blk + WOCT_HDR_BYTES);
        for (int L=tid; L<1024; L+=1024) flat[L]=bmp[L];
        if (tid==0){ blk[0]=0; sig_sizes[bid]=(size_t)WOCT_FLAT_SIG_BYTES; }
        return;
    }

    unsigned char* sig = blk + WOCT_HDR_BYTES;
    int base16=1, base8=1+n16, base4=1+n16+n8, base2=1+n16+n8+n4;
    if (tid==0){
        blk[0]=2; sig_sizes[bid]=(size_t)oct_total;
        int rb=0; for (int c=0;c<8;++c) if (s.o16[c]) rb|=(1<<c);
        sig[0]=(unsigned char)rb;               // root byte
    }
    // size-16 -> bytes over o8.
    for (int n=tid;n<8;n+=1024) if (s.o16[n]) {
        int Ix=n&1,Iy=(n>>1)&1,Iz=(n>>2)&1,b=0;
        for (int c=0;c<8;++c){int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1);
            if (s.o8[z*16+y*4+x]) b|=(1<<c);}
        sig[base16 + (s.r16[n]-s.o16[n])]=(unsigned char)b;
    }
    // size-8 -> bytes over o4.
    for (int n=tid;n<64;n+=1024) if (s.o8[n]) {
        int Ix=n&3,Iy=(n>>2)&3,Iz=(n>>4)&3,b=0;
        for (int c=0;c<8;++c){int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1);
            if (s.o4[z*64+y*8+x]) b|=(1<<c);}
        sig[base8 + (s.r8[n]-s.o8[n])]=(unsigned char)b;
    }
    // size-4 -> bytes over o2.
    for (int n=tid;n<512;n+=1024) if (s.o4[n]) {
        int Ix=n&7,Iy=(n>>3)&7,Iz=(n>>6)&7,b=0;
        for (int c=0;c<8;++c){int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1);
            if (s.o2[z*256+y*16+x]) b|=(1<<c);}
        sig[base4 + (s.r4[n]-s.o4[n])]=(unsigned char)b;
    }
    // size-2 -> voxel bytes.
    for (int n=tid;n<4096;n+=1024) if (s.o2[n]) {
        int Ix=n&15,Iy=(n>>4)&15,Iz=(n>>8)&15,b=0;
        for (int c=0;c<8;++c){int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1);
            if ((s.masks_sp[y*32+x]>>z)&1u) b|=(1<<c);}
        sig[base2 + (s.r2[n]-s.o2[n])]=(unsigned char)b;
    }
}

// Parallel level-major decode: reconstruct 1024 z-line masks (L order) into
// out_masks[bid*1024+L].
__global__ void waveletOctreeSigDecodeParKernel(
    const unsigned char* __restrict__ coded,
    const size_t* __restrict__ sig_sizes,
    uint32_t* __restrict__ out_masks)
{
    int bid = blockIdx.x, tid = threadIdx.x;
    const unsigned char* blk = coded + (long)bid * WOCT_SLOT_BYTES;
    const unsigned char* sig = blk + WOCT_HDR_BYTES;

    __shared__ WoctShared s;
    __shared__ int n16,n8,n4;
    for (int L=tid; L<1024; L+=1024) s.masks_sp[L]=0;
    for (int i=tid;i<4096;i+=1024) s.o2[i]=0;
    for (int i=tid;i<512; i+=1024) s.o4[i]=0;
    for (int i=tid;i<64;  i+=1024) s.o8[i]=0;
    for (int i=tid;i<8;   i+=1024) s.o16[i]=0;
    __syncthreads();

    int mode = blk[0];
    if (mode == 0) {                            // flat fallback
        const uint32_t* flat = reinterpret_cast<const uint32_t*>(sig);
        for (int L=tid; L<1024; L+=1024) s.masks_sp[woct_L_to_spatial(L)]=flat[L];
        __syncthreads();
        for (int L=tid; L<1024; L+=1024) out_masks[(long)bid*1024+L]=s.masks_sp[woct_L_to_spatial(L)];
        return;
    }
    if (sig_sizes[bid]==0) {                    // empty block
        for (int L=tid; L<1024; L+=1024) out_masks[(long)bid*1024+L]=0;
        return;
    }

    // root byte -> o16 present set.
    if (tid==0){ int b=sig[0]; for (int c=0;c<8;++c) if (b&(1<<c)) s.o16[c]=1; }
    __syncthreads();

    // level-16: scan o16, expand present nodes' bytes -> o8.
    for (int i=tid;i<8;i+=1024) s.r16[i]=s.o16[i];
    __syncthreads();
    woct_incl_scan(s.r16,8);
    if (tid==0) n16=s.r16[7];
    __syncthreads();
    for (int n=tid;n<8;n+=1024) if (s.o16[n]) {
        int Ix=n&1,Iy=(n>>1)&1,Iz=(n>>2)&1;
        int b=sig[1 + (s.r16[n]-s.o16[n])];
        for (int c=0;c<8;++c) if (b&(1<<c)){
            int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1); s.o8[z*16+y*4+x]=1; }
    }
    __syncthreads();
    // level-8: scan o8, expand -> o4.
    for (int i=tid;i<64;i+=1024) s.r8[i]=s.o8[i];
    __syncthreads();
    woct_incl_scan(s.r8,64);
    if (tid==0) n8=s.r8[63];
    __syncthreads();
    int base8=1+n16;
    for (int n=tid;n<64;n+=1024) if (s.o8[n]) {
        int Ix=n&3,Iy=(n>>2)&3,Iz=(n>>4)&3;
        int b=sig[base8 + (s.r8[n]-s.o8[n])];
        for (int c=0;c<8;++c) if (b&(1<<c)){
            int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1); s.o4[z*64+y*8+x]=1; }
    }
    __syncthreads();
    // level-4: scan o4, expand -> o2.
    for (int i=tid;i<512;i+=1024) s.r4[i]=s.o4[i];
    __syncthreads();
    woct_incl_scan(s.r4,512);
    if (tid==0) n4=s.r4[511];
    __syncthreads();
    int base4=1+n16+n8;
    for (int n=tid;n<512;n+=1024) if (s.o4[n]) {
        int Ix=n&7,Iy=(n>>3)&7,Iz=(n>>6)&7;
        int b=sig[base4 + (s.r4[n]-s.o4[n])];
        for (int c=0;c<8;++c) if (b&(1<<c)){
            int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1); s.o2[z*256+y*16+x]=1; }
    }
    __syncthreads();
    // level-2: scan o2, expand -> voxels (atomicOr: nodes share column words).
    for (int i=tid;i<4096;i+=1024) s.r2[i]=s.o2[i];
    __syncthreads();
    woct_incl_scan(s.r2,4096);
    __syncthreads();
    int base2=1+n16+n8+n4;
    for (int n=tid;n<4096;n+=1024) if (s.o2[n]) {
        int Ix=n&15,Iy=(n>>4)&15,Iz=(n>>8)&15;
        int b=sig[base2 + (s.r2[n]-s.o2[n])];
        for (int c=0;c<8;++c) if (b&(1<<c)){
            int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1);
            atomicOr((unsigned int*)&s.masks_sp[y*32+x], 1u<<z); }
    }
    __syncthreads();
    for (int L=tid; L<1024; L+=1024) out_masks[(long)bid*1024+L]=s.masks_sp[woct_L_to_spatial(L)];
}

// ===========================================================================
// FUSED kernel-2 (1024 threads/block): octree significance + per-line width
// table + packed values, in one pass over the kernel-1 output.
//
//   [4B mode][significance][pad->4][2b/ne width table][per-line values]
//
// The significance region is byte-identical to waveletOctreeSigEncodeParKernel;
// the width table + value payload are byte-identical to the two-level coder
// (same nonempty-line order, per-line widths and packing).  block_sizes2[bid]
// records the exact coded length.  Values are streamed straight to global
// (byte stores); the significance and value regions are disjoint by construction
// (values start at the 4-aligned end of the significance region).
// ===========================================================================
__launch_bounds__(1024)
__global__ void waveletOctreeCodeParKernel(
    const unsigned char* __restrict__ scratch1,
    unsigned char* __restrict__ out,
    size_t* __restrict__ block_sizes2,
    size_t* __restrict__ sig_sizes)
{
    using namespace wbmp_opt;
    int bid = blockIdx.x, tid = threadIdx.x;
    const unsigned char* blk_in = scratch1 + (long)bid * WBMP_SLOT_BYTES;
    const uint32_t* bmp = reinterpret_cast<const uint32_t*>(blk_in);
    const int32_t*  val = reinterpret_cast<const int32_t*>(blk_in + WBMP_BITMAP_BYTES);
    unsigned char* blk_out = out + (long)bid * WOCT_CODE_SLOT_BYTES;
    unsigned char* sig = blk_out + WOCT_HDR_BYTES;

    __shared__ WoctShared s;
    __shared__ int warp_part[2 * 16];

    uint32_t m = bmp[tid];                      // L-order word for this z-line
    s.masks_sp[woct_L_to_spatial(tid)] = m;     // spatial order for the octree
    __syncthreads();

    // ---- octree significance pyramid + per-level wave64 DPP scans ----
    woct_build_pyramid(s);
    int rootocc = 0;
    #pragma unroll
    for (int i=0;i<8;++i) rootocc |= s.o16[i];
    int n16 = woct_incl_scan_fast(s.o16, s.r16, 8,    warp_part);
    int n8  = woct_incl_scan_fast(s.o8,  s.r8,  64,   warp_part);
    int n4  = woct_incl_scan_fast(s.o4,  s.r4,  512,  warp_part);
    int n2  = woct_incl_scan_fast(s.o2,  s.r2,  4096, warp_part);
    int oct_total = 1 + n16 + n8 + n4 + n2;
    int mode = (rootocc && oct_total < WOCT_FLAT_SIG_BYTES) ? 2 : (rootocc ? 0 : 2);
    int sig_bytes = (mode==2) ? (rootocc ? oct_total : 0) : WOCT_FLAT_SIG_BYTES;
    if (tid==0){ blk_out[0] = (unsigned char)mode; sig_sizes[bid] = (size_t)sig_bytes; }

    if (mode==2 && rootocc) {                   // octree significance
        int base16=1, base8=1+n16, base4=1+n16+n8, base2=1+n16+n8+n4;
        if (tid==0){ int rb=0; for(int c=0;c<8;++c) if(s.o16[c]) rb|=(1<<c); sig[0]=(unsigned char)rb; }
        for (int n=tid;n<8;n+=1024) if (s.o16[n]) {
            int Ix=n&1,Iy=(n>>1)&1,Iz=(n>>2)&1,b=0;
            for (int c=0;c<8;++c){int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1);
                if (s.o8[z*16+y*4+x]) b|=(1<<c);}
            sig[base16 + (s.r16[n]-s.o16[n])]=(unsigned char)b;
        }
        for (int n=tid;n<64;n+=1024) if (s.o8[n]) {
            int Ix=n&3,Iy=(n>>2)&3,Iz=(n>>4)&3,b=0;
            for (int c=0;c<8;++c){int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1);
                if (s.o4[z*64+y*8+x]) b|=(1<<c);}
            sig[base8 + (s.r8[n]-s.o8[n])]=(unsigned char)b;
        }
        for (int n=tid;n<512;n+=1024) if (s.o4[n]) {
            int Ix=n&7,Iy=(n>>3)&7,Iz=(n>>6)&7,b=0;
            for (int c=0;c<8;++c){int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1);
                if (s.o2[z*256+y*16+x]) b|=(1<<c);}
            sig[base4 + (s.r4[n]-s.o4[n])]=(unsigned char)b;
        }
        for (int n=tid;n<4096;n+=1024) if (s.o2[n]) {
            int Ix=n&15,Iy=(n>>4)&15,Iz=(n>>8)&15,b=0;
            for (int c=0;c<8;++c){int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1);
                if ((s.masks_sp[y*32+x]>>z)&1u) b|=(1<<c);}
            sig[base2 + (s.r2[n]-s.o2[n])]=(unsigned char)b;
        }
    } else if (mode==0) {                       // flat fallback masks (L order)
        uint32_t* flat = reinterpret_cast<uint32_t*>(sig);
        for (int L=tid; L<1024; L+=1024) flat[L]=bmp[L];
    }
    __syncthreads();

    long wtab_base = ((long)WOCT_HDR_BYTES + sig_bytes + 3) & ~3L;

    // ---- per-line width table + packed values (two-level layout) ----
    int nz = __popc(m), occb = m ? 1 : 0;
    int in_off, occ_rank, tot_nz, tot_ne;
    block_exscan2(nz, occb, in_off, occ_rank, tot_nz, tot_ne, warp_part);
    (void)tot_nz;
    int mx = 0;
    for (int k=0;k<nz;++k){ int a=val[in_off+k]; a=a<0?-a:a; mx|=a; }
    int W = nz ? wbmp_width_bytes(mx) : 1;
    int val_off, tot_val;
    val_off = block_exscan(nz * W, tot_val, warp_part);

    long vals_base = wtab_base + (2L * tot_ne + 7) / 8;
    uint32_t* wtab = reinterpret_cast<uint32_t*>(blk_out + wtab_base);
    int wtab_words = (2 * tot_ne + 31) / 32;
    for (int i=tid;i<wtab_words;i+=1024) wtab[i]=0;
    __syncthreads();

    if (occb) {
        atomicOr(&wtab[occ_rank >> 4], (uint32_t)(W - 1) << ((occ_rank & 15) * 2));
        long p = vals_base + val_off;
        for (int k=0;k<nz;++k){
            unsigned uv = (unsigned)val[in_off + k];
            long q = p + (long)k * W;
            #pragma unroll
            for (int b=0;b<4;++b) if (b<W) blk_out[q + b] = (unsigned char)(uv >> (8*b));
        }
    }
    if (tid==0) block_sizes2[bid] = (size_t)(vals_base + tot_val);
}

inline hipError_t hipWaveletOctreeCode(
    const unsigned char* scratch1, unsigned char* out, size_t* block_sizes2,
    size_t* sig_sizes, int nblocks, hipStream_t stream = 0)
{
    waveletOctreeCodeParKernel<<<nblocks, dim3(WOCT_PAR_THREADS), 0, stream>>>(scratch1, out, block_sizes2, sig_sizes);
    return hipGetLastError();
}

// Round each per-block coded length up to `align` bytes.  In the compacted
// stream the blocks are packed back-to-back, so this keeps every block start
// (and thus its uint32 width-table / flat-mask reads) aligned.  Overhead is
// <align bytes/block; the padding bytes are never read by the decoder.
__global__ void woctAlignSizesKernel(size_t* __restrict__ sizes, int nb, int align)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nb) sizes[i] = (sizes[i] + (align - 1)) & ~(size_t)(align - 1);
}

inline hipError_t waveletOctreeCodeParAlignSizes(size_t* sizes, int nb, hipStream_t stream = 0)
{
    int threads = 256, blocks = (nb + threads - 1) / threads;
    woctAlignSizesKernel<<<blocks, threads, 0, stream>>>(sizes, nb, 4);
    return hipGetLastError();
}

// Compaction: copies the variable-length coded blocks from the fixed-stride
// (WOCT_CODE_SLOT_BYTES) intermediate into a tightly packed payload using the
// exclusive-scan offsets, and writes the self-contained octree header (block
// offsets + per-block significance sizes + mulfac).  dst points to the payload
// region (after the header).  Mirrors wrleCompactKernel; the extra sig-size
// table is what lets the decoder recover each block's significance length.
__global__ void woctCompactKernel(
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
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    size_t size = block_sizes[bid];
    size_t dst_off = offsets[bid];
    size_t src_off = (size_t)bid * WOCT_CODE_SLOT_BYTES;

    if (hdr != nullptr && tid == 0) {
        ((size_t*)(hdr + 8))[bid] = offsets[bid];
        ((uint32_t*)(hdr + 8 + 8L * num_blocks))[bid] = (uint32_t)sig_sizes[bid];
        if (bid == 0) {
            ((int*)hdr)[0] = num_blocks;
            ((int*)hdr)[1] = num_mulfacs;
            float* mf_dst = (float*)(hdr + 8 + 12L * num_blocks);
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

// ===========================================================================
// FULL DECODE, stage A (1024 threads): fused octree coded stream -> kernel-1
// scratch layout [4096B bitmap][packed int32 values].  Exact inverse of the
// fused encoder's value packing, so the output is byte-identical to the
// original kernel-1 output.
// ===========================================================================
// Shared decode body: reconstruct the kernel-1 scratch layout
// [4096B bitmap][packed int32 values] for one block from its coded bytes.
//   blk       — pointer to the block's coded bytes ([4B mode][sig][pad][wtab][values])
//   sig_bytes — significance-region length (octree stream length, or 4096 flat)
//   out       — this block's WBMP_SLOT_BYTES scratch slot
//   bsz_this  — optional: receives WBMP_BITMAP_BYTES + nnz*4 (may be null)
// Addressing is caller-supplied so the same body serves both the fixed-stride
// prototype kernel and the header-addressed (compacted stream) API kernel.
__device__ __forceinline__ void woct_decode_block_to_scratch(
    const unsigned char* __restrict__ blk, int sig_bytes,
    unsigned char* __restrict__ out, size_t* __restrict__ bsz_this)
{
    using namespace wbmp_opt;
    int tid = threadIdx.x;
    const unsigned char* sig = blk + WOCT_HDR_BYTES;
    uint32_t* bmp_out = reinterpret_cast<uint32_t*>(out);
    int32_t*  val_out = reinterpret_cast<int32_t*>(out + WBMP_BITMAP_BYTES);

    __shared__ WoctShared s;
    __shared__ int warp_part[2 * 16];

    s.masks_sp[tid] = 0;
    for (int i=tid;i<4096;i+=1024) s.o2[i]=0;
    for (int i=tid;i<512; i+=1024) s.o4[i]=0;
    for (int i=tid;i<64;  i+=1024) s.o8[i]=0;
    for (int i=tid;i<8;   i+=1024) s.o16[i]=0;
    __syncthreads();

    int mode = blk[0];

    if (mode == 0) {                            // flat masks (L order)
        const uint32_t* flat = reinterpret_cast<const uint32_t*>(sig);
        s.masks_sp[woct_L_to_spatial(tid)] = flat[tid];
        __syncthreads();
    } else if (sig_bytes > 0) {                 // octree decode -> masks_sp
        if (tid==0){ int b=sig[0]; for (int c=0;c<8;++c) if (b&(1<<c)) s.o16[c]=1; }
        __syncthreads();
        int n16 = woct_incl_scan_fast(s.o16, s.r16, 8, warp_part);
        for (int n=tid;n<8;n+=1024) if (s.o16[n]) {
            int Ix=n&1,Iy=(n>>1)&1,Iz=(n>>2)&1; int b=sig[1 + (s.r16[n]-s.o16[n])];
            for (int c=0;c<8;++c) if (b&(1<<c)){
                int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1); s.o8[z*16+y*4+x]=1; }
        }
        __syncthreads();
        int n8 = woct_incl_scan_fast(s.o8, s.r8, 64, warp_part);
        int base8 = 1 + n16;
        for (int n=tid;n<64;n+=1024) if (s.o8[n]) {
            int Ix=n&3,Iy=(n>>2)&3,Iz=(n>>4)&3; int b=sig[base8 + (s.r8[n]-s.o8[n])];
            for (int c=0;c<8;++c) if (b&(1<<c)){
                int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1); s.o4[z*64+y*8+x]=1; }
        }
        __syncthreads();
        int n4 = woct_incl_scan_fast(s.o4, s.r4, 512, warp_part);
        int base4 = 1 + n16 + n8;
        for (int n=tid;n<512;n+=1024) if (s.o4[n]) {
            int Ix=n&7,Iy=(n>>3)&7,Iz=(n>>6)&7; int b=sig[base4 + (s.r4[n]-s.o4[n])];
            for (int c=0;c<8;++c) if (b&(1<<c)){
                int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1); s.o2[z*256+y*16+x]=1; }
        }
        __syncthreads();
        woct_incl_scan_fast(s.o2, s.r2, 4096, warp_part);
        int base2 = 1 + n16 + n8 + n4;
        for (int n=tid;n<4096;n+=1024) if (s.o2[n]) {
            int Ix=n&15,Iy=(n>>4)&15,Iz=(n>>8)&15; int b=sig[base2 + (s.r2[n]-s.o2[n])];
            for (int c=0;c<8;++c) if (b&(1<<c)){
                int x=2*Ix+(c&1),y=2*Iy+((c>>1)&1),z=2*Iz+((c>>2)&1);
                atomicOr((unsigned int*)&s.masks_sp[y*32+x], 1u<<z); }
        }
        __syncthreads();
    }
    // else: empty (mode 2, sig 0) -> masks stay zero.

    uint32_t maskL = s.masks_sp[woct_L_to_spatial(tid)];
    bmp_out[tid] = maskL;                        // bitmap in L order

    // ---- value unpack (reverse of the fused encoder) ----
    long wtab_base = ((long)WOCT_HDR_BYTES + sig_bytes + 3) & ~3L;
    int nz = __popc(maskL), occb = maskL ? 1 : 0;
    int in_off, occ_rank, tot_nz, tot_ne;
    block_exscan2(nz, occb, in_off, occ_rank, tot_nz, tot_ne, warp_part);
    __syncthreads();                            // protect warp_part reuse
    const uint32_t* wtab = reinterpret_cast<const uint32_t*>(blk + wtab_base);
    int W = occb ? (int)((wtab[occ_rank >> 4] >> ((occ_rank & 15) * 2)) & 3u) + 1 : 1;
    int val_off, tot_val;
    val_off = block_exscan(nz * W, tot_val, warp_part);
    (void)tot_val;
    long vals_base = wtab_base + (2L * tot_ne + 7) / 8;
    if (occb) {
        const unsigned char* vp = blk + vals_base + val_off;
        int sh = 32 - 8*W;
        for (int k=0;k<nz;++k){
            unsigned uv = 0;
            #pragma unroll
            for (int b=0;b<4;++b) if (b<W) uv |= (unsigned)vp[(long)k*W + b] << (8*b);
            val_out[in_off + k] = (int)(uv << sh) >> sh;   // sign-extend from W bytes
        }
    }
    if (tid==0 && bsz_this) *bsz_this = (size_t)WBMP_BITMAP_BYTES + (size_t)tot_nz * 4;
}

// Prototype kernel: fixed-stride coded slots + per-block sig-size array.
__launch_bounds__(1024)
__global__ void waveletOctreeDecodeToBitmapKernel(
    const unsigned char* __restrict__ coded,
    const size_t* __restrict__ sig_sizes,
    unsigned char* __restrict__ scratch1_out,
    size_t* __restrict__ block_sizes)
{
    int bid = blockIdx.x;
    woct_decode_block_to_scratch(
        coded + (long)bid * WOCT_CODE_SLOT_BYTES, (int)sig_sizes[bid],
        scratch1_out + (long)bid * WBMP_SLOT_BYTES, &block_sizes[bid]);
}

// Octree self-contained stream header:
//   [int nb][int nmf][size_t offsets[nb]][uint32 sig_sizes[nb]][float mulfac[nmf]]
__host__ __device__ inline int hipOctreeHeaderSize(int num_blocks, int num_mulfacs) {
    return 8 + 12 * num_blocks + 4 * num_mulfacs;
}

// API decode kernel: locates each block in the compacted stream via the header
// offset table, reads its significance length from the header sig-size table,
// reconstructs the kernel-1 scratch layout, and (block 0) publishes
// inv_scale = 1/mulfac for stage B.
__launch_bounds__(1024)
__global__ void waveletOctreeDecodeToBitmapHdrKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ scratch1_out,
    float* __restrict__ inv_scale_out)
{
    int bid = blockIdx.x, tid = threadIdx.x;
    const int* hdr = reinterpret_cast<const int*>(input);
    int num_blocks  = hdr[0];
    int num_mulfacs = hdr[1];
    const size_t*   offsets = reinterpret_cast<const size_t*>(input + 8);
    const uint32_t* sig_u32 = reinterpret_cast<const uint32_t*>(input + 8 + 8L * num_blocks);
    const float*    mulfacs = reinterpret_cast<const float*>(input + 8 + 12L * num_blocks);
    const unsigned char* data_base = input + hipOctreeHeaderSize(num_blocks, num_mulfacs);

    if (bid == 0 && tid == 0 && inv_scale_out)
        *inv_scale_out = 1.0f / mulfacs[0];

    woct_decode_block_to_scratch(
        data_base + offsets[bid], (int)sig_u32[bid],
        scratch1_out + (long)bid * WBMP_SLOT_BYTES, nullptr);
}

inline hipError_t hipWaveletOctreeDecodeToBitmap(
    const unsigned char* coded, const size_t* sig_sizes,
    unsigned char* scratch1_out, size_t* block_sizes, int nblocks, hipStream_t stream = 0)
{
    waveletOctreeDecodeToBitmapKernel<<<nblocks, dim3(WOCT_PAR_THREADS), 0, stream>>>(
        coded, sig_sizes, scratch1_out, block_sizes);
    return hipGetLastError();
}

// ===========================================================================
// FULL DECODE, stage B (256 threads): kernel-1 scratch layout
// [4096B bitmap][packed int32] -> dequantize + inverse wavelet ZYX -> wavefield.
// The inverse transform mirrors waveletRLEInverseFusedKernel; only the front-end
// (RLE decode) is swapped for the bitmap+value reconstruction, which reverses
// waveletBitmapFusedKernel's Phase 4.  Requires DS79_INCLUDE_REG32.
// ===========================================================================
__device__ __forceinline__ void woct_bitmap_inverse_body(
    const unsigned char* __restrict__ scratch1,
    float* __restrict__ output,
    float inv_scale, int ldimx, int ldimxy)
{
    constexpr int PLANES=32, BATCH=8, SLC=2, NTHREADS=256;
    using BlockScan = rocprim::block_scan<int, NTHREADS>;
    __shared__ union { float wavelet[BATCH*1024]; typename BlockScan::storage_type scan; } lds;

    int tid=threadIdx.x, xg=tid%8, yr=tid/8;
    int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    const unsigned char* blk = scratch1 + (long)bid*WBMP_SLOT_BYTES;
    const uint32_t* bmp = reinterpret_cast<const uint32_t*>(blk);
    const int32_t*  vals = reinterpret_cast<const int32_t*>(blk + WBMP_BITMAP_BYTES);

    ds79_float4_vec regs[PLANES];

    // ---- Phase 1: reconstruct dequantized coefficients (reverse fwd Phase 4) ----
    int block_val_base = 0;
    for (int x_off=0; x_off<4; ++x_off){
        uint32_t mask = bmp[x_off*256 + tid];
        int nnz = __popc(mask);
        int my_off, pass_total;
        BlockScan().exclusive_scan(nnz, my_off, 0, pass_total, lds.scan);
        __syncthreads();
        int base = block_val_base + my_off;
        #pragma unroll
        for (int z=0; z<32; ++z){
            float f = 0.0f;
            if (mask & (1u<<z)){ int rank=__popc(mask & ((1u<<z)-1)); f = inv_scale * (float)vals[base+rank]; }
            regs[z][x_off] = f;
        }
        block_val_base += pass_total;
        __syncthreads();
    }

    // ---- Phase 2: inverse X+Y in LDS ----
    for (int pb=0; pb<PLANES; pb+=BATCH){
        for (int dp=0; dp<BATCH; dp++){
            ds79_float4_vec v = regs[pb+dp];
            int x0 = xg*4;
            lds.wavelet[dp*1024 + (x0+0)*32 + (yr^(x0+0))] = v[0];
            lds.wavelet[dp*1024 + (x0+1)*32 + (yr^(x0+1))] = v[1];
            lds.wavelet[dp*1024 + (x0+2)*32 + (yr^(x0+2))] = v[2];
            lds.wavelet[dp*1024 + (x0+3)*32 + (yr^(x0+3))] = v[3];
        }
        __syncthreads();
        int pl=tid/32, pos=tid%32; float line[32];
        for (int x=0;x<32;x++) line[x]=lds.wavelet[pl*1024 + x*32 + (pos^x)];
        us79_inverse_reg32(line);
        for (int x=0;x<32;x++) lds.wavelet[pl*1024 + x*32 + (pos^x)]=line[x];
        __syncthreads();
        for (int y=0;y<32;y++) line[y]=lds.wavelet[pl*1024 + pos*32 + (y^pos)];
        us79_inverse_reg32(line);
        for (int y=0;y<32;y++) lds.wavelet[pl*1024 + pos*32 + (y^pos)]=line[y];
        __syncthreads();
        for (int dp=0; dp<BATCH; dp++){
            ds79_float4_vec v; int x0=xg*4;
            v[0]=lds.wavelet[dp*1024 + (x0+0)*32 + (yr^(x0+0))];
            v[1]=lds.wavelet[dp*1024 + (x0+1)*32 + (yr^(x0+1))];
            v[2]=lds.wavelet[dp*1024 + (x0+2)*32 + (yr^(x0+2))];
            v[3]=lds.wavelet[dp*1024 + (x0+3)*32 + (yr^(x0+3))];
            regs[pb+dp]=v;
        }
        __syncthreads();
    }

    // ---- Phase 3: inverse Z in registers ----
    us79_inverse_f4_scalar_tmp(regs, PLANES);

    // ---- Phase 4: store to global ----
    float* block_base = output + (size_t)blockIdx.z*32*ldimxy;
    int gx = blockIdx.x*32 + xg*4;
    int gy = blockIdx.y*32 + yr;
    uint32_t byte_off = (gx + gy*ldimx)*(uint32_t)sizeof(float);
    #pragma unroll
    for (int p=0;p<PLANES;p++){
        auto rsrc = __builtin_amdgcn_make_buffer_rsrc(block_base + (long)p*ldimxy, 0, -1, 0x00027000);
        auto v = __builtin_bit_cast(__attribute__((__vector_size__(16))) int, regs[p]);
        __builtin_amdgcn_raw_buffer_store_b128(v, rsrc, byte_off, 0, SLC);
    }
}

// Stage B with a host-scalar inv_scale (prototype / test path).
__launch_bounds__(256, 2)
__global__ void waveletBitmapInverseFusedKernel(
    const unsigned char* __restrict__ scratch1,
    float* __restrict__ output,
    float inv_scale, int ldimx, int ldimxy)
{
    woct_bitmap_inverse_body(scratch1, output, inv_scale, ldimx, ldimxy);
}

// Stage B with a device inv_scale pointer (API path): reads 1/mulfac published
// by the header-addressed decode kernel, so no host readback is needed.
__launch_bounds__(256, 2)
__global__ void waveletBitmapInverseFusedDevKernel(
    const unsigned char* __restrict__ scratch1,
    float* __restrict__ output,
    const float* __restrict__ inv_scale, int ldimx, int ldimxy)
{
    woct_bitmap_inverse_body(scratch1, output, *inv_scale, ldimx, ldimxy);
}

inline hipError_t hipWaveletBitmapInverseFused(
    const unsigned char* scratch1, float* output, float inv_scale,
    int nx, int ny, int nz, int ldimx, int ldimxy, hipStream_t stream=0)
{
    dim3 grid((nx+31)/32,(ny+31)/32,(nz+31)/32);
    waveletBitmapInverseFusedKernel<<<grid, dim3(256), 0, stream>>>(scratch1, output, inv_scale, ldimx, ldimxy);
    return hipGetLastError();
}

inline hipError_t hipWaveletOctreeSigEncode(
    const unsigned char* scratch1, unsigned char* out, size_t* sig_sizes,
    int nblocks, int threads = 256, hipStream_t stream = 0)
{
    waveletOctreeSigEncodeKernel<<<nblocks, dim3(threads), 0, stream>>>(scratch1, out, sig_sizes);
    return hipGetLastError();
}
inline hipError_t hipWaveletOctreeSigEncodePar(
    const unsigned char* scratch1, unsigned char* out, size_t* sig_sizes,
    int nblocks, hipStream_t stream = 0)
{
    waveletOctreeSigEncodeParKernel<<<nblocks, dim3(WOCT_PAR_THREADS), 0, stream>>>(scratch1, out, sig_sizes);
    return hipGetLastError();
}
inline hipError_t hipWaveletOctreeSigDecodePar(
    const unsigned char* coded, const size_t* sig_sizes, uint32_t* out_masks,
    int nblocks, hipStream_t stream = 0)
{
    waveletOctreeSigDecodeParKernel<<<nblocks, dim3(WOCT_PAR_THREADS), 0, stream>>>(coded, sig_sizes, out_masks);
    return hipGetLastError();
}
inline hipError_t hipWaveletOctreeSigDecode(
    const unsigned char* coded, const size_t* sig_sizes, uint32_t* out_masks,
    int nblocks, int threads = 256, hipStream_t stream = 0)
{
    waveletOctreeSigDecodeKernel<<<nblocks, dim3(threads), 0, stream>>>(coded, sig_sizes, out_masks);
    return hipGetLastError();
}

#endif // HIPWAVELET_OCTREE_H
