// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// Octree significance coder for the 3D compression path. It replaces the flat
// 4096 B/block significance bitmap with a depth-5 octree of the 32^3
// significance set. A per-block mode tag falls back to the flat bitmap when
// that representation is smaller.
//
// Coded block layout:  [1B mode][significance][2b/nonempty-line width table]
//                      [per-line packed values]
//   mode 0 (flat)   : significance = 1024 uint32 masks in L order  (4096 B)
//   mode 2 (octree) : significance = variable-length octree stream
// The width table and value payload use per-line widths and packed values.

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

// Fused kernel-2 coded slot: [4B mode][significance][<=3B pad to 4-align][per-block
// PFOR value payload].  PFOR region = [Wbyte+3 pad][exception mask: <=(32768/32)*4
// =4096 B][base + patch: <= nnz*W_hi <= WBMP_MAX_VAL_BYTES].
//   sig = octree stream (<=WOCT_MAX_SIG_BYTES) or flat masks (4096 B)
static constexpr long WOCT_CODE_SLOT_RAW =
    WOCT_HDR_BYTES + WOCT_MAX_SIG_BYTES + 3 + 4 + ((32768 + 31) / 32) * 4 + WBMP_MAX_VAL_BYTES;
static constexpr long WOCT_CODE_SLOT_BYTES = (WOCT_CODE_SLOT_RAW + 15) & ~15L;

// Kernel-1 bitmap word order L=x_off*256+tid maps to spatial line (ix,iy):
//   tid = L & 255 ; xg = tid & 7 ; yr = tid >> 3 ; ix = xg*4 + (L>>8) ; iy = yr
__host__ __device__ __forceinline__ int woct_L_to_spatial(int L) {
    int x_off = L >> 8, tid = L & 255, xg = tid & 7, yr = tid >> 3;
    return yr * 32 + (xg * 4 + x_off);   // spatial index iy*32+ix
}

// The production encoder and decoder process one z-line per thread.
static constexpr int WOCT_PAR_THREADS = 1024;

// Fast inclusive scan of occ[0..N)->r[0..N) (lidx order), N in {8,64,512,4096}.
// Uses a blocked layout (thread t owns the contiguous chunk [t*C, t*C+C), C=N/1024
// rounded up) so a single wave64 DPP block scan (wbmp_opt::block_exscan) replaces
// the ~log2(N) Hillis-Steele passes.  r[n] holds the inclusive prefix at n exactly
// as the occupancy input, so downstream uses rank = r[n]-occ[n].
// Returns the total occupancy. warp_part is 2*16 ints.
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

// ===========================================================================
// FUSED kernel-2 (1024 threads/block): octree significance + per-line width
// table + packed values, in one pass over the kernel-1 output.
//
//   [4B mode][significance][pad->4][2b/ne width table][per-line values]
//
// The value payload uses nonempty-line order and per-line widths. block_sizes2[bid]
// records the exact coded length.  Values are streamed straight to global
// (byte stores); the significance and value regions are disjoint by construction
// (values start at the 4-aligned end of the significance region).
// ===========================================================================
__launch_bounds__(1024)
__global__ void hipcvx_waveletOctreeCodeParKernel(
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
    // Header is WOCT_HDR_BYTES=4 but carries only the mode byte.  Store it as a
    // u32 so bytes 1..3 are written zero rather than left as whatever the slot
    // buffer last held -- compaction copies them into the output stream.
    if (tid==0){ *reinterpret_cast<uint32_t*>(blk_out) = (uint32_t)mode;
                 sig_sizes[bid] = (size_t)sig_bytes; }

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

    // ---- per-block PFOR value coder ----
    // Layout from val_base: [Wbyte=(Whi<<4)|Wlo, +3 pad][mask u32 words (only if
    // Wlo<Whi)][base: Wlo low bytes per nonzero][patch: (Whi-Wlo) high bytes per
    // exception].  All streams fixed-width/byte-aligned -> coalesced GPU decode.
    long val_base = ((long)WOCT_HDR_BYTES + sig_bytes + 3) & ~3L;
    // <=3 B of alignment slack between the significance region and val_base.
    // Never read on decode, but it is inside block_sizes2 and lands in the
    // stream, so it has to be written to keep the output reproducible.
    if (tid==0) for (long p=(long)WOCT_HDR_BYTES+sig_bytes; p<val_base; ++p) blk_out[p]=0;
    int nz = __popc(m);
    int in_off, occ_rank_u, tot_nz, tot_ne_u;
    block_exscan2(nz, m ? 1 : 0, in_off, occ_rank_u, tot_nz, tot_ne_u, warp_part);
    (void)occ_rank_u; (void)tot_ne_u;

    // per-line width histogram (local counts, then block reduce)
    int lc1=0,lc2=0,lc3=0,lc4=0;
    for (int k=0;k<nz;++k){ unsigned a=hipcvx_abs_i32(val[in_off+k]); int w=hipcvx_width_bytes(a);
        if(w<=1)++lc1; else if(w==2)++lc2; else if(w==3)++lc3; else ++lc4; }
    __shared__ int sh_h[5];
    __shared__ int sh_wlo;
    if (tid<5) sh_h[tid]=0;
    __syncthreads();
    if(lc2)atomicAdd(&sh_h[2],lc2);
    if(lc3)atomicAdd(&sh_h[3],lc3);
    if(lc4)atomicAdd(&sh_h[4],lc4);
    __syncthreads();
    int Whi = sh_h[4]?4 : sh_h[3]?3 : sh_h[2]?2 : 1;
    int nnz = tot_nz;
    if (tid==0){
        long best=-1; int Wlo=Whi; long maskbytes=(long)((nnz+31)/32)*4;
        for (int wl=1; wl<=Whi; ++wl){
            long nexc=0; for (int w=wl+1;w<=4;++w) nexc+=sh_h[w];
            long cost=(long)nnz*wl + nexc*(Whi-wl) + (wl<Whi?maskbytes:0);
            if (best<0||cost<best){ best=cost; Wlo=wl; }
        }
        sh_wlo = Wlo;
    }
    __syncthreads();
    int Wlo = sh_wlo;
    int have_mask = (Wlo < Whi);
    long mask_base  = val_base + 4;                  // Wbyte(1)+pad(3): keep mask 4-aligned
    int  mask_words = have_mask ? (nnz + 31) / 32 : 0;
    long base_base  = mask_base + (long)mask_words * 4;
    long patch_base = base_base + (long)nnz * Wlo;

    int e_line = 0;                                  // local exceptions (width > Wlo)
    if (2 > Wlo) e_line += lc2;
    if (3 > Wlo) e_line += lc3;
    if (4 > Wlo) e_line += lc4;
    __syncthreads();                                 // protect warp_part reuse
    int ex_base, tot_ex;
    ex_base = block_exscan(e_line, tot_ex, warp_part);

    uint32_t* mp = reinterpret_cast<uint32_t*>(blk_out + mask_base);
    for (int i=tid;i<mask_words;i+=1024) mp[i]=0;
    __syncthreads();
    // Wbyte occupies 1 of the 4 B reserved before mask_base; store it as a u32
    // (val_base is 4-aligned by construction) so the 3 pad bytes are zero.
    if (tid==0) *reinterpret_cast<uint32_t*>(blk_out + val_base) =
                    (uint32_t)((Whi << 4) | Wlo);

    if (nz) {
        int local_ex = 0;
        for (int k=0;k<nz;++k){
            int v = val[in_off+k]; unsigned uv = (unsigned)v;
            long g = (long)in_off + k;
            long q = base_base + g * Wlo;
            #pragma unroll
            for (int b=0;b<4;++b) if (b<Wlo) blk_out[q + b] = (unsigned char)(uv >> (8*b));
            unsigned a = hipcvx_abs_i32(v);
            if (hipcvx_width_bytes(a) > Wlo){
                atomicOr(&mp[g>>5], 1u << (g & 31));
                long pq = patch_base + (long)(ex_base + local_ex) * (Whi - Wlo);
                for (int b=0;b<Whi-Wlo;++b) blk_out[pq + b] = (unsigned char)(uv >> (8*(Wlo+b)));
                ++local_ex;
            }
        }
    }
    // woctAlignSizes rounds this length up to 4 afterwards; zero the <=3 B that
    // rounding exposes, since compaction will copy them into the stream.
    if (tid==0){
        long end = patch_base + (long)tot_ex * (Whi - Wlo);
        for (long p=end; p<((end+3)&~3L); ++p) blk_out[p]=0;
        block_sizes2[bid] = (size_t)end;
    }
}

// Round each per-block coded length up to `align` bytes.  In the compacted
// stream the blocks are packed back-to-back, so this keeps every block start
// (and thus its uint32 width-table / flat-mask reads) aligned.  Overhead is
// <align bytes/block; the padding bytes are never read by the decoder.
__global__ void hipcvx_woctAlignSizesKernel(size_t* __restrict__ sizes, int nb, int align)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nb) sizes[i] = (sizes[i] + (align - 1)) & ~(size_t)(align - 1);
}

inline hipError_t waveletOctreeCodeParAlignSizes(size_t* sizes, int nb, hipStream_t stream = 0)
{
    int threads = 256, blocks = (nb + threads - 1) / threads;
    hipcvx_woctAlignSizesKernel<<<blocks, threads, 0, stream>>>(sizes, nb, 4);
    return hipGetLastError();
}

// Compaction: copies the variable-length coded blocks from the fixed-stride
// (WOCT_CODE_SLOT_BYTES) intermediate into a tightly packed payload using the
// exclusive-scan offsets, and writes the self-contained octree header (block
// offsets + per-block significance sizes + mulfac). dst points to the payload
// region after the header. The significance-size table lets the decoder recover
// each block's significance length.
__global__ void hipcvx_woctCompactKernel(
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

    if (bid == num_blocks - 1 && tid == 0) {
        size_t total = 8 + 12L * num_blocks + 4L * num_mulfacs
                     + dst_off + size;
        size_t padded = (total + 7) & ~(size_t)7;
        for (size_t i = 0; i < padded - total; ++i)
            dst[dst_off + size + i] = 0;
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

    // ---- per-block PFOR value unpack (reverse of the fused encoder) ----
    long val_base = ((long)WOCT_HDR_BYTES + sig_bytes + 3) & ~3L;
    int nz = __popc(maskL);
    int in_off, occ_rank_u, tot_nz, tot_ne_u;
    block_exscan2(nz, maskL ? 1 : 0, in_off, occ_rank_u, tot_nz, tot_ne_u, warp_part);
    (void)occ_rank_u; (void)tot_ne_u;
    __syncthreads();                            // protect warp_part reuse
    int nnz = tot_nz;
    if (nnz > 0) {
        int Wlo = blk[val_base] & 0xF, Whi = (blk[val_base] >> 4) & 0xF;
        int have_mask = (Wlo < Whi);
        long mask_base  = val_base + 4;
        int  mask_words = have_mask ? (nnz + 31) / 32 : 0;
        long base_base  = mask_base + (long)mask_words * 4;
        long patch_base = base_base + (long)nnz * Wlo;
        const uint32_t* mp = reinterpret_cast<const uint32_t*>(blk + mask_base);

        int e_line = 0;                         // local exceptions in this line's rank range
        if (have_mask) for (int k=0;k<nz;++k){ long g=(long)in_off+k; if((mp[g>>5]>>(g&31))&1u) ++e_line; }
        int ex_base, tot_ex;
        ex_base = block_exscan(e_line, tot_ex, warp_part);
        (void)tot_ex;

        int shlo = 32 - 8*Wlo, shhi = 32 - 8*Whi;
        if (nz) {
            int local_ex = 0;
            for (int k=0;k<nz;++k){
                long g = (long)in_off + k; unsigned uv = 0;
                #pragma unroll
                for (int b=0;b<4;++b) if (b<Wlo) uv |= (unsigned)blk[base_base + g*Wlo + b] << (8*b);
                int exc = have_mask && ((mp[g>>5]>>(g&31))&1u);
                if (exc){
                    long pq = patch_base + (long)(ex_base + local_ex) * (Whi - Wlo);
                    for (int b=0;b<Whi-Wlo;++b) uv |= (unsigned)blk[pq + b] << (8*(Wlo+b));
                    val_out[in_off + k] = (int)(uv << shhi) >> shhi;
                    ++local_ex;
                } else {
                    val_out[in_off + k] = (int)(uv << shlo) >> shlo;
                }
            }
        }
    }
    if (tid==0 && bsz_this) *bsz_this = (size_t)WBMP_BITMAP_BYTES + (size_t)tot_nz * 4;
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
__global__ void hipcvx_waveletOctreeDecodeToBitmapHdrKernel(
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

// ===========================================================================
// FULL DECODE, stage B (256 threads): kernel-1 scratch layout
// [4096B bitmap][packed int32] -> dequantize + inverse wavelet ZYX -> wavefield.
// The bitmap and value reconstruction reverses
// hipcvx_waveletBitmapFusedKernel's Phase 4.
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
    size_t byte_off = ((size_t)gy*ldimx + gx)*sizeof(float);
    #pragma unroll
    for (int p=0;p<PLANES;p++)
        hipPlaneStoreNT<ds79_float4_vec>(block_base + (size_t)p*ldimxy, byte_off, regs[p]);
}

// Stage B with a device inv_scale pointer (API path): reads 1/mulfac published
// by the header-addressed decode kernel, so no host readback is needed.
__launch_bounds__(256, 2)
__global__ void hipcvx_waveletBitmapInverseFusedDevKernel(
    const unsigned char* __restrict__ scratch1,
    float* __restrict__ output,
    const float* __restrict__ inv_scale, int ldimx, int ldimxy)
{
    woct_bitmap_inverse_body(scratch1, output, *inv_scale, ldimx, ldimxy);
}

#endif // HIPWAVELET_OCTREE_H
