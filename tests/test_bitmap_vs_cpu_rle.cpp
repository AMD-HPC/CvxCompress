// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// CPU-only comparison of the production full-block RLE (Run_Length_Encode_Slow,
// with 8x packed escape codes) against the bitmap-significance-split, on
// identical CPU-wavelet-transformed, identically-quantized 32^3 blocks.
//
// Emits a four-bucket byte breakdown for both coders:
//   RLE     : zero-run structure | escape-tag overhead | value payload
//   bitmap  : significance (4096B) | width header       | value payload (+padding)
//
// The instrumented RLE clone reproduces the exact control flow of
// Run_Length_Encode_Slow; its total is asserted against the real encoder.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define MY_AVX_DEFINED
#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/x86/avx512.h"

#include "Block_Copy.hxx"
#include "Wavelet_Transform_Fast.hxx"
#include "Run_Length_Encode_Slow.hxx"
#include "Run_Length_Escape_Codes.hxx"

// ---------------------------------------------------------------------------
// Instrumented clone of Run_Length_Encode_Slow (AVX / TMJ_AVX_RLE path).
// Adds byte-category counters at every write site.  Byte format identical.
// ---------------------------------------------------------------------------
struct RleBuckets { long zero = 0, tag = 0, payload = 0; };

static inline void ib_rle(int& rle, char* dst, int& bp, RleBuckets& b) {
    if (rle > 0) {
        if (rle == 1) { dst[bp++] = (char)0; b.zero += 1; }
        else if (rle < 256) {
            int v = (RLESC1 & 0xFF) | ((rle & 0xFF) << 8);
            *((short*)(dst + bp)) = (short)v; bp += 2; b.zero += 2;
        } else {
            int v = (RLESC3 & 0xFF) | ((rle & 0xFFFFFF) << 8);
            *((int*)(dst + bp)) = v; bp += 4; b.zero += 4;
        }
        rle = 0;
    }
}

static inline void ib_word(int i, int zeros, int* esc, int* pay, int* nb,
                           int& rle, char* dst, int& bp, RleBuckets& b) {
    if (zeros & (1 << i)) { ++rle; return; }
    ib_rle(rle, dst, bp, b);
    long rval = (long)esc[i] | ((long)pay[i] << 8);
    *((long*)(dst + bp)) = rval;
    int n = nb[i]; bp += n;
    if (n == 1) b.payload += 1;                 // byte value carries no tag
    else { b.tag += 1; b.payload += (n - 1); }  // esc + (n-1) payload bytes
}

static inline int ib_count_true(__m256 predicate) {
    __m128i sum = _mm_hadd_epi32(_mm256_castsi256_si128(_mm256_castps_si256(predicate)),
                                 _mm256_extractf128_si256(_mm256_castps_si256(predicate), 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return -_mm_extract_epi32(sum, 0);
}

// Returns total bytes; fills buckets.  Mirrors Run_Length_Encode_Slow exactly.
static int rle_breakdown(float scale, float* vals, int num, char* dst, RleBuckets& b) {
    int rle = 0, bp = 0;
    __m256 _mm_scale = _mm256_set1_ps(scale);
    __m256 _mm_byte_lo  = _mm256_cvtepi32_ps(_mm256_set1_epi32(VLESC2));
    __m256 _mm_byte_hi  = _mm256_cvtepi32_ps(_mm256_set1_epi32(RLESC3));
    __m256 _mm_short_lo = _mm256_cvtepi32_ps(_mm256_set1_epi32(-32768));
    __m256 _mm_short_hi = _mm256_cvtepi32_ps(_mm256_set1_epi32(32767));
    __m256 _mm_i3_lo    = _mm256_cvtepi32_ps(_mm256_set1_epi32(-8388608));
    __m256 _mm_i3_hi    = _mm256_cvtepi32_ps(_mm256_set1_epi32(8388607));
    for (int i = 0; i < num; i += 8) {
        __m256 fvals = _mm256_mul_ps(_mm_scale, _mm256_load_ps(vals + i));
        __m256i ivals = _mm256_cvttps_epi32(fvals);
        __m256 fivals = _mm256_cvtepi32_ps(ivals);
        __m256 is_zero = _mm256_cmp_ps(fivals, _mm256_setzero_ps(), 0);
        int zeros = _mm256_movemask_ps(is_zero);
        if (zeros == 255) { rle += 8; continue; }

        __m256 is_byte = _mm256_and_ps(_mm256_cmp_ps(fivals, _mm_byte_lo, 30),
                                       _mm256_cmp_ps(fivals, _mm_byte_hi, 17));
        if (zeros == 0 && _mm256_movemask_ps(is_byte) == 255) {
            ib_rle(rle, dst, bp, b);
            bp += 8; b.payload += 8;             // 8 raw bytes, no tag
            continue;
        }
        int num_bytes = ib_count_true(is_byte);
        __m256 is_short = _mm256_and_ps(_mm256_cmp_ps(fivals, _mm_short_lo, 29),
                                        _mm256_cmp_ps(fivals, _mm_short_hi, 18));
        if (zeros == 0 && _mm256_movemask_ps(is_short) == 255 &&
            (num_bytes + (8 - num_bytes) * 3) > 17) {
            ib_rle(rle, dst, bp, b);
            bp += 17; b.tag += 1; b.payload += 16;   // VLESC2_8x
            continue;
        }
        int num_shorts = ib_count_true(is_short);
        __m256 is_i3 = _mm256_and_ps(_mm256_cmp_ps(fivals, _mm_i3_lo, 29),
                                     _mm256_cmp_ps(fivals, _mm_i3_hi, 18));
        if (zeros == 0 && _mm256_movemask_ps(is_i3) == 255 &&
            (num_bytes + (num_shorts - num_bytes) * 3 + (8 - num_shorts) * 4) > 25) {
            ib_rle(rle, dst, bp, b);
            bp += 25; b.tag += 1; b.payload += 24;   // VLESC3_8x
            continue;
        }
        is_i3 = _mm256_andnot_ps(is_short, is_i3);
        is_short = _mm256_andnot_ps(is_byte, is_short);
        is_byte = _mm256_andnot_ps(is_zero, is_byte);
        __m256 is_not_float = _mm256_or_ps(is_zero, _mm256_or_ps(is_byte, _mm256_or_ps(is_short, is_i3)));

        __m256 esc = _mm256_and_ps(is_byte, _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0xFF)),
                                                          _mm256_castsi256_ps(ivals)));
        esc = _mm256_or_ps(esc, _mm256_and_ps(is_short, _mm256_castsi256_ps(_mm256_set1_epi32(VLESC2 & 0xFF))));
        esc = _mm256_or_ps(esc, _mm256_and_ps(is_i3, _mm256_castsi256_ps(_mm256_set1_epi32(VLESC3 & 0xFF))));
        esc = _mm256_or_ps(esc, _mm256_andnot_ps(is_not_float, _mm256_castsi256_ps(_mm256_set1_epi32(VLESC4 & 0xFF))));
        __m256 payload = _mm256_and_ps(_mm256_or_ps(is_short, is_i3), _mm256_castsi256_ps(ivals));
        payload = _mm256_or_ps(payload, _mm256_andnot_ps(is_not_float, fvals));
        __m256 nbytes = _mm256_and_ps(is_byte, _mm256_castsi256_ps(_mm256_set1_epi32(1)));
        nbytes = _mm256_or_ps(nbytes, _mm256_and_ps(is_short, _mm256_castsi256_ps(_mm256_set1_epi32(3))));
        nbytes = _mm256_or_ps(nbytes, _mm256_and_ps(is_i3, _mm256_castsi256_ps(_mm256_set1_epi32(4))));
        nbytes = _mm256_or_ps(nbytes, _mm256_andnot_ps(is_not_float, _mm256_castsi256_ps(_mm256_set1_epi32(5))));

        int* p_esc = (int*)(&esc);
        int* p_payload = (int*)(&payload);
        int* p_nbytes = (int*)(&nbytes);
        for (int k = 0; k < 8; ++k)
            ib_word(k, zeros, p_esc, p_payload, p_nbytes, rle, dst, bp, b);
    }
    ib_rle(rle, dst, bp, b);
    return bp;
}

// Bitmap fixed-width per-value minimal byte width (no in-band escape collision).
static inline int bmp_width(int maxabs) {
    if (maxabs <= 0x7f)     return 1;
    if (maxabs <= 0x7fff)   return 2;
    if (maxabs <= 0x7fffff) return 3;
    return 4;
}

int main(int argc, char** argv)
{
    const int NX = (argc > 1) ? atoi(argv[1]) : 128;
    const int NY = NX, NZ = NX;
    const float mulfac = (argc > 2) ? (float)atof(argv[2]) : 20.0f;
    const int bx = 32, by = 32, bz = 32, bsz = bx * by * bz;
    if (NX % 32) { printf("NX must be multiple of 32\n"); return 1; }

    const int nbx = NX/32, nby = NY/32, nbz = NZ/32, nblocks = nbx*nby*nbz;
    const size_t nelem = (size_t)NX*NY*NZ;
    printf("bitmap-vs-CPU-RLE: %dx%dx%d nblocks=%d mulfac=%.3f\n", NX,NY,NZ,nblocks,mulfac);

    float* vol = (float*)malloc(nelem * sizeof(float));
    for (size_t i = 0; i < nelem; ++i) {
        int x=(int)(i%NX), y=(int)((i/NX)%NY), z=(int)(i/((size_t)NX*NY));
        float s = sinf(0.11f*x)*cosf(0.07f*y)*sinf(0.05f*z);
        float n = 0.15f*(float)(((i*1103515245u+12345u)>>16)&0x7fff)/32768.0f;
        vol[i] = 0.5f*s + n - 0.075f;
    }

    float* block; posix_memalign((void**)&block, 64, sizeof(float)*bsz);
    float* tmp;   posix_memalign((void**)&tmp, 64, sizeof(float)*bx*8);
    unsigned long* comp; posix_memalign((void**)&comp, 64, sizeof(float)*bsz*2);
    char* scratch = (char*)malloc(bsz*5);

    // RLE accumulators
    RleBuckets rle_b; long rle_total = 0;
    // bitmap accumulators
    long bmp_sig = 0, bmp_hdr = 0, bmp_val = 0, bmp_val_min = 0;
    long bmp_pl_pay = 0;   // per-line-width payload
    long total_nnz = 0;
    long total_nonempty = 0;   // z-lines with >=1 nonzero (mask != 0)
    long bmp_rle_runtok = 0;   // zero-word-RLE control-token bytes
    long mism = 0;
    long w_hist[5] = {0};

    for (int ibz=0; ibz<nbz; ++ibz)
    for (int iby=0; iby<nby; ++iby)
    for (int ibx=0; ibx<nbx; ++ibx) {
        Copy_To_Block(vol, ibx*bx, iby*by, ibz*bz, NX,NY,NZ, (__m128*)block, bx,by,bz);
        Wavelet_Transform_Fast_Forward((__m256*)block, (__m256*)tmp, bx,by,bz);

        // real encoder total (ground truth)
        int ref_bp = 0;
        Run_Length_Encode_Slow(mulfac, block, bsz, comp, ref_bp);
        rle_total += ref_bp;

        // instrumented breakdown (must equal ref_bp)
        RleBuckets b;
        int bp = rle_breakdown(mulfac, block, bsz, scratch, b);
        if (bp != ref_bp) { if (mism<10) printf("  block bp %d != ref %d\n", bp, ref_bp); ++mism; }
        rle_b.zero += b.zero; rle_b.tag += b.tag; rle_b.payload += b.payload;

        // bitmap breakdown on the same quantized ints
        int nnz = 0, maxabs = 0;
        long val_min = 0;
        for (int i=0;i<bsz;++i) {
            int iv = (int)(mulfac * block[i]);
            if (iv != 0) {
                ++nnz;
                int a = iv<0 ? -iv : iv;
                if (a>maxabs) maxabs=a;
                val_min += bmp_width(a);
            }
        }
        int W = bmp_width(maxabs);
        if (nnz==0) W=0; else w_hist[W]++;
        total_nnz += nnz;
        bmp_sig += 4096;
        bmp_hdr += 4;
        bmp_val += (long)nnz * W;
        bmp_val_min += val_min;

        // per-line width: one W per z-line (fixed (ix,iy), z inner).
        // block raster: block[iz*xy + iy*bx + ix], xy = bx*by.
        // Also gather bitmap-compression stats: occupancy (mask!=0 per line)
        // and zero-word-RLE run tokens over raster line order.
        int xy = bx * by;
        int prev = -1, nzero_runs = 0, nnz_runs = 0, blk_nonempty = 0;
        for (int iy=0; iy<by; ++iy)
        for (int ix=0; ix<bx; ++ix) {
            int lnnz=0, lmax=0;
            for (int iz=0; iz<bz; ++iz) {
                int iv = (int)(mulfac * block[iz*xy + iy*bx + ix]);
                if (iv != 0) { ++lnnz; int a = iv<0?-iv:iv; if (a>lmax) lmax=a; }
            }
            int Wl = lnnz ? bmp_width(lmax) : 0;
            bmp_pl_pay += (long)lnnz * Wl;
            int occ = lnnz > 0 ? 1 : 0;
            blk_nonempty += occ;
            if (occ != prev) { if (occ) ++nnz_runs; else ++nzero_runs; prev = occ; }
        }
        total_nonempty += blk_nonempty;
        // zero-run token = 2 B (count up to 1024), nonzero-run token = 1 B (len).
        bmp_rle_runtok += (long)nzero_runs * 2 + (long)nnz_runs * 1;
    }

    long hdr = 8 + 8L*nblocks + 4;
    long rle_all = rle_b.zero + rle_b.tag + rle_b.payload + hdr;
    long bmp_all = bmp_sig + bmp_hdr + bmp_val + hdr;
    double raw = (double)nelem*4.0;

    printf("\n--- CPU full-block RLE (proper path, 8x codes) ---\n");
    printf("  zero-run structure : %10ld B  (%.1f%%)\n", rle_b.zero, 100.0*rle_b.zero/rle_all);
    printf("  escape-tag overhead: %10ld B  (%.1f%%)\n", rle_b.tag,  100.0*rle_b.tag/rle_all);
    printf("  value payload      : %10ld B  (%.1f%%)\n", rle_b.payload, 100.0*rle_b.payload/rle_all);
    printf("  header             : %10ld B\n", hdr);
    printf("  TOTAL              : %10ld B   CR=%.3f\n", rle_all, raw/rle_all);

    printf("\n--- bitmap + per-block fixed width ---\n");
    printf("  significance bitmap: %10ld B  (%.1f%%)  [%d B/block, flat]\n",
           bmp_sig, 100.0*bmp_sig/bmp_all, 4096);
    printf("  nnz metadata       : %10ld B  (derived from popcount)\n", 0L);
    printf("  width header       : %10ld B\n", bmp_hdr);
    printf("  value payload      : %10ld B  (%.1f%%)  [minimal=%ld, padding=%ld]\n",
           bmp_val, 100.0*bmp_val/bmp_all, bmp_val_min, bmp_val - bmp_val_min);
    printf("  header             : %10ld B\n", hdr);
    printf("  TOTAL              : %10ld B   CR=%.3f\n", bmp_all, raw/bmp_all);

    // per-line width variant: sig 4096/blk + 2-bit width table (1024 lines*2b
    // = 256 B/blk) + per-line-width payload.
    long bmp_pl_wtab = 256L * nblocks;
    long bmp_pl_all = bmp_sig + bmp_pl_wtab + bmp_pl_pay + hdr;
    printf("\n--- bitmap + PER-LINE width (2-bit width table) ---\n");
    printf("  significance bitmap: %10ld B  (%.1f%%)\n", bmp_sig, 100.0*bmp_sig/bmp_pl_all);
    printf("  width table (2b/line): %8ld B  (%.1f%%)  [256 B/block]\n",
           bmp_pl_wtab, 100.0*bmp_pl_wtab/bmp_pl_all);
    printf("  value payload      : %10ld B  (%.1f%%)  [minimal=%ld, padding=%ld]\n",
           bmp_pl_pay, 100.0*bmp_pl_pay/bmp_pl_all, bmp_val_min, bmp_pl_pay - bmp_val_min);
    printf("  TOTAL              : %10ld B   CR=%.3f   vs RLE ratio=%.3f\n",
           bmp_pl_all, raw/bmp_pl_all, (double)bmp_pl_all/rle_all);

    // --- Compressed-bitmap variants (both keep per-line width payload) ---
    // Width table now covers only NONEMPTY lines (2 bits each); empty lines
    // carry no value and no width.
    long wtab_ne = (2L * total_nonempty + 7) / 8;

    // (A) two-level: 128 B/block line-occupancy bitmap + 4 B per nonempty mask.
    long tl_sig  = 128L * nblocks + 4L * total_nonempty;
    long tl_all  = tl_sig + wtab_ne + bmp_pl_pay + hdr;
    printf("\n--- bitmap TWO-LEVEL occupancy + per-line width ---\n");
    printf("  occupancy(128B/blk)+masks: %ld B  [%ld occ + %ld masks]\n",
           tl_sig, 128L*nblocks, 4L*total_nonempty);
    printf("  width table (2b/nonempty): %ld B\n", wtab_ne);
    printf("  value payload      : %10ld B  [minimal=%ld]\n", bmp_pl_pay, bmp_val_min);
    printf("  TOTAL              : %10ld B   CR=%.3f   vs RLE ratio=%.3f\n",
           tl_all, raw/tl_all, (double)tl_all/rle_all);

    // (B) zero-word RLE: run tokens + 4 B per nonempty mask.
    long rl_sig = bmp_rle_runtok + 4L * total_nonempty;
    long rl_all = rl_sig + wtab_ne + bmp_pl_pay + hdr;
    printf("\n--- bitmap ZERO-WORD RLE + per-line width ---\n");
    printf("  runtokens+masks    : %10ld B  [%ld tokens + %ld masks]\n",
           rl_sig, bmp_rle_runtok, 4L*total_nonempty);
    printf("  width table (2b/nonempty): %ld B\n", wtab_ne);
    printf("  value payload      : %10ld B\n", bmp_pl_pay);
    printf("  TOTAL              : %10ld B   CR=%.3f   vs RLE ratio=%.3f\n",
           rl_all, raw/rl_all, (double)rl_all/rle_all);
    printf("  nonempty z-lines   : %ld / %ld  (%.1f%%, %.1f/block)\n",
           total_nonempty, 1024L*nblocks, 100.0*total_nonempty/(1024.0*nblocks),
           (double)total_nonempty/nblocks);

    printf("\nnonzero = %ld (%.1f%%)   W hist: 1B=%ld 2B=%ld 3B=%ld 4B=%ld\n",
           total_nnz, 100.0*total_nnz/((double)nblocks*bsz),
           w_hist[1],w_hist[2],w_hist[3],w_hist[4]);
    printf("bitmap/RLE size ratio = %.3f  (<1 = bitmap smaller)\n", (double)bmp_all/rle_all);
    printf("structure: RLE(zero+tag)=%ld  vs  bitmap(sig)=%ld\n",
           rle_b.zero+rle_b.tag, bmp_sig);
    printf("payload:   RLE=%ld  vs  bitmap=%ld  (bitmap minimal=%ld)\n",
           rle_b.payload, bmp_val, bmp_val_min);

    free(vol); free(block); free(tmp); free(comp); free(scratch);
    if (mism) { printf("\nInstrumented total mismatch in %ld blocks\nFAIL\n", mism); return 1; }
    printf("\nPASS (instrumented RLE total matches real encoder)\n");
    return 0;
}
