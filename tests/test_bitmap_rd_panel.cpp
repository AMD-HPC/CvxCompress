// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// Rate-distortion sweep on a REAL seismic wavefield panel (e.g. the Marmousi
// RTM snapshots used in the GTAC paper).  For each quantization scale we report,
// on identically CPU-wavelet-transformed / identically-quantized 32^3 blocks:
//
//   distortion : vol_rel_l2 = sqrt( sum (x-xhat)^2 / sum x^2 ), measured in the
//                SPATIAL domain via the real inverse wavelet transform of the
//                truncation-quantized coefficients q/scale  (q = (int)(scale*c)).
//   rate       : compression ratio CR = raw_fp32_bytes / compressed_bytes for
//                - CPU Cvx full-block RLE  (ground truth Run_Length_Encode_Slow)
//                - bitmap TWO-LEVEL occupancy + per-line width  (the new codec)
//                - bitmap per-line width  and  bitmap fixed width  (reference)
//
// Quantization is identical across all codecs, so the distortion column is shared
// and the CR columns are directly comparable at matched fidelity => clean R-D
// curves.  Same header/CR basis as test_bitmap_vs_cpu_rle.cpp.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#define MY_AVX_DEFINED
#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/x86/avx512.h"

#include "Block_Copy.hxx"
#include "Wavelet_Transform_Fast.hxx"
#include "Run_Length_Encode_Slow.hxx"

// per-value minimal signed byte width (no in-band escape collision)
static inline int bmp_width(int maxabs) {
    if (maxabs <= 0x7f)     return 1;
    if (maxabs <= 0x7fff)   return 2;
    if (maxabs <= 0x7fffff) return 3;
    return 4;
}

// 1-D octree (bisection) cost of a 32-bit z-mask: number of non-empty internal
// nodes over sizes {32,16,8,4,2}; the encoder stores 2 bits per such node
// (its two children's occupancy), descending only into non-empty subtrees.
// bits = 2 * bisect_nodes(m).  Caller guarantees m != 0.
static inline int bisect_nodes(unsigned m) {
    int n = 1;  // size-32 root (active line => non-empty)
    for (int j=0;j<2; ++j) if (m & (0xFFFFu << (16*j))) ++n;
    for (int j=0;j<4; ++j) if (m & (0xFFu   << (8*j)))  ++n;
    for (int j=0;j<8; ++j) if (m & (0xFu    << (4*j)))  ++n;
    for (int j=0;j<16;++j) if (m & (0x3u    << (2*j)))  ++n;
    return n;
}

// number of contiguous 1-runs in a 32-bit mask (contiguity probe)
static inline int mask_runs(unsigned m) {
    int runs=0; unsigned prev=0;
    for (int b=0;b<32;++b){ unsigned cur=(m>>b)&1u; if (cur & ~prev) ++runs; prev=cur; }
    return runs;
}

// nonzeros-per-active-line histogram bins: 1,2,3,4,5-6,7-8,9-12,13-16,17-32
static const int NKB = 9;
static inline int khist_bin(int k) {
    if (k<=4) return k-1;
    if (k<=6) return 4;
    if (k<=8) return 5;
    if (k<=12) return 6;
    if (k<=16) return 7;
    return 8;
}

// Load a raw x-fastest fp32 volume of grid (gnz,gny,gnx), center-crop an
// (cz,cy,cx) sub-volume into out (z,y,x order, x fastest).
static bool load_center_crop(const std::string& path, int gnz, int gny, int gnx,
                             int cz, int cy, int cx, std::vector<float>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "open fail %s\n", path.c_str()); return false; }
    std::vector<float> full((size_t)gnz*gny*gnx);
    size_t r = std::fread(full.data(), sizeof(float), full.size(), f);
    std::fclose(f);
    if (r != full.size()) { std::fprintf(stderr, "short read %s (got %zu need %zu)\n",
                                         path.c_str(), r, full.size()); return false; }
    int oz = (gnz - cz) / 2, oy = (gny - cy) / 2, ox = (gnx - cx) / 2;
    if (oz < 0 || oy < 0 || ox < 0) { std::fprintf(stderr, "crop > grid\n"); return false; }
    out.resize((size_t)cz*cy*cx);
    for (int k = 0; k < cz; ++k)
        for (int j = 0; j < cy; ++j) {
            const float* src = &full[((size_t)(oz+k)*gny + (oy+j))*gnx + ox];
            float* dst = &out[((size_t)k*cy + j)*cx];
            std::memcpy(dst, src, (size_t)cx*sizeof(float));
        }
    return true;
}

int main(int argc, char** argv)
{
    std::string panel, out_csv;
    int gnz = 512, gny = 512, gnx = 512;   // grid of the raw file
    int NX = 512;                          // cropped cube edge (mult of 32)
    // scale is applied to the RMS-normalized volume (see below), so it is
    // interpretable as an RMS-relative quantizer step, matching CvxCompress.
    std::vector<float> scales = {4.f,8.f,16.f,32.f,64.f,128.f,256.f,512.f};

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i],"--panel") && i+1<argc) panel = argv[++i];
        else if (!std::strcmp(argv[i],"--out-csv") && i+1<argc) out_csv = argv[++i];
        else if (!std::strcmp(argv[i],"--pdims") && i+3<argc) { gnz=atoi(argv[++i]); gny=atoi(argv[++i]); gnx=atoi(argv[++i]); }
        else if (!std::strcmp(argv[i],"--nx") && i+1<argc) NX = atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--scales") && i+1<argc) {
            scales.clear(); char b[4096]; std::strncpy(b,argv[++i],4095); b[4095]='\0';
            for (char* t=std::strtok(b,","); t; t=std::strtok(nullptr,",")) scales.push_back((float)atof(t));
        }
    }
    if (panel.empty()) { std::fprintf(stderr,
        "Usage: %s --panel FILE.raw [--pdims NZ NY NX] [--nx CROP] [--scales a,b,..] [--out-csv F]\n", argv[0]);
        return 1; }
    if (NX % 32) { std::fprintf(stderr, "--nx must be a multiple of 32\n"); return 1; }

    const int bx=32, by=32, bz=32, bsz=bx*by*bz, xy=bx*by;
    const int NY=NX, NZ=NX;
    const int nbx=NX/32, nby=NY/32, nbz=NZ/32, nblocks=nbx*nby*nbz;
    const size_t nelem=(size_t)NX*NY*NZ;

    std::vector<float> vol;
    if (!load_center_crop(panel, gnz, gny, gnx, NZ, NY, NX, vol)) return 1;

    // Normalize the cropped volume to unit RMS so that `scale` is an
    // RMS-relative quantizer step (raw wavefield amplitudes are ~1e-8).
    // vol_rel_l2 is a ratio and thus invariant to this uniform scaling.
    double ss = 0.0; for (size_t i=0;i<nelem;++i) ss += (double)vol[i]*(double)vol[i];
    double rms = std::sqrt(ss/(double)nelem);
    if (rms > 0) { float inv=(float)(1.0/rms); for (size_t i=0;i<nelem;++i) vol[i]*=inv; }
    std::printf("R-D sweep: panel=%s grid=%dx%dx%d crop=%d^3 nblocks=%d rms=%.3e (normalized)\n",
                panel.c_str(), gnz, gny, gnx, NX, nblocks, rms);

    float* block; posix_memalign((void**)&block, 64, sizeof(float)*bsz);
    float* orig;  posix_memalign((void**)&orig,  64, sizeof(float)*bsz);
    float* recon; posix_memalign((void**)&recon, 64, sizeof(float)*bsz);
    float* tmp;   posix_memalign((void**)&tmp,   64, sizeof(float)*bx*8);
    unsigned long* comp; posix_memalign((void**)&comp, 64, sizeof(float)*bsz*2);

    const int S = (int)scales.size();
    std::vector<double> sse(S,0.0);
    std::vector<long>   rle_total(S,0), tl_nnz(S,0), tl_nonempty(S,0), tl_pl_pay(S,0);
    std::vector<long>   fx_val(S,0);          // fixed per-block-width payload
    std::vector<long>   bisect_bits(S,0);     // 1-D octree z-mask total bits
    std::vector<long>   runs_tot(S,0);        // total z-runs over active lines
    std::vector<long>   khist(S*NKB,0);       // nonzeros-per-active-line histogram
    // full 3D octree (quadtree-xy x bisection-z) significance, per-block adaptive
    std::vector<long>   oct_bits(S,0);        // full-octree significance total bits
    std::vector<long>   sig_oct(S,0), sig_best(S,0);  // significance bytes (sum of per-block)
    std::vector<long>   mode_win(S*4,0);      // per-block winner: flat/2lvl/zbis/oct
    double sig = 0.0;                          // signal energy (scale-independent)

    for (int ibz=0; ibz<nbz; ++ibz)
    for (int iby=0; iby<nby; ++iby)
    for (int ibx=0; ibx<nbx; ++ibx) {
        Copy_To_Block(vol.data(), ibx*bx, iby*by, ibz*bz, NX,NY,NZ, (__m128*)block, bx,by,bz);
        std::memcpy(orig, block, sizeof(float)*bsz);
        for (int i=0;i<bsz;++i) sig += (double)orig[i]*orig[i];
        Wavelet_Transform_Fast_Forward((__m256*)block, (__m256*)tmp, bx,by,bz);

        for (int s=0; s<S; ++s) {
            const float mulfac = scales[s];

            // rate: CPU Cvx RLE ground truth
            int bp=0; Run_Length_Encode_Slow(mulfac, block, bsz, comp, bp);
            rle_total[s] += bp;

            // rate: bitmap fixed per-block width
            int nnz=0, maxabs=0;
            for (int i=0;i<bsz;++i) { int iv=(int)(mulfac*block[i]);
                if (iv){ ++nnz; int a=iv<0?-iv:iv; if(a>maxabs)maxabs=a; } }
            int W = nnz? bmp_width(maxabs):0;
            tl_nnz[s]+=nnz;
            fx_val[s]+=(long)nnz*W;

            // rate: per-line width + occupancy (two-level), plus z-mask
            // structure probes (bisection cost, run count, k-histogram) and the
            // full 32^3 significance for the 3D octree cost model.
            int blk_nonempty=0; long blk_bisbits=0;
            unsigned lines[1024];   // z-mask per (x,y) line: lines[iy*32+ix]
            for (int iy=0; iy<by; ++iy)
            for (int ix=0; ix<bx; ++ix) {
                int lnnz=0, lmax=0; unsigned mask=0;
                for (int iz=0; iz<bz; ++iz) {
                    int iv=(int)(mulfac*block[iz*xy + iy*bx + ix]);
                    if (iv){ ++lnnz; mask|=(1u<<iz); int a=iv<0?-iv:iv; if(a>lmax)lmax=a; }
                }
                lines[iy*32+ix]=mask;
                if (lnnz) {
                    tl_pl_pay[s]+=(long)lnnz*bmp_width(lmax); ++blk_nonempty;
                    long bb=2L*bisect_nodes(mask); blk_bisbits+=bb; bisect_bits[s]+=bb;
                    runs_tot[s]+=mask_runs(mask);
                    khist[s*NKB+khist_bin(lnnz)]++;
                }
            }
            tl_nonempty[s]+=blk_nonempty;

            // Full 3D octree of the 32^3 significance: 8 bits per non-empty node
            // of size {32,16,8,4,2}. Built bottom-up as a pyramid; count[level]
            // = number of non-empty nodes at that level.  z is pooled in pairs at
            // each level (bisection-z), xy is 2x2 pooled (quadtree-xy).
            unsigned L2[256];        // 16x16 nodes, 16-bit z-occ (size-2)
            int c2=0,c4=0,c8=0,c16=0,c32=0;
            auto poolz=[&](unsigned oo)->unsigned {           // pool adjacent z-pairs
                unsigned r=0; for (int g=0; oo; ++g, oo>>=2) if (oo&3u) r|=(1u<<g); return r; };
            for (int Y=0;Y<16;++Y) for (int X=0;X<16;++X) {
                unsigned oo = lines[(2*Y)*32+2*X] | lines[(2*Y)*32+2*X+1]
                            | lines[(2*Y+1)*32+2*X] | lines[(2*Y+1)*32+2*X+1];
                unsigned m = poolz(oo); L2[Y*16+X]=m; c2+=__builtin_popcount(m);
            }
            unsigned L4[64];
            for (int Y=0;Y<8;++Y) for (int X=0;X<8;++X) {
                unsigned oo = L2[(2*Y)*16+2*X] | L2[(2*Y)*16+2*X+1]
                            | L2[(2*Y+1)*16+2*X] | L2[(2*Y+1)*16+2*X+1];
                unsigned m = poolz(oo); L4[Y*8+X]=m; c4+=__builtin_popcount(m);
            }
            unsigned L8[16];
            for (int Y=0;Y<4;++Y) for (int X=0;X<4;++X) {
                unsigned oo = L4[(2*Y)*8+2*X] | L4[(2*Y)*8+2*X+1]
                            | L4[(2*Y+1)*8+2*X] | L4[(2*Y+1)*8+2*X+1];
                unsigned m = poolz(oo); L8[Y*4+X]=m; c8+=__builtin_popcount(m);
            }
            unsigned L16[4];
            for (int Y=0;Y<2;++Y) for (int X=0;X<2;++X) {
                unsigned oo = L8[(2*Y)*4+2*X] | L8[(2*Y)*4+2*X+1]
                            | L8[(2*Y+1)*4+2*X] | L8[(2*Y+1)*4+2*X+1];
                unsigned m = poolz(oo); L16[Y*2+X]=m; c16+=__builtin_popcount(m);
            }
            unsigned oo32 = L16[0]|L16[1]|L16[2]|L16[3];
            c32 = oo32? 1:0;
            long octbits = 8L*(c2+c4+c8+c16+c32);
            oct_bits[s]+=octbits;

            // per-block adaptive significance: pick smallest encoding.
            long s_flat = 4096;
            long s_2lvl = 128 + 4L*blk_nonempty;
            long s_zbis = 128 + (blk_bisbits+7)/8;
            long s_oct  = (octbits+7)/8;
            long best=s_flat; int win=0;
            if (s_2lvl<best){best=s_2lvl;win=1;}
            if (s_zbis<best){best=s_zbis;win=2;}
            if (s_oct <best){best=s_oct; win=3;}
            sig_oct[s]  += s_oct;
            sig_best[s] += best;
            mode_win[s*4+win]++;

            // distortion: dequant q/scale + inverse transform, spatial SSE
            for (int i=0;i<bsz;++i) recon[i]=(float)((int)(mulfac*block[i]))/mulfac;
            Wavelet_Transform_Fast_Inverse((__m256*)recon, (__m256*)tmp, bx,by,bz);
            double e=0.0; for (int i=0;i<bsz;++i){ double d=(double)recon[i]-(double)orig[i]; e+=d*d; }
            sse[s]+=e;
        }
    }

    const long hdr = 8 + 8L*nblocks + 4;
    const double raw = (double)nelem*4.0;

    FILE* csv = nullptr;
    if (!out_csv.empty()) {
        csv = std::fopen(out_csv.c_str(), "w");
        if (csv) std::fprintf(csv, "scale,nnz_pct,vol_rel_l2,rle_bytes,rle_cr,"
                                   "twolevel_bytes,twolevel_cr,perline_bytes,perline_cr,"
                                   "fixed_bytes,fixed_cr,ratio_tl_over_rle,"
                                   "occ_bytes,mask_bytes,wtab_bytes,payload_bytes,"
                                   "nonempty_lines,avg_k,avg_runs,"
                                   "mask_bisect_bytes,twolevel_bisect_bytes,twolevel_bisect_cr,"
                                   "bisect_ratio_over_rle,"
                                   "octree_sig_bytes,octree_total_bytes,octree_cr,octree_ratio_over_rle,"
                                   "best_sig_bytes,best_total_bytes,best_cr,best_ratio_over_rle,"
                                   "mode_flat_pct,mode_2lvl_pct,mode_zbis_pct,mode_oct_pct\n");
    }

    std::printf("\n%-7s %-8s %-11s %-9s %-9s %-9s %-9s %-8s\n",
                "scale","nnz%","vol_rel_l2","RLE_CR","2lvl_CR","perln_CR","fix_CR","2lvl/RLE");
    for (int s=0; s<S; ++s) {
        double vol_rel_l2 = (sig>0)? std::sqrt(sse[s]/sig) : 0.0;
        double nnz_pct = 100.0*tl_nnz[s]/((double)nblocks*bsz);

        long rle_all = rle_total[s] + hdr;

        long wtab_ne  = (2L*tl_nonempty[s] + 7)/8;             // 2b width / nonempty line
        long tl_sig   = 128L*nblocks + 4L*tl_nonempty[s];      // occupancy + per-line masks
        long tl_all   = tl_sig + wtab_ne + tl_pl_pay[s] + hdr;

        long pl_wtab  = 256L*nblocks;                          // 2b width / all 1024 lines
        long pl_all   = 4096L*nblocks + pl_wtab + tl_pl_pay[s] + hdr;

        long fx_all   = 4096L*nblocks + 4L*nblocks + fx_val[s] + hdr;  // sig + 4B width hdr/blk

        double rle_cr = raw/rle_all, tl_cr = raw/tl_all, pl_cr = raw/pl_all, fx_cr = raw/fx_all;

        std::printf("%-7.3f %-8.3f %-11.4e %-9.3f %-9.3f %-9.3f %-9.3f %-8.3f\n",
                    scales[s], nnz_pct, vol_rel_l2, rle_cr, tl_cr, pl_cr, fx_cr,
                    (double)tl_all/rle_all);

        // exact structural split + 1-D bisection z-mask replacement
        long occ_b  = 128L*nblocks;
        long mask_b = 4L*tl_nonempty[s];
        long mbis_b = (bisect_bits[s] + 7)/8;                 // bisection masks
        long tl_bis_all = occ_b + mbis_b + wtab_ne + tl_pl_pay[s] + hdr;
        double avg_k    = tl_nonempty[s]? (double)tl_nnz[s]/tl_nonempty[s] : 0.0;
        double avg_runs = tl_nonempty[s]? (double)runs_tot[s]/tl_nonempty[s] : 0.0;

        long oct_all  = sig_oct[s]  + wtab_ne + tl_pl_pay[s] + hdr;
        long best_all = sig_best[s] + wtab_ne + tl_pl_pay[s] + hdr;
        double mtot = (double)nblocks;

        if (csv) std::fprintf(csv,
            "%.4f,%.4f,%.6e,%ld,%.4f,%ld,%.4f,%ld,%.4f,%ld,%.4f,%.4f,"
            "%ld,%ld,%ld,%ld,%ld,%.3f,%.3f,%ld,%ld,%.4f,%.4f,"
            "%ld,%ld,%.4f,%.4f,%ld,%ld,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f\n",
            scales[s], nnz_pct, vol_rel_l2, rle_all, rle_cr,
            tl_all, tl_cr, pl_all, pl_cr, fx_all, fx_cr, (double)tl_all/rle_all,
            occ_b, mask_b, wtab_ne, tl_pl_pay[s], tl_nonempty[s], avg_k, avg_runs,
            mbis_b, tl_bis_all, raw/tl_bis_all, (double)tl_bis_all/rle_all,
            sig_oct[s], oct_all, raw/oct_all, (double)oct_all/rle_all,
            sig_best[s], best_all, raw/best_all, (double)best_all/rle_all,
            100.0*mode_win[s*4+0]/mtot, 100.0*mode_win[s*4+1]/mtot,
            100.0*mode_win[s*4+2]/mtot, 100.0*mode_win[s*4+3]/mtot);
    }

    // structure breakdown + 1-D bisection z-mask, per scale
    std::printf("\n%-7s %-9s %-7s %-7s | %-10s %-10s %-10s %-10s | %-9s %-9s %-8s\n",
                "scale","nonempty","avg_k","avgruns","occ_B","mask_B","wtab_B","payld_B",
                "maskbisB","2lvlbisCR","bis/RLE");
    for (int s=0; s<S; ++s) {
        long occ_b  = 128L*nblocks;
        long mask_b = 4L*tl_nonempty[s];
        long wtab_ne= (2L*tl_nonempty[s] + 7)/8;
        long mbis_b = (bisect_bits[s] + 7)/8;
        long tl_bis_all = occ_b + mbis_b + wtab_ne + tl_pl_pay[s] + hdr;
        long rle_all = rle_total[s] + hdr;
        double avg_k    = tl_nonempty[s]? (double)tl_nnz[s]/tl_nonempty[s] : 0.0;
        double avg_runs = tl_nonempty[s]? (double)runs_tot[s]/tl_nonempty[s] : 0.0;
        std::printf("%-7.3f %-9ld %-7.2f %-7.2f | %-10ld %-10ld %-10ld %-10ld | %-9ld %-9.3f %-8.3f\n",
                    scales[s], tl_nonempty[s], avg_k, avg_runs,
                    occ_b, mask_b, wtab_ne, tl_pl_pay[s], mbis_b, raw/tl_bis_all,
                    (double)tl_bis_all/rle_all);
    }

    // nonzeros-per-active-line histogram (bisection wins when k is small)
    std::printf("\nnonzeros-per-active-line histogram (%% of active lines):\n");
    std::printf("%-7s %6s %6s %6s %6s %6s %6s %6s %6s %6s\n",
                "scale","k=1","k=2","k=3","k=4","5-6","7-8","9-12","13-16","17-32");
    for (int s=0; s<S; ++s) {
        std::printf("%-7.3f", scales[s]);
        double tot = (double)tl_nonempty[s];
        for (int b=0;b<NKB;++b)
            std::printf(" %6.2f", tot>0? 100.0*khist[s*NKB+b]/tot : 0.0);
        std::printf("\n");
    }

    // full 3D octree + per-block adaptive best-significance codec
    std::printf("\n%-7s %-9s | %-11s %-11s %-11s | %-8s %-8s %-8s %-8s | %-8s %-8s\n",
                "scale","nnz%","2lvl_sigB","octree_sigB","best_sigB",
                "RLE_CR","2lvl_CR","oct_CR","best_CR","oct/RLE","best/RLE");
    for (int s=0; s<S; ++s) {
        double nnz_pct = 100.0*tl_nnz[s]/((double)nblocks*bsz);
        long wtab_ne = (2L*tl_nonempty[s]+7)/8;
        long sig_2l  = 128L*nblocks + 4L*tl_nonempty[s];
        long rle_all = rle_total[s] + hdr;
        long tl_all  = sig_2l + wtab_ne + tl_pl_pay[s] + hdr;
        long oct_all = sig_oct[s]  + wtab_ne + tl_pl_pay[s] + hdr;
        long best_all= sig_best[s] + wtab_ne + tl_pl_pay[s] + hdr;
        std::printf("%-7.3f %-9.3f | %-11ld %-11ld %-11ld | %-8.3f %-8.3f %-8.3f %-8.3f | %-8.3f %-8.3f\n",
                    scales[s], nnz_pct, sig_2l, sig_oct[s], sig_best[s],
                    raw/rle_all, raw/tl_all, raw/oct_all, raw/best_all,
                    (double)oct_all/rle_all, (double)best_all/rle_all);
    }
    std::printf("\nper-block winning significance encoding (%% of blocks):\n");
    std::printf("%-7s %8s %8s %8s %8s\n","scale","flat","2level","zbisect","octree");
    for (int s=0; s<S; ++s) {
        double mtot=(double)nblocks;
        std::printf("%-7.3f %7.2f%% %7.2f%% %7.2f%% %7.2f%%\n", scales[s],
                    100.0*mode_win[s*4+0]/mtot, 100.0*mode_win[s*4+1]/mtot,
                    100.0*mode_win[s*4+2]/mtot, 100.0*mode_win[s*4+3]/mtot);
    }

    if (csv) { std::fclose(csv); std::printf("\nwrote %s\n", out_csv.c_str()); }

    free(block); free(orig); free(recon); free(tmp); free(comp);
    return 0;
}
