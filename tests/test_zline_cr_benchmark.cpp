// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// Benchmark: z-line RLE vs full-block RLE compression ratio comparison.
// Links against libcvxcompress for wavelet transform + reference encoder.
// Compile with: make test_zline_cr_benchmark
// Run with: LD_LIBRARY_PATH=. ./build/test_zline_cr_benchmark

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <omp.h>

#define MY_AVX_DEFINED
#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/x86/avx512.h"

#include "Block_Copy.hxx"
#include "Wavelet_Transform_Fast.hxx"
#include "Run_Length_Encode_Slow.hxx"
#include "Read_Raw_Volume.hxx"
#include "quantize_rle_ref.h"

static float compute_block_rms(float* block, int bx, int by, int bz)
{
    double acc = 0.0;
    int n = bx * by * bz;
    for (int i = 0; i < n; ++i) acc += (double)block[i] * (double)block[i];
    return (float)sqrt(acc / (double)n);
}

static float compute_global_rms(float* vol, int nx, int ny, int nz)
{
    long nn = (long)nx * (long)ny * (long)nz;
    double acc = 0.0;
    for (long i = 0; i < nn; ++i) acc += (double)vol[i] * (double)vol[i];
    return (float)sqrt(acc / (double)nn);
}

int main(int argc, char* argv[])
{
    float scale = 1e-1f;
    if (argc > 1) scale = (float)atof(argv[1]);

    int bx = 32, by = 32, bz = 32;
    int block_size = bx * by * bz;
    float raw_bytes = (float)(block_size * sizeof(float));

    int nx, ny, nz;
    float* vol;
    Read_Raw_Volume("", nx, ny, nz, vol);
    printf("Volume: %d x %d x %d, scale=%e\n", nx, ny, nz, scale);

    float global_rms = compute_global_rms(vol, nx, ny, nz);
    float glob_mulfac = (global_rms != 0.0f) ? 1.0f / (global_rms * scale) : 1.0f;
    printf("Global RMS: %e, mulfac: %e\n\n", global_rms, glob_mulfac);

    int nbx = (nx + bx - 1) / bx;
    int nby = (ny + by - 1) / by;
    int nbz = (nz + bz - 1) / bz;
    int nblocks = nbx * nby * nbz;

    float* block;
    posix_memalign((void**)&block, 64, sizeof(float) * block_size);
    float* tmp;
    posix_memalign((void**)&tmp, 64, sizeof(float) * bx * 8);
    unsigned long* comp_buf;
    posix_memalign((void**)&comp_buf, 64, sizeof(float) * block_size * 2);
    unsigned char* zline_buf = (unsigned char*)malloc(block_size * 5);

    long total_full_bytes = 0;
    long total_zline_bytes = 0;
    long total_raw_bytes = 0;
    int nblocks_processed = 0;

    // Per-block stats for reporting
    double worst_ratio_diff = 0.0;
    double sum_ratio_diff = 0.0;

    printf("%-6s %-10s %-10s %-10s %-10s %-10s\n",
           "Block", "FullBytes", "ZlineBytes", "FullCR", "ZlineCR", "ZL/Full");
    printf("%-6s %-10s %-10s %-10s %-10s %-10s\n",
           "-----", "---------", "----------", "------", "-------", "------");

    for (int ibz = 0; ibz < nbz; ++ibz) {
        for (int iby = 0; iby < nby; ++iby) {
            for (int ibx = 0; ibx < nbx; ++ibx) {
                int x0 = ibx * bx;
                int y0 = iby * by;
                int z0 = ibz * bz;

                Copy_To_Block(vol, x0, y0, z0, nx, ny, nz, (__m128*)block, bx, by, bz);
                Wavelet_Transform_Fast_Forward((__m256*)block, (__m256*)tmp, bx, by, bz);

                float mulfac = glob_mulfac;

                // Full-block RLE (existing CPU encoder)
                int full_bytepos = 0;
                Run_Length_Encode_Slow(mulfac, block, block_size, comp_buf, full_bytepos);

                // Z-line RLE: encode each z-line independently.
                // Block layout: block[(iz*by + iy)*bx + ix], X innermost.
                // A z-line at (ix, iy): block[iz*by*bx + iy*bx + ix] for iz=0..bz-1
                int zline_total = 0;
                int xy_stride = bx * by;
                float zline_data[32];

                for (int iy = 0; iy < by; ++iy) {
                    for (int ix = 0; ix < bx; ++ix) {
                        for (int iz = 0; iz < bz; ++iz)
                            zline_data[iz] = block[iz * xy_stride + iy * bx + ix];
                        int enc = quantize_encode_zline(zline_data, mulfac, bz, zline_buf);
                        zline_total += enc;
                    }
                }

                float full_cr = raw_bytes / (float)full_bytepos;
                float zline_cr = raw_bytes / (float)zline_total;
                float ratio = (float)zline_total / (float)full_bytepos;

                total_full_bytes += full_bytepos;
                total_zline_bytes += zline_total;
                total_raw_bytes += (long)(block_size * sizeof(float));

                double diff = (double)full_cr - (double)zline_cr;
                if (diff > worst_ratio_diff) worst_ratio_diff = diff;
                sum_ratio_diff += diff;
                nblocks_processed++;

                if (nblocks_processed <= 20 || nblocks_processed == nblocks) {
                    printf("%-6d %-10d %-10d %-10.2f %-10.2f %-10.3f\n",
                           nblocks_processed, full_bytepos, zline_total,
                           full_cr, zline_cr, ratio);
                }
            }
        }
    }

    float overall_full_cr = (float)total_raw_bytes / (float)total_full_bytes;
    float overall_zline_cr = (float)total_raw_bytes / (float)total_zline_bytes;
    float overall_ratio = (float)total_zline_bytes / (float)total_full_bytes;
    float avg_ratio_diff = (float)(sum_ratio_diff / nblocks_processed);

    printf("\n=== Summary (%d blocks, scale=%e) ===\n", nblocks_processed, scale);
    printf("Full-block RLE:  %ld bytes, CR = %.2f:1\n", total_full_bytes, overall_full_cr);
    printf("Z-line RLE:      %ld bytes, CR = %.2f:1\n", total_zline_bytes, overall_zline_cr);
    printf("Z-line / Full:   %.3f (%.1f%% overhead)\n",
           overall_ratio, (overall_ratio - 1.0f) * 100.0f);
    printf("Avg CR diff:     %.2f\n", avg_ratio_diff);
    printf("Worst CR diff:   %.2f\n", worst_ratio_diff);

    free(block);
    free(tmp);
    free(comp_buf);
    free(zline_buf);
    free(vol);
    return 0;
}
