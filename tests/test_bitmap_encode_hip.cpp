// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// Validation for kernel 1 of the bitmap-significance-split prototype.
//
// Strategy: the bitmap kernel and the RLE kernel share identical wavelet
// transform + identical quantization (int)(mulfac*coeff).  So the set of
// nonzero coefficients and their integer values MUST match bit-for-bit.
// We run both kernels on the same input, decode the RLE stream on the host
// (ground truth), scatter the bitmap+packed values back to a dense grid, and
// require exact equality over all 32768 coefficients per block.

#define DS79_INCLUDE_REG32
#include "hipWaveletRLE.h"
#include "hipWaveletBitmap.h"
#include "quantize_rle_ref.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HIPCHECK(cmd) do { \
    hipError_t _e = (cmd); \
    if (_e != hipSuccess) { \
        printf("HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

// Canonical per-coefficient index shared by both kernels: (x_off, tid, z).
static inline int coeff_index(int x_off, int tid, int z) {
    return (x_off * 256 + tid) * 32 + z;
}

int main(int argc, char** argv)
{
    const int NX = (argc > 1) ? atoi(argv[1]) : 64;
    const int NY = NX, NZ = NX;
    const float mulfac = (argc > 2) ? (float)atof(argv[2]) : 8.0f;

    if (NX % 32 != 0) { printf("NX must be multiple of 32\n"); return 1; }

    const int nbx = NX / 32, nby = NY / 32, nbz = NZ / 32;
    const int nblocks = nbx * nby * nbz;
    const int ldimx = NX, ldimxy = NX * NY;
    const size_t nelem = (size_t)NX * NY * NZ;
    const int COEFFS_PER_BLOCK = 4 * 256 * 32;   // 32768

    printf("bitmap-encode test: %dx%dx%d  nblocks=%d  mulfac=%.3f\n",
           NX, NY, NZ, nblocks, mulfac);

    // ---- Deterministic input (mix of smooth + noise so we get both zeros
    //      and nonzeros after transform+quantize) ----
    std::vector<float> h_in(nelem);
    for (size_t i = 0; i < nelem; ++i) {
        int x = (int)(i % NX);
        int y = (int)((i / NX) % NY);
        int z = (int)(i / ((size_t)NX * NY));
        float s = sinf(0.11f * x) * cosf(0.07f * y) * sinf(0.05f * z);
        float n = 0.15f * (float)(((i * 1103515245u + 12345u) >> 16) & 0x7fff) / 32768.0f;
        h_in[i] = 0.5f * s + n - 0.075f;
    }

    float* d_in = nullptr;
    HIPCHECK(hipMalloc(&d_in, nelem * sizeof(float)));
    HIPCHECK(hipMemcpy(d_in, h_in.data(), nelem * sizeof(float), hipMemcpyHostToDevice));

    // ---- Run RLE kernel (ground truth) ----
    const long rle_stride = 4L * WRLE_LDS_BYTES;
    unsigned char* d_rle = nullptr;
    size_t* d_rle_sizes = nullptr;
    HIPCHECK(hipMalloc(&d_rle, (size_t)nblocks * rle_stride));
    HIPCHECK(hipMalloc(&d_rle_sizes, nblocks * sizeof(size_t)));
    {
        dim3 grid(nbx, nby, nbz);
        waveletRLEFusedKernel<<<grid, dim3(256)>>>(
            d_in, d_rle, d_rle_sizes, mulfac, ldimx, ldimxy, nullptr, nullptr);
        HIPCHECK(hipGetLastError());
    }

    // ---- Run bitmap kernel ----
    unsigned char* d_bmp = nullptr;
    size_t* d_bmp_sizes = nullptr;
    HIPCHECK(hipMalloc(&d_bmp, (size_t)nblocks * WBMP_SLOT_BYTES));
    HIPCHECK(hipMalloc(&d_bmp_sizes, nblocks * sizeof(size_t)));
    HIPCHECK(hipWaveletBitmapFused(d_in, d_bmp, d_bmp_sizes, mulfac,
                                   NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    // ---- Copy results to host ----
    std::vector<unsigned char> h_rle((size_t)nblocks * rle_stride);
    std::vector<unsigned char> h_bmp((size_t)nblocks * WBMP_SLOT_BYTES);
    std::vector<size_t> h_bmp_sizes(nblocks);
    HIPCHECK(hipMemcpy(h_rle.data(), d_rle, h_rle.size(), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_bmp.data(), d_bmp, h_bmp.size(), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_bmp_sizes.data(), d_bmp_sizes, nblocks * sizeof(size_t),
                       hipMemcpyDeviceToHost));

    // ---- Cross-check per block ----
    long total_nonzero = 0, total_coeffs = 0;
    long mismatches = 0, hot_values = 0;
    std::vector<int> truth(COEFFS_PER_BLOCK);
    std::vector<int> recon(COEFFS_PER_BLOCK);

    for (int bid = 0; bid < nblocks; ++bid) {
        // (a) Ground truth: decode the RLE stream.
        const unsigned char* block = h_rle.data() + (long)bid * rle_stride;
        const unsigned char* meta  = block;
        const unsigned char* rle   = block + WRLE_META_PER_BLOCK;
        int stream_offset = 0;
        for (int x_off = 0; x_off < 4; ++x_off) {
            int off = 0;
            for (int tid = 0; tid < 256; ++tid) {
                int my_bytes = meta[x_off * 256 + tid];
                int q[32];
                int got = decode_zline(rle + stream_offset + off, my_bytes, q, 32);
                if (got != 32) {
                    printf("  block %d x_off %d tid %d: decoded %d != 32\n",
                           bid, x_off, tid, got);
                    return 1;
                }
                for (int z = 0; z < 32; ++z) {
                    truth[coeff_index(x_off, tid, z)] = q[z];
                    if (q[z] != 0 && (q[z] > (1 << 23) || q[z] < -(1 << 23)))
                        ++hot_values;
                }
                off += my_bytes;
            }
            stream_offset += off;
        }

        // (b) Reconstruct dense grid from bitmap + packed values.
        const uint32_t* bmp = reinterpret_cast<const uint32_t*>(
            h_bmp.data() + (long)bid * WBMP_SLOT_BYTES);
        const int32_t* vals = reinterpret_cast<const int32_t*>(
            h_bmp.data() + (long)bid * WBMP_SLOT_BYTES + WBMP_BITMAP_BYTES);
        long idx = 0, blk_nnz = 0;
        for (int x_off = 0; x_off < 4; ++x_off) {
            for (int tid = 0; tid < 256; ++tid) {
                uint32_t mask = bmp[x_off * 256 + tid];
                blk_nnz += __builtin_popcount(mask);
                for (int z = 0; z < 32; ++z) {
                    if (mask & (1u << z)) recon[coeff_index(x_off, tid, z)] = vals[idx++];
                    else                  recon[coeff_index(x_off, tid, z)] = 0;
                }
            }
        }

        // (c) block_sizes consistency: 4096 + nnz*4.
        long exp_size = WBMP_BITMAP_BYTES + blk_nnz * 4;
        if ((long)h_bmp_sizes[bid] != exp_size) {
            printf("  block %d: block_size %ld != expected %ld (nnz=%ld)\n",
                   bid, (long)h_bmp_sizes[bid], exp_size, blk_nnz);
            ++mismatches;
        }

        // (d) exact coefficient equality.
        for (int i = 0; i < COEFFS_PER_BLOCK; ++i) {
            ++total_coeffs;
            if (truth[i] != 0) ++total_nonzero;
            if (truth[i] != recon[i]) {
                if (mismatches < 20)
                    printf("  block %d coeff %d: truth %d != recon %d\n",
                           bid, i, truth[i], recon[i]);
                ++mismatches;
            }
        }
    }

    double nz_frac = total_coeffs ? (double)total_nonzero / total_coeffs : 0.0;
    printf("coeffs=%ld  nonzero=%ld (%.1f%%)  hot(|v|>2^23)=%ld  mismatches=%ld\n",
           total_coeffs, total_nonzero, 100.0 * nz_frac, hot_values, mismatches);
    if (hot_values > 0)
        printf("  NOTE: hot values hit RLE VLESC4 float round-trip; lower mulfac "
               "to keep |ival|<2^23 for an exact cross-check.\n");

    hipFree(d_in); hipFree(d_rle); hipFree(d_rle_sizes);
    hipFree(d_bmp); hipFree(d_bmp_sizes);

    if (mismatches != 0) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
