// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// Kernel 2 (coding) prototype: per-block fixed-width packing of the bitmap
// value stream.  Validates correctness (unpacked values == kernel-1 packed
// values), and reports compressed size (CR vs the RLE path at matched
// quantization) plus per-kernel throughput.

#define DS79_INCLUDE_REG32
#include "hipWaveletRLE.h"
#include "hipWaveletBitmap.h"

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

static inline int unpack_val(const unsigned char* p, int W) {
    unsigned u = 0;
    for (int b = 0; b < W; ++b) u |= (unsigned)p[b] << (8 * b);
    unsigned sign = 1u << (8 * W - 1);
    if (u & sign) u |= ~((sign << 1) - 1);   // sign-extend (no-op for W=4)
    return (int)u;
}

static float time_kernel(void (*launch)(void*), void* ctx, int iters) {
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    launch(ctx);                       // warmup
    hipDeviceSynchronize();
    hipEventRecord(a);
    for (int i = 0; i < iters; ++i) launch(ctx);
    hipEventRecord(b);
    hipEventSynchronize(b);
    float ms = 0; hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a); hipEventDestroy(b);
    return ms / iters;                 // ms per launch
}

struct Ctx {
    const float* d_in; unsigned char* d_rle; size_t* d_rle_sizes;
    unsigned char* d_bmp; size_t* d_bmp_sizes;
    unsigned char* d_code; size_t* d_code_sizes;
    unsigned char* d_codepl; size_t* d_codepl_sizes;
    unsigned char* d_codetl; size_t* d_codetl_sizes;
    unsigned char* d_codetlopt; size_t* d_codetlopt_sizes;
    int NX, NY, NZ, ldimx, ldimxy, nbx, nby, nbz, nblocks;
    float mulfac;
};

static void launch_rle(void* p) {
    Ctx* c = (Ctx*)p;
    dim3 grid(c->nbx, c->nby, c->nbz);
    waveletRLEFusedKernel<<<grid, dim3(256)>>>(
        c->d_in, c->d_rle, c->d_rle_sizes, c->mulfac, c->ldimx, c->ldimxy, nullptr, nullptr);
}
static void launch_bmp(void* p) {
    Ctx* c = (Ctx*)p;
    dim3 grid(c->nbx, c->nby, c->nbz);
    waveletBitmapFusedKernel<<<grid, dim3(256)>>>(
        c->d_in, c->d_bmp, c->d_bmp_sizes, c->mulfac, c->ldimx, c->ldimxy, nullptr, nullptr);
}
static void launch_code(void* p) {
    Ctx* c = (Ctx*)p;
    waveletBitmapCodeKernel<<<c->nblocks, dim3(256)>>>(
        c->d_bmp, c->d_bmp_sizes, c->d_code, c->d_code_sizes);
}
static void launch_codepl(void* p) {
    Ctx* c = (Ctx*)p;
    waveletBitmapCodePerLineKernel<<<c->nblocks, dim3(256)>>>(
        c->d_bmp, c->d_bmp_sizes, c->d_codepl, c->d_codepl_sizes);
}
static void launch_codetl(void* p) {
    Ctx* c = (Ctx*)p;
    waveletBitmapCodeTwoLevelKernel<<<c->nblocks, dim3(256)>>>(
        c->d_bmp, c->d_bmp_sizes, c->d_codetl, c->d_codetl_sizes);
}
static void launch_codetlopt(void* p) {
    Ctx* c = (Ctx*)p;
    waveletBitmapCodeTwoLevelOptKernel<<<c->nblocks, dim3(wbmp_opt::WBMP_OPT_THREADS)>>>(
        c->d_bmp, c->d_bmp_sizes, c->d_codetlopt, c->d_codetlopt_sizes);
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const int NX = (argc > 1) ? atoi(argv[1]) : 128;
    const int NY = NX, NZ = NX;
    const float mulfac = (argc > 2) ? (float)atof(argv[2]) : 20.0f;
    const int iters = (argc > 3) ? atoi(argv[3]) : 100;

    if (NX % 32 != 0) { printf("NX must be multiple of 32\n"); return 1; }

    Ctx c;
    c.NX = NX; c.NY = NY; c.NZ = NZ; c.mulfac = mulfac;
    c.ldimx = NX; c.ldimxy = NX * NY;
    c.nbx = NX / 32; c.nby = NY / 32; c.nbz = NZ / 32;
    c.nblocks = c.nbx * c.nby * c.nbz;
    const size_t nelem = (size_t)NX * NY * NZ;

    printf("bitmap-code test: %dx%dx%d  nblocks=%d  mulfac=%.3f  iters=%d\n",
           NX, NY, NZ, c.nblocks, mulfac, iters);

    std::vector<float> h_in(nelem);
    for (size_t i = 0; i < nelem; ++i) {
        int x = (int)(i % NX);
        int y = (int)((i / NX) % NY);
        int z = (int)(i / ((size_t)NX * NY));
        float s = sinf(0.11f * x) * cosf(0.07f * y) * sinf(0.05f * z);
        float n = 0.15f * (float)(((i * 1103515245u + 12345u) >> 16) & 0x7fff) / 32768.0f;
        h_in[i] = 0.5f * s + n - 0.075f;
    }

    HIPCHECK(hipMalloc(&c.d_in, nelem * sizeof(float)));
    HIPCHECK(hipMemcpy((void*)c.d_in, h_in.data(), nelem * sizeof(float), hipMemcpyHostToDevice));

    const long rle_stride = 4L * WRLE_LDS_BYTES;
    HIPCHECK(hipMalloc(&c.d_rle, (size_t)c.nblocks * rle_stride));
    HIPCHECK(hipMalloc(&c.d_rle_sizes, c.nblocks * sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_bmp, (size_t)c.nblocks * WBMP_SLOT_BYTES));
    HIPCHECK(hipMalloc(&c.d_bmp_sizes, c.nblocks * sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_code, (size_t)c.nblocks * WBMP_CODE_SLOT_BYTES));
    HIPCHECK(hipMalloc(&c.d_code_sizes, c.nblocks * sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_codepl, (size_t)c.nblocks * WBMP_PL_SLOT_BYTES));
    HIPCHECK(hipMalloc(&c.d_codepl_sizes, c.nblocks * sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_codetl, (size_t)c.nblocks * WBMP_TL_SLOT_BYTES));
    HIPCHECK(hipMalloc(&c.d_codetl_sizes, c.nblocks * sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_codetlopt, (size_t)c.nblocks * WBMP_TL_SLOT_BYTES));
    HIPCHECK(hipMalloc(&c.d_codetlopt_sizes, c.nblocks * sizeof(size_t)));

    // Run the pipeline once for correctness + sizes.
    launch_rle(&c); launch_bmp(&c); HIPCHECK(hipDeviceSynchronize());
    launch_code(&c); launch_codepl(&c); launch_codetl(&c); launch_codetlopt(&c);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipGetLastError());

    // ---- Correctness: unpack coded values, compare to kernel-1 int32 ----
    std::vector<unsigned char> h_bmp((size_t)c.nblocks * WBMP_SLOT_BYTES);
    std::vector<unsigned char> h_code((size_t)c.nblocks * WBMP_CODE_SLOT_BYTES);
    std::vector<size_t> h_bmp_sizes(c.nblocks), h_code_sizes(c.nblocks), h_rle_sizes(c.nblocks);
    HIPCHECK(hipMemcpy(h_bmp.data(), c.d_bmp, h_bmp.size(), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_code.data(), c.d_code, h_code.size(), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_bmp_sizes.data(), c.d_bmp_sizes, c.nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_code_sizes.data(), c.d_code_sizes, c.nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_rle_sizes.data(), c.d_rle_sizes, c.nblocks*sizeof(size_t), hipMemcpyDeviceToHost));

    long mismatches = 0;
    long w_hist[5] = {0,0,0,0,0};
    for (int bid = 0; bid < c.nblocks; ++bid) {
        const unsigned char* bin = h_bmp.data() + (long)bid * WBMP_SLOT_BYTES;
        const int32_t* vin = reinterpret_cast<const int32_t*>(bin + WBMP_BITMAP_BYTES);
        int nnz = (int)((h_bmp_sizes[bid] - WBMP_BITMAP_BYTES) / 4);

        const unsigned char* cin = h_code.data() + (long)bid * WBMP_CODE_SLOT_BYTES;
        int W = reinterpret_cast<const int*>(cin + WBMP_BITMAP_BYTES)[0];
        const unsigned char* vpk = cin + WBMP_BITMAP_BYTES + 4;
        if (W >= 1 && W <= 4) ++w_hist[W];

        long exp = WBMP_BITMAP_BYTES + 4 + (long)nnz * W;
        if ((long)h_code_sizes[bid] != exp) { ++mismatches; if (mismatches<10)
            printf("  block %d size %ld != %ld\n", bid, (long)h_code_sizes[bid], exp); }

        for (int i = 0; i < nnz; ++i) {
            int got = unpack_val(vpk + (long)i * W, W);
            if (got != vin[i]) { if (mismatches<20)
                printf("  block %d val %d: %d != %d (W=%d)\n", bid, i, got, vin[i], W);
                ++mismatches; }
        }
    }

    // ---- Per-line coding validation (unpack vs kernel-1 int32) ----
    std::vector<unsigned char> h_codepl((size_t)c.nblocks * WBMP_PL_SLOT_BYTES);
    std::vector<size_t> h_codepl_sizes(c.nblocks);
    HIPCHECK(hipMemcpy(h_codepl.data(), c.d_codepl, h_codepl.size(), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_codepl_sizes.data(), c.d_codepl_sizes, c.nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
    for (int bid = 0; bid < c.nblocks; ++bid) {
        const unsigned char* bin = h_bmp.data() + (long)bid * WBMP_SLOT_BYTES;
        const int32_t* vin = reinterpret_cast<const int32_t*>(bin + WBMP_BITMAP_BYTES);
        const unsigned char* cin = h_codepl.data() + (long)bid * WBMP_PL_SLOT_BYTES;
        const uint32_t* bmp = reinterpret_cast<const uint32_t*>(cin);
        const uint32_t* wtab = reinterpret_cast<const uint32_t*>(cin + WBMP_BITMAP_BYTES);
        const unsigned char* vpk = cin + WBMP_BITMAP_BYTES + WBMP_WTAB_BYTES;
        long idx = 0, ob = 0;
        for (int x = 0; x < 4; ++x)
        for (int tid = 0; tid < 256; ++tid) {
            int L = x*256+tid;
            int nnz = __builtin_popcount(bmp[L]);
            int W = ((wtab[L>>4] >> ((L&15)*2)) & 3) + 1;
            for (int k = 0; k < nnz; ++k) {
                int got = unpack_val(vpk + ob, W); ob += W;
                if (got != vin[idx]) { if (mismatches<20)
                    printf("  [PL] block %d line %d k %d: %d != %d (W=%d)\n", bid, L, k, got, vin[idx], W);
                    ++mismatches; }
                ++idx;
            }
        }
        long exp = WBMP_BITMAP_BYTES + WBMP_WTAB_BYTES + ob;
        if ((long)h_codepl_sizes[bid] != exp) { ++mismatches; if (mismatches<25)
            printf("  [PL] block %d size %ld != %ld\n", bid, (long)h_codepl_sizes[bid], exp); }
    }

    // ---- Two-level occupancy coding validation (decode from occupancy) ----
    std::vector<unsigned char> h_codetl((size_t)c.nblocks * WBMP_TL_SLOT_BYTES);
    std::vector<size_t> h_codetl_sizes(c.nblocks);
    HIPCHECK(hipMemcpy(h_codetl.data(), c.d_codetl, h_codetl.size(), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_codetl_sizes.data(), c.d_codetl_sizes, c.nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
    for (int bid = 0; bid < c.nblocks; ++bid) {
        const unsigned char* bin = h_bmp.data() + (long)bid * WBMP_SLOT_BYTES;
        const int32_t* vin = reinterpret_cast<const int32_t*>(bin + WBMP_BITMAP_BYTES);
        const unsigned char* cin = h_codetl.data() + (long)bid * WBMP_TL_SLOT_BYTES;
        const uint32_t* occ = reinterpret_cast<const uint32_t*>(cin);
        int n_ne = 0;
        for (int w = 0; w < 32; ++w) n_ne += __builtin_popcount(occ[w]);
        const uint32_t* masks = reinterpret_cast<const uint32_t*>(cin + WBMP_OCC_BYTES);
        long wtab_base = WBMP_OCC_BYTES + 4L*n_ne;
        const uint32_t* wtab = reinterpret_cast<const uint32_t*>(cin + wtab_base);
        long vals_base = wtab_base + (2L*n_ne + 7)/8;
        const unsigned char* vpk = cin + vals_base;
        long idx = 0, vo = 0; int rank = 0;
        for (int L = 0; L < 1024; ++L) {
            if (!(occ[L>>5] & (1u << (L&31)))) continue;
            uint32_t m = masks[rank];
            int W = ((wtab[rank>>4] >> ((rank&15)*2)) & 3) + 1;
            int nnz = __builtin_popcount(m);
            for (int k = 0; k < nnz; ++k) {
                int got = unpack_val(vpk + vo, W); vo += W;
                if (got != vin[idx]) { if (mismatches<20)
                    printf("  [TL] block %d line %d k %d: %d != %d (W=%d)\n", bid, L, k, got, vin[idx], W);
                    ++mismatches; }
                ++idx;
            }
            ++rank;
        }
        long exp = vals_base + vo;
        if ((long)h_codetl_sizes[bid] != exp) { ++mismatches; if (mismatches<25)
            printf("  [TL] block %d size %ld != %ld\n", bid, (long)h_codetl_sizes[bid], exp); }
    }

    // ---- Optimized two-level: byte-exact vs reference two-level + decode ----
    std::vector<unsigned char> h_codetlopt((size_t)c.nblocks * WBMP_TL_SLOT_BYTES);
    std::vector<size_t> h_codetlopt_sizes(c.nblocks);
    HIPCHECK(hipMemcpy(h_codetlopt.data(), c.d_codetlopt, h_codetlopt.size(), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_codetlopt_sizes.data(), c.d_codetlopt_sizes, c.nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
    long tlopt_mismatch = 0;
    for (int bid = 0; bid < c.nblocks; ++bid) {
        // Size must match the reference two-level exactly.
        if (h_codetlopt_sizes[bid] != h_codetl_sizes[bid]) { ++tlopt_mismatch; if (tlopt_mismatch<10)
            printf("  [TLopt] block %d size %ld != ref %ld\n", bid,
                   (long)h_codetlopt_sizes[bid], (long)h_codetl_sizes[bid]); continue; }
        // The used prefix [0, used) must be byte-for-byte identical to the
        // reference kernel; the unused tail is intentionally left untouched.
        const unsigned char* a = h_codetlopt.data() + (long)bid * WBMP_TL_SLOT_BYTES;
        const unsigned char* b = h_codetl.data()    + (long)bid * WBMP_TL_SLOT_BYTES;
        long used = (long)h_codetlopt_sizes[bid];
        if (memcmp(a, b, used) != 0) {
            ++tlopt_mismatch;
            for (long i = 0; i < used && tlopt_mismatch < 20; ++i)
                if (a[i] != b[i]) { printf("  [TLopt] block %d byte %ld: %u != %u\n",
                    bid, i, a[i], b[i]); ++tlopt_mismatch; break; }
        }
    }
    mismatches += tlopt_mismatch;
    printf("two-level opt vs reference: %s (%ld byte/size mismatches)\n",
           tlopt_mismatch == 0 ? "byte-exact" : "MISMATCH", tlopt_mismatch);

    // ---- Size / CR at matched quantization ----
    long hdr = 8 + 8L*c.nblocks + 4;
    long rle_total = hdr, bmp_i32_total = hdr, code_total = hdr, codepl_total = hdr, codetl_total = hdr;
    for (int bid = 0; bid < c.nblocks; ++bid) {
        rle_total     += (long)h_rle_sizes[bid];
        bmp_i32_total += (long)h_bmp_sizes[bid];
        code_total    += (long)h_code_sizes[bid];
        codepl_total  += (long)h_codepl_sizes[bid];
        codetl_total  += (long)h_codetl_sizes[bid];
    }
    double raw = (double)nelem * 4.0;
    printf("W histogram (bytes/value): 1B=%ld 2B=%ld 3B=%ld 4B=%ld\n",
           w_hist[1], w_hist[2], w_hist[3], w_hist[4]);
    printf("sizes (bytes):  RLE=%ld  bmp+int32=%ld  bmp+fixedW=%ld  bmp+perline=%ld  bmp+twolevel=%ld\n",
           rle_total, bmp_i32_total, code_total, codepl_total, codetl_total);
    printf("CR:             RLE=%.3f  bmp+fixedW=%.3f  bmp+perline=%.3f  bmp+twolevel=%.3f\n",
           raw/rle_total, raw/code_total, raw/codepl_total, raw/codetl_total);
    printf("vs GPU-RLE size ratio:  fixedW=%.3f  perline=%.3f  twolevel=%.3f  (<1 = smaller)\n",
           (double)code_total / rle_total, (double)codepl_total / rle_total,
           (double)codetl_total / rle_total);

    // ---- Throughput (raw input volume / kernel time) ----
    float t_rle    = time_kernel(launch_rle,    &c, iters);
    float t_bmp    = time_kernel(launch_bmp,    &c, iters);
    float t_code   = time_kernel(launch_code,   &c, iters);
    float t_codepl = time_kernel(launch_codepl, &c, iters);
    float t_codetl = time_kernel(launch_codetl, &c, iters);
    float t_codetlopt = time_kernel(launch_codetlopt, &c, iters);
    double gbraw = raw / 1e9;
    printf("throughput (raw GB/s):  RLE_fused=%.1f  bmp_fused=%.1f  codepl=%.1f  codetl=%.1f  codetl_opt=%.1f  bmp+codetl_opt=%.1f\n",
           gbraw/(t_rle/1e3), gbraw/(t_bmp/1e3), gbraw/(t_codepl/1e3), gbraw/(t_codetl/1e3),
           gbraw/(t_codetlopt/1e3), gbraw/((t_bmp+t_codetlopt)/1e3));
    printf("kernel ms:  RLE_fused=%.3f  bmp_fused=%.3f  code=%.3f  codepl=%.3f  codetl=%.3f  codetl_opt=%.3f\n",
           t_rle, t_bmp, t_code, t_codepl, t_codetl, t_codetlopt);
    printf("two-level coder speedup (codetl / codetl_opt): %.3fx\n", t_codetl / t_codetlopt);

    if (mismatches != 0) { printf("mismatches=%ld\nFAIL\n", mismatches); return 1; }
    printf("PASS\n");
    return 0;
}
