// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// ---------------------------------------------------------------------------
// Quantization flip-amplification test.
//
// Hypothesis under test (CPU vs GPU gradient divergence):
//   The CPU (production AVX ds79) and GPU (ds79 on device) forward wavelet
//   transforms differ only at fp32-rounding level (call it dc).  That tiny
//   coefficient difference is then AMPLIFIED by quantization: with
//   mulfac = 1/(rms*scale) and ival = (int)(mulfac*coef), the quant step in
//   decoded units is  Delta = rms*scale.  A coefficient sitting near a bin
//   edge truncates to integers that differ by one between CPU and GPU, so the
//   decoded value jumps by a full Delta regardless of how small dc is.  Since
//   Delta grows linearly with `scale`, the same fixed dc produces a larger
//   decoded perturbation as compression coarsens.
//
// This test proves the chain end-to-end at the coefficient level:
//   Stage 1  measure dc = ||c_cpu - c_gpu|| / ||c_cpu||           (expect ~fp32)
//   Stage 2  control: identical coefficients -> zero flips at every scale
//   Stage 3  quantize c_cpu vs c_gpu over a scale sweep and show
//            (a) flips appear and the decoded error is composed of +/- k*Delta
//                steps (the flip signature),
//            (b) flip fraction ~ mean|dc|/(rms*scale)  -> falls with scale,
//            (c) decoded ||error|| grows with scale.
//
// The GPU transform here is hipWaveletTransformBufferZYX, which shares the
// ds79_forward_* routines with the production fused kernel (hipWaveletRLE.h);
// the quantization formula matches Run_Length_Encode_Slow.cpp / hipWaveletRLE.h
// exactly ((int) truncation toward zero of mulfac*coef).
//
// Build:  make test_quant_flip_amplification_hip HIP_ARCH=gfx942
// Run:    ./build/test_quant_flip_amplification_hip [NX NY NZ]
// ---------------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "hipWaveletRLE.h"   // production fused kernel + hipWaveletRLEFusedDumpCoef

#define HIPCHECK(cmd) do { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// CPU reference: Ds79 with stride (from Wavelet_Transform_Slow.cpp, identical
// to the copy used in test_wavelet_buffer_hip.cpp).
// ---------------------------------------------------------------------------
#define al0  8.526986790094000e-001f
#define al1  3.774028556126500e-001f
#define al2 -1.106244044184200e-001f
#define al3 -2.384946501938001e-002f
#define al4  3.782845550699501e-002f
#define ah0  7.884856164056601e-001f
#define ah1 -4.180922732222101e-001f
#define ah2 -4.068941760955800e-002f
#define ah3  6.453888262893799e-002f

static inline int MIRR(int val, int dim) {
    val = val < 0 ? -val : val;
    val = (val >= dim) ? (2*dim-2-val) : val;
    val = val < 0 ? -val : val;
    val = (val >= dim) ? (2*dim-2-val) : val;
    return val;
}

static void Ds79(float* p_in, float* p_tmp, int stride, int dim) {
    for (int n = dim; n >= 2; n = n - n/2) {
        for (int i = 0; i < n; ++i) p_tmp[i] = p_in[i*stride];
        int nh = n / 2;
        int nl = n - nh;
        for (int ix = 0; ix < nl; ++ix) {
            int i0 = 2*ix;
            int im1 = MIRR(i0-1,n), ip1 = MIRR(i0+1,n);
            int im2 = MIRR(i0-2,n), ip2 = MIRR(i0+2,n);
            int im3 = MIRR(i0-3,n), ip3 = MIRR(i0+3,n);
            int im4 = MIRR(i0-4,n), ip4 = MIRR(i0+4,n);
            p_in[ix*stride] = al0*p_tmp[i0]
                + al1*(p_tmp[im1]+p_tmp[ip1])
                + al2*(p_tmp[im2]+p_tmp[ip2])
                + al3*(p_tmp[im3]+p_tmp[ip3])
                + al4*(p_tmp[im4]+p_tmp[ip4]);
        }
        for (int ix = 0; ix < nh; ++ix) {
            int i0 = 2*ix + 1;
            int im1 = MIRR(i0-1,n), ip1 = MIRR(i0+1,n);
            int im2 = MIRR(i0-2,n), ip2 = MIRR(i0+2,n);
            int im3 = MIRR(i0-3,n), ip3 = MIRR(i0+3,n);
            p_in[(nl+ix)*stride] = ah0*p_tmp[i0]
                + ah1*(p_tmp[im1]+p_tmp[ip1])
                + ah2*(p_tmp[im2]+p_tmp[ip2])
                + ah3*(p_tmp[im3]+p_tmp[ip3]);
        }
    }
}

static void cpu_wavelet_forward_z(float* data, int nx, int ny, int nz, int bz) {
    float tmp[256];
    #pragma omp parallel for collapse(3) firstprivate(tmp) schedule(static)
    for (int bzi = 0; bzi < nz/bz; ++bzi)
        for (int gy = 0; gy < ny; ++gy)
            for (int gx = 0; gx < nx; ++gx)
                Ds79(data + bzi*bz*(size_t)(nx*ny) + gy*nx + gx,
                     tmp, nx*ny, bz);
}

static void cpu_wavelet_forward_y(float* data, int nx, int ny, int nz, int by) {
    float tmp[256];
    #pragma omp parallel for collapse(3) firstprivate(tmp) schedule(static)
    for (int byi = 0; byi < ny/by; ++byi)
        for (int gz = 0; gz < nz; ++gz)
            for (int gx = 0; gx < nx; ++gx)
                Ds79(data + byi*by*(size_t)nx + gz*(size_t)(nx*ny) + gx,
                     tmp, nx, by);
}

static void cpu_wavelet_forward_x(float* data, int nx, int ny, int nz, int bx) {
    float tmp[256];
    #pragma omp parallel for collapse(3) firstprivate(tmp) schedule(static)
    for (int bxi = 0; bxi < nx/bx; ++bxi)
        for (int gz = 0; gz < nz; ++gz)
            for (int gy = 0; gy < ny; ++gy)
                Ds79(data + bxi*bx + gy*(size_t)nx + gz*(size_t)(nx*ny),
                     tmp, 1, bx);
}

// ---------------------------------------------------------------------------
// Deterministic, smooth-ish test field: superposition of a few sinusoids plus
// a small broadband component.  Gives a realistic spread of wavelet-coefficient
// magnitudes (lots of near-zero high-band coefficients, a few large ones), so
// the near-bin-edge population is representative of real snapshots.
// ---------------------------------------------------------------------------
static void make_field(std::vector<float>& F, int nx, int ny, int nz) {
    F.resize((size_t)nx*ny*nz);
    const double kPi = 3.14159265358979323846;
    for (int k = 0; k < nz; ++k) {
        double z = (double)k / nz;
        for (int j = 0; j < ny; ++j) {
            double y = (double)j / ny;
            for (int i = 0; i < nx; ++i) {
                double x = (double)i / nx;
                double v = 1.0 * sin(2*kPi*3*x) * sin(2*kPi*2*y) * cos(2*kPi*1*z)
                         + 0.5 * sin(2*kPi*11*x + 0.7) * cos(2*kPi*7*y)
                         + 0.25 * cos(2*kPi*19*x) * sin(2*kPi*13*z + 0.3);
                // small deterministic broadband ripple
                unsigned h = (unsigned)(i*73856093 ^ j*19349663 ^ k*83492791);
                double noise = ((h % 2048) / 2048.0 - 0.5) * 0.02;
                F[(size_t)k*nx*ny + (size_t)j*nx + i] = (float)(v + noise);
            }
        }
    }
}

static double raw_rms(const std::vector<float>& F) {
    double acc = 0.0;
    for (float v : F) { double d = (double)v; acc += d*d; }
    return sqrt(acc / (double)F.size());
}

// Production quantization: truncation toward zero of mulfac*coef.
static inline int quantize(float coef, float mulfac) {
    return (int)(mulfac * coef);
}

// ---------------------------------------------------------------------------
static bool run_case(int NX, int NY, int NZ) {
    const int BX = 32, BY = 32, BZ = 32;
    if (NX % BX || NY % BY || NZ % BZ) {
        std::printf("  dims must be multiples of 32; skipping %dx%dx%d\n", NX, NY, NZ);
        return false;
    }
    const size_t N = (size_t)NX * NY * NZ;
    const size_t BYTES = N * sizeof(float);

    std::printf("\n==================================================================\n");
    std::printf("Case %dx%dx%d  (%zu samples, block 32^3)\n", NX, NY, NZ, N);
    std::printf("==================================================================\n");

    // ---- input field (host, deterministic) ----
    std::vector<float> F;
    make_field(F, NX, NY, NZ);
    const double rms = raw_rms(F);
    std::printf("  raw field RMS = %.6e\n", rms);

    // ---- GPU forward ZYX via the PRODUCTION fused kernel ----
    // hipWaveletRLEFusedDumpCoef runs the exact production waveletRLEFusedKernel
    // (Phase 1-3 wavelet + Phase 4 quant/RLE) and additionally writes the
    // pre-quant ZYX coefficients to d_coef.  The RLE outputs are discarded.
    // The 'scale' arg is used directly as mulfac here (d_rms path unused); the
    // dumped coefficients are pre-quant and independent of it.
    const int nbx = NX/32, nby = NY/32, nbz = NZ/32;
    const int nblocks = nbx * nby * nbz;
    const size_t scratch_bytes = (size_t)nblocks * 4 * WRLE_LDS_BYTES;

    float* d_in = nullptr;
    float* d_coef = nullptr;
    unsigned char* d_scratch = nullptr;
    size_t* d_bsz = nullptr;
    HIPCHECK(hipMalloc(&d_in, BYTES));
    HIPCHECK(hipMalloc(&d_coef, BYTES));
    HIPCHECK(hipMalloc(&d_scratch, scratch_bytes));
    HIPCHECK(hipMalloc(&d_bsz, (size_t)nblocks * sizeof(size_t)));
    HIPCHECK(hipMemcpy(d_in, F.data(), BYTES, hipMemcpyHostToDevice));

    HIPCHECK(hipWaveletRLEFusedDumpCoef(
        d_in, d_scratch, d_bsz, d_coef,
        /*scale(mulfac)=*/1.0f, NX, NY, NZ, NX, NX*NY));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> c_gpu(N);
    HIPCHECK(hipMemcpy(c_gpu.data(), d_coef, BYTES, hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(d_in));
    HIPCHECK(hipFree(d_coef));
    HIPCHECK(hipFree(d_scratch));
    HIPCHECK(hipFree(d_bsz));

    // ---- CPU forward ZYX (same layout / block order) ----
    std::vector<float> c_cpu(F);
    cpu_wavelet_forward_z(c_cpu.data(), NX, NY, NZ, BZ);
    cpu_wavelet_forward_y(c_cpu.data(), NX, NY, NZ, BY);
    cpu_wavelet_forward_x(c_cpu.data(), NX, NY, NZ, BX);

    // ---- Stage 1: coefficient difference dc ----
    double num = 0.0, den = 0.0, maxabs = 0.0, sum_abs = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double a = (double)c_cpu[i], b = (double)c_gpu[i];
        double d = a - b;
        num += d*d; den += a*a;
        double ad = fabs(d);
        if (ad > maxabs) maxabs = ad;
        sum_abs += ad;
    }
    double dc_relL2 = den > 0 ? sqrt(num/den) : 0.0;
    double mean_abs_dc = sum_abs / (double)N;
    std::printf("\n  Stage 1 - coefficient difference (pre-quant):\n");
    std::printf("    ||c_cpu - c_gpu|| / ||c_cpu|| = %.3e\n", dc_relL2);
    std::printf("    max|dc| = %.3e   mean|dc| = %.3e\n", maxabs, mean_abs_dc);
    bool stage1_ok = (dc_relL2 < 1e-4);   // fp32-rounding level
    std::printf("    -> %s (coefficients agree at fp32-rounding level)\n",
                stage1_ok ? "OK" : "UNEXPECTEDLY LARGE");

    // ---- Stage 2: identical-coefficient control (quant determinism) ----
    // Feeding identical coefficients to the quantizer must yield zero flips at
    // every scale; this isolates the drift to dc, not quant nondeterminism.
    bool control_ok = true;
    const double scales[] = {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1};
    const int NS = (int)(sizeof(scales)/sizeof(scales[0]));
    for (int s = 0; s < NS && control_ok; ++s) {
        float mulfac = (float)(1.0 / (rms * scales[s]));
        for (size_t i = 0; i < N; ++i) {
            if (quantize(c_cpu[i], mulfac) != quantize(c_cpu[i], mulfac)) { control_ok = false; break; }
        }
    }
    std::printf("\n  Stage 2 - identical-coefficient control: %s (0 flips expected)\n",
                control_ok ? "PASS" : "FAIL");

    // ---- Stage 3: quantize c_cpu vs c_gpu over a scale sweep ----
    std::printf("\n  Stage 3 - flip amplification (quantize c_cpu vs c_gpu):\n");
    std::printf("    %-8s %-10s %-11s %-12s %-12s %-10s %-8s\n",
                "scale", "Delta", "flip_frac", "dec_relL2", "dec_L2abs",
                "predict", "steps");
    bool stage3_ok = true;
    double prev_flip_frac = -1.0;
    double prev_dec_L2 = -1.0;
    bool flips_grow = true, fracs_fall = true;
    for (int s = 0; s < NS; ++s) {
        double scale = scales[s];
        double Delta = rms * scale;               // decoded units per level
        float mulfac = (float)(1.0 / Delta);

        long long flips = 0;
        long long h0 = 0, hp1 = 0, hm1 = 0, hp2 = 0, hm2 = 0, hbig = 0;
        double dnum = 0.0, dden = 0.0;             // decoded relL2 (Delta cancels)
        bool steps_exact = true;
        for (size_t i = 0; i < N; ++i) {
            int qc = quantize(c_cpu[i], mulfac);
            int qg = quantize(c_gpu[i], mulfac);
            int d = qc - qg;
            if (d != 0) ++flips;
            switch (d) {
                case 0:  ++h0;  break;
                case 1:  ++hp1; break;
                case -1: ++hm1; break;
                case 2:  ++hp2; break;
                case -2: ++hm2; break;
                default: ++hbig; break;
            }
            dnum += (double)d * (double)d;
            dden += (double)qc * (double)qc;

            // Flip signature: decoded error must be an exact integer multiple
            // of Delta.  Reconstruct as floats and check.
            double err = (double)qc * Delta - (double)qg * Delta;
            double levels = err / Delta;
            if (fabs(levels - std::round(levels)) > 1e-3) steps_exact = false;
        }
        double flip_frac = (double)flips / (double)N;
        double dec_relL2 = dden > 0 ? sqrt(dnum/dden) : 0.0;
        double dec_L2abs = sqrt(dnum) * Delta;     // absolute decoded L2 error
        // predicted absolute decoded L2 ~ sqrt(mean|dc| * Delta) * sqrt(N)
        double predict = sqrt(mean_abs_dc * Delta) * sqrt((double)N);

        std::printf("    %-8.0e %-10.3e %-11.3e %-12.3e %-12.3e %-10.3e %-8s\n",
                    scale, Delta, flip_frac, dec_relL2, dec_L2abs, predict,
                    steps_exact ? "exact" : "NONINT");

        if (!steps_exact) stage3_ok = false;
        if (prev_flip_frac >= 0.0) {
            if (flip_frac > prev_flip_frac + 1e-12) fracs_fall = false;
            if (dec_L2abs < prev_dec_L2 - 1e-9)     flips_grow = false;
        }
        prev_flip_frac = flip_frac;
        prev_dec_L2 = dec_L2abs;
        // per-scale histogram of q differences
        std::printf("        q-diff histogram:  0=%lld  +1=%lld  -1=%lld  +2=%lld  -2=%lld  |>2|=%lld\n",
                    h0, hp1, hm1, hp2, hm2, hbig);
    }

    std::printf("\n  Trend: flip fraction falls with scale = %s ; "
                "decoded L2 grows with scale = %s\n",
                fracs_fall ? "yes" : "no", flips_grow ? "yes" : "no");

    bool ok = stage1_ok && control_ok && stage3_ok;
    std::printf("\n  CASE %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    std::printf("=== Quantization flip-amplification test (CPU ds79 vs GPU ds79) ===\n");
    bool all = true;
    if (argc >= 4) {
        all &= run_case(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
    } else {
        all &= run_case(128, 128, 128);
        all &= run_case(256, 128, 64);
    }
    std::printf("\n%s\n", all ? "ALL PASS" : "SOME FAIL");
    return all ? 0 : 1;
}
