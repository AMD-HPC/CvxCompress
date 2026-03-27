// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// Test for wavelet kernels: baseline (float4 pointer) vs buffer instructions.
// Identity copy correctness + bandwidth measurement.
// Z-direction wavelet correctness vs CPU reference.

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>
#include "hip_test_helpers.h"
#include "hipWaveletTransformBuffer.h"

struct KernelMetrics {
    int vgprs;
    int scratch_size;
    int lds_byte_size;
    int occupancy;
};

#ifndef BUILDDIR
#define BUILDDIR "build"
#endif

KernelMetrics get_kernel_metrics_from_asm(const char* kernel_name, const char* kernel_pattern) {
    KernelMetrics m = {-1, -1, -1, -1};
    const char* files[] = {
        BUILDDIR "/hipWaveletTransformBuffer-hip-amdgcn-amd-amdhsa-gfx942.s",
        BUILDDIR "/test_wavelet_buffer-hip-amdgcn-amd-amdhsa-gfx942.s",
        nullptr
    };
    for (int i = 0; files[i]; i++) {
        FILE* f = fopen(files[i], "r");
        if (!f) continue;
        char line[4096];
        bool found = false;
        int since = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, kernel_name) && strstr(line, kernel_pattern)) {
                found = true;
                since = 0;
            }
            if (found) {
                since++;
                char* v;
                if ((v = strstr(line, "NumVgprs:")))    m.vgprs        = atoi(v + 9);
                if ((v = strstr(line, "ScratchSize:")))  m.scratch_size = atoi(v + 12);
                if ((v = strstr(line, "LDSByteSize:")))  m.lds_byte_size = atoi(v + 12);
                if ((v = strstr(line, "Occupancy:")))    m.occupancy    = atoi(v + 10);
                if (m.vgprs >= 0 && m.scratch_size >= 0 && m.lds_byte_size >= 0 && m.occupancy >= 0)
                    break;
                if (since > 400) found = false;
            }
        }
        fclose(f);
        if (m.vgprs >= 0) break;
    }
    return m;
}

void bench(const char* label, const char* asm_name, const char* asm_pattern,
           std::function<hipError_t(float*)> launcher) {
    const int NX = 512, NY = 512, NZ = 512;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_input = nullptr;
    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_input, TOTAL_SIZE));
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));

    randomInit<float>(d_input, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(d_data, d_input, TOTAL_SIZE, hipMemcpyDeviceToDevice));

    HIPCHECK(launcher(d_data));
    HIPCHECK(launcher(d_data));
    HIPCHECK(hipDeviceSynchronize());

    float l2_err = l2Error<float>(d_input, d_data, TOTAL);
    bool passed = (l2_err < 1e-5f);

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK(launcher(d_data));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm(asm_name, asm_pattern);
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << label << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << (passed ? "PASS" : "FAIL") << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_input));
    HIPCHECK(hipFree(d_data));
}

// ---------------------------------------------------------------------------
// CPU reference: Ds79 with stride (from Wavelet_Transform_Slow.cpp)
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

// Z-wavelet correctness test: GPU vs CPU Ds79 reference.
void test_wavelet_z_correctness() {
    const int NX = 128, NY = 128, NZ = 128;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 42ULL);
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_input(TOTAL);
    HIPCHECK(hipMemcpy(h_input.data(), d_data, TOTAL_SIZE, hipMemcpyDeviceToHost));

    // CPU reference
    std::vector<float> h_ref(h_input);
    cpu_wavelet_forward_z(h_ref.data(), NX, NY, NZ, BZ);

    auto check_vs_ref = [&](const char* name, float* d_ptr) {
        std::vector<float> h_gpu(TOTAL);
        HIPCHECK(hipMemcpy(h_gpu.data(), d_ptr, TOTAL_SIZE, hipMemcpyDeviceToHost));
        double max_err = 0.0, sum_sq = 0.0;
        for (int i = 0; i < TOTAL; ++i) {
            double diff = fabs((double)h_ref[i] - (double)h_gpu[i]);
            if (diff > max_err) max_err = diff;
            sum_sq += diff * diff;
        }
        double rms_err = sqrt(sum_sq / TOTAL);
        bool passed = (max_err < 1e-5);
        std::cout << "  " << name << ": " << (passed ? "PASS" : "FAIL")
                  << "  max_err=" << std::scientific << std::setprecision(3) << max_err
                  << "  rms_err=" << rms_err << std::endl;
    };

    // float4_vec tmp variant
    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformBufferZ<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("Z (f4 tmp)", d_data);

    // scalar tmp variant
    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformBufferZScalarTmp<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("Z (scalar tmp)", d_data);

    // saddr Z (f4 tmp)
    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformSaddrZ<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("Z saddr (f4 tmp)", d_data);

    // saddr Z (scalar tmp)
    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformSaddrZScalarTmp<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("Z saddr (scalar tmp)", d_data);

    HIPCHECK(hipFree(d_data));
}

// ZY-wavelet correctness test: GPU Z+Y vs CPU Z then Y reference.
void test_wavelet_zy_correctness() {
    const int NX = 128, NY = 128, NZ = 128;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 42ULL);
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_input(TOTAL);
    HIPCHECK(hipMemcpy(h_input.data(), d_data, TOTAL_SIZE, hipMemcpyDeviceToHost));

    // CPU reference: Z then Y
    std::vector<float> h_ref(h_input);
    cpu_wavelet_forward_z(h_ref.data(), NX, NY, NZ, BZ);
    cpu_wavelet_forward_y(h_ref.data(), NX, NY, NZ, BY);

    auto check_vs_ref = [&](const char* name, float* d_ptr) {
        std::vector<float> h_gpu(TOTAL);
        HIPCHECK(hipMemcpy(h_gpu.data(), d_ptr, TOTAL_SIZE, hipMemcpyDeviceToHost));
        double max_err = 0.0, sum_sq = 0.0;
        for (int i = 0; i < TOTAL; ++i) {
            double diff = fabs((double)h_ref[i] - (double)h_gpu[i]);
            if (diff > max_err) max_err = diff;
            sum_sq += diff * diff;
        }
        double rms_err = sqrt(sum_sq / TOTAL);
        bool passed = (max_err < 1e-5);
        std::cout << "  " << name << ": " << (passed ? "PASS" : "FAIL")
                  << "  max_err=" << std::scientific << std::setprecision(3) << max_err
                  << "  rms_err=" << rms_err << std::endl;
    };

    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformBufferZY<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("ZY buffer", d_data);

    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformSaddrZY<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("ZY saddr", d_data);

    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformBufferZYXor<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("ZY buf xor", d_data);

    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformSaddrZYXor<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("ZY saddr xor", d_data);

    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformBufferZYXor4<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("ZY buf xor4", d_data);

    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformSaddrZYXor4<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("ZY saddr xor4", d_data);

    HIPCHECK(hipFree(d_data));
}

// ZYX-wavelet correctness test: GPU Z+Y+X vs CPU Z then Y then X reference.
void test_wavelet_zyx_correctness() {
    const int NX = 128, NY = 128, NZ = 128;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 42ULL);
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_input(TOTAL);
    HIPCHECK(hipMemcpy(h_input.data(), d_data, TOTAL_SIZE, hipMemcpyDeviceToHost));

    std::vector<float> h_ref(h_input);
    cpu_wavelet_forward_z(h_ref.data(), NX, NY, NZ, BZ);
    cpu_wavelet_forward_y(h_ref.data(), NX, NY, NZ, BY);
    cpu_wavelet_forward_x(h_ref.data(), NX, NY, NZ, BX);

    auto check_vs_ref = [&](const char* name, float* d_ptr) {
        std::vector<float> h_gpu(TOTAL);
        HIPCHECK(hipMemcpy(h_gpu.data(), d_ptr, TOTAL_SIZE, hipMemcpyDeviceToHost));
        double max_err = 0.0, sum_sq = 0.0;
        for (int i = 0; i < TOTAL; ++i) {
            double diff = fabs((double)h_ref[i] - (double)h_gpu[i]);
            if (diff > max_err) max_err = diff;
            sum_sq += diff * diff;
        }
        double rms_err = sqrt(sum_sq / TOTAL);
        bool passed = (max_err < 1e-4);
        std::cout << "  " << name << ": " << (passed ? "PASS" : "FAIL")
                  << "  max_err=" << std::scientific << std::setprecision(3) << max_err
                  << "  rms_err=" << rms_err << std::endl;
    };

    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformBufferZYX<256, 2>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("ZYX buffer", d_data);

    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformSaddrZYX<256, 2>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("ZYX saddr", d_data);

    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformBufferPipeXYZ<256, 2>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("pipeXYZ buf", d_data);

    HIPCHECK(hipMemcpy(d_data, h_input.data(), TOTAL_SIZE, hipMemcpyHostToDevice));
    HIPCHECK((hipWaveletTransformSaddrPipeXYZ<256, 2>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());
    check_vs_ref("pipeXYZ saddr", d_data);

    HIPCHECK(hipFree(d_data));
}

// Z-wavelet performance benchmark on 512^3.
void bench_wavelet_z() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    // Warmup
    HIPCHECK((hipWaveletTransformBufferZ<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformBufferZ<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("BufferZKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "buffer Z" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_z_scalar_tmp() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformBufferZScalarTmp<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformBufferZScalarTmp<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("BufferZScalarTmpKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "buf Z scalar" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_z_saddr() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformSaddrZ<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformSaddrZ<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("SaddrZKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "saddr Z" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_z_saddr_scalar_tmp() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformSaddrZScalarTmp<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformSaddrZScalarTmp<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("SaddrZScalarTmpKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "saddr Z scl" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_zy_buffer() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformBufferZY<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformBufferZY<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("BufferZYKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "buffer ZY" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_zy_saddr() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformSaddrZY<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformSaddrZY<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("SaddrZYKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "saddr ZY" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_zy_buffer_xor() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformBufferZYXor<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformBufferZYXor<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("BufferZYXorKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "buf ZY xor" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_zy_saddr_xor() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformSaddrZYXor<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformSaddrZYXor<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("SaddrZYXorKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "saddr ZY xor" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_zy_buffer_xor4() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformBufferZYXor4<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformBufferZYXor4<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("BufferZYXor4Kernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "buf ZY xor4" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_zy_saddr_xor4() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformSaddrZYXor4<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformSaddrZYXor4<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("SaddrZYXor4Kernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "saddr ZY xor4" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_zyx_buffer() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformBufferZYX<256, 2>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformBufferZYX<256, 2>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("BufferZYXKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "buffer ZYX" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_pipe_xyz_buffer() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformBufferPipeXYZ<256, 2>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformBufferPipeXYZ<256, 2>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("BufferPipeXYZKernel", "ILi256ELi2E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "buf pipeXYZ" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_pipe_xyz_saddr() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformSaddrPipeXYZ<256, 2>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformSaddrPipeXYZ<256, 2>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("SaddrPipeXYZKernel", "ILi256ELi2E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "saddr pipeXYZ" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_pipe_xyz_buffer_occ1() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformBufferPipeXYZ<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformBufferPipeXYZ<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("BufferPipeXYZKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "buf pipe o1" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_pipe_xyz_saddr_occ1() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformSaddrPipeXYZ<256, 1>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformSaddrPipeXYZ<256, 1>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("SaddrPipeXYZKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "saddr pipe o1" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

void bench_wavelet_zyx_saddr() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;
    const int TOTAL = NX * NY * NZ;
    const size_t TOTAL_SIZE = (size_t)TOTAL * sizeof(float);

    float* d_data = nullptr;
    HIPCHECK(hipMalloc(&d_data, TOTAL_SIZE));
    randomInit<float>(d_data, TOTAL, 12345ULL);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK((hipWaveletTransformSaddrZYX<256, 2>(
        d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));

    const int RUNS = 10;
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        HIPCHECK((hipWaveletTransformSaddrZYX<256, 2>(
            d_data, NX, NY, NZ, BX, BY, BZ, NX, NX*NY)));
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / RUNS;
    double bw = 2.0 * TOTAL_SIZE * RUNS / (ms * 1e-3) / 1e9;

    KernelMetrics m = get_kernel_metrics_from_asm("SaddrZYXKernel", "ILi256ELi1E");
    auto s = [](int v) { return v >= 0 ? std::to_string(v) : std::string("N/A"); };

    std::cout << std::setw(14) << "saddr ZYX" << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << avg_ms << " ms | "
              << std::setprecision(1) << std::setw(7) << bw << " GB/s | "
              << "VGPRs " << std::setw(3) << s(m.vgprs) << " | "
              << "Scratch " << std::setw(5) << s(m.scratch_size) << " | "
              << "LDS " << std::setw(5) << s(m.lds_byte_size) << " | "
              << "Occ " << std::setw(2) << s(m.occupancy) << " | "
              << "WAVE" << std::endl;

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_data));
}

int main() {
    const int NX = 512, NY = 512, NZ = 512;
    const int BX = 32, BY = 32, BZ = 32;

    std::cout << "=== Wavelet kernel ld/st comparison ===" << std::endl;
    std::cout << "Grid: 512^3, Block: 32^3, 32 planes in VGPRs" << std::endl;
    std::cout << std::endl;

    bench("scalar", "ScalarBaselineKernel", "ILi256ELi3E",
        [&](float* d) { return hipWaveletTransformScalarBaseline<256, 3>(
            d, NX, NY, NZ, BX, BY, BZ, NX, NX*NY); });

    bench("scalar nt", "ScalarNTBaselineKernel", "ILi256ELi3E",
        [&](float* d) { return hipWaveletTransformScalarNTBaseline<256, 3>(
            d, NX, NY, NZ, BX, BY, BZ, NX, NX*NY); });

    bench("float4 nt", "BaselineKernel", "ILi256ELi3E",
        [&](float* d) { return hipWaveletTransformBaseline<256, 3>(
            d, NX, NY, NZ, BX, BY, BZ, NX, NX*NY); });

    bench("saddr nt", "SaddrKernel", "ILi256ELi3E",
        [&](float* d) { return hipWaveletTransformSaddr<256, 3>(
            d, NX, NY, NZ, BX, BY, BZ, NX, NX*NY); });

    bench("buffer", "BufferKernel", "ILi256ELi3E",
        [&](float* d) { return hipWaveletTransformBuffer<256, 3>(
            d, NX, NY, NZ, BX, BY, BZ, NX, NX*NY); });

    std::cout << std::endl;
    std::cout << "=== Z-direction wavelet correctness (GPU vs CPU Ds79) ===" << std::endl;
    std::cout << "Grid: 128^3, Block: 32^3" << std::endl;
    std::cout << std::endl;
    test_wavelet_z_correctness();

    std::cout << std::endl;
    std::cout << "=== ZY-direction wavelet correctness (GPU vs CPU Ds79) ===" << std::endl;
    std::cout << "Grid: 128^3, Block: 32^3" << std::endl;
    std::cout << std::endl;
    test_wavelet_zy_correctness();

    std::cout << std::endl;
    std::cout << "=== Z-direction wavelet performance ===" << std::endl;
    std::cout << "Grid: 512^3, Block: 32^3" << std::endl;
    std::cout << std::endl;
    bench_wavelet_z();
    bench_wavelet_z_scalar_tmp();
    bench_wavelet_z_saddr();
    bench_wavelet_z_saddr_scalar_tmp();

    std::cout << std::endl;
    std::cout << "=== ZY-direction wavelet performance ===" << std::endl;
    std::cout << "Grid: 512^3, Block: 32^3" << std::endl;
    std::cout << std::endl;
    bench_wavelet_zy_buffer();
    bench_wavelet_zy_saddr();
    bench_wavelet_zy_buffer_xor();
    bench_wavelet_zy_saddr_xor();
    bench_wavelet_zy_buffer_xor4();
    bench_wavelet_zy_saddr_xor4();

    std::cout << std::endl;
    std::cout << "=== ZYX-direction wavelet correctness (GPU vs CPU Ds79) ===" << std::endl;
    std::cout << "Grid: 128^3, Block: 32^3" << std::endl;
    std::cout << std::endl;
    test_wavelet_zyx_correctness();

    std::cout << std::endl;
    std::cout << "=== ZYX-direction wavelet performance ===" << std::endl;
    std::cout << "Grid: 512^3, Block: 32^3" << std::endl;
    std::cout << std::endl;
    bench_wavelet_zyx_buffer();
    bench_wavelet_zyx_saddr();
    bench_wavelet_pipe_xyz_buffer();
    bench_wavelet_pipe_xyz_saddr();
    bench_wavelet_pipe_xyz_buffer_occ1();
    bench_wavelet_pipe_xyz_saddr_occ1();

    return 0;
}
