// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <vector>
#include <thread>
#include <omp.h>
#include <hip/hip_runtime.h>
#include "hipCompress.h"
#include "CvxCompress.hxx"

#define HIPCHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__global__ void initSinKernel(float* data, int nx, int ny, int nz, float kx, float ky, float kz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;
    int iz = idx / (nx * ny);
    int iy = (idx - iz * nx * ny) / nx;
    int ix = idx - iz * nx * ny - iy * nx;
    float x = (float)ix / (float)nx;
    float y = (float)iy / (float)ny;
    float z = (float)iz / (float)nz;
    data[idx] = sinf(kx * x) * sinf(ky * y) * sinf(kz * z);
}

static float hostRMS(const float* data, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)data[i] * (double)data[i];
    return (float)sqrt(sum / n);
}

// Compute RMS on device using plan's copy-to-wavelet infrastructure.
// d_input must be wavelet-layout (32-divisible, contiguous).
static hipError_t deviceRMS(const float* d_input, int nx, int ny, int nz,
                            double* d_rms, hipCompressPlan* plan,
                            hipStream_t s = 0)
{
    return hipCopyToWaveletLayout(
        d_input, nx, nx * ny,
        0, 0, 0, nx, ny, nz,
        nullptr, d_rms, plan, s);
}

// Convenience: compute RMS into plan->d_rms, then compress + synchronize.
static hipError_t compressWithAutoRMS(float scale, const float* d_input,
                                      unsigned char* d_output, long* len,
                                      float* cr, hipCompressPlan* plan,
                                      hipStream_t s = 0)
{
    int nx = plan->nx, ny = plan->ny, nz = plan->nz;
    hipError_t err = deviceRMS(d_input, nx, ny, nz, plan->d_rms, plan, s);
    if (err != hipSuccess) return err;
    err = hipCompress(scale, plan->d_rms, d_input, d_output, plan, s);
    if (err != hipSuccess) return err;
    return hipCompressSynchronize(plan, len, cr);
}

static float maxAbsError(const float* a, const float* b, int n)
{
    float mx = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static bool test_plan_lifecycle()
{
    printf("Test 1: Plan lifecycle\n");
    hipCompressPlan* plan = nullptr;

    // Valid
    hipError_t err = hipCompressCreatePlan(&plan, 128, 128, 128, 0);
    if (err != hipSuccess || !plan) { printf("  FAIL: create valid plan\n"); return false; }
    err = hipCompressDestroyPlan(plan);
    if (err != hipSuccess) { printf("  FAIL: destroy plan\n"); return false; }
    printf("  create/destroy 128^3: PASS\n");

    // Invalid: not multiple of 32
    plan = nullptr;
    err = hipCompressCreatePlan(&plan, 100, 128, 128, 0);
    if (err == hipSuccess) { printf("  FAIL: should reject nx=100\n"); hipCompressDestroyPlan(plan); return false; }
    printf("  reject non-32-multiple: PASS\n");

    // MaxOutputSize
    err = hipCompressCreatePlan(&plan, 64, 64, 64, 0);
    if (err != hipSuccess) { printf("  FAIL: create 64^3 plan\n"); return false; }
    size_t max_sz = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &max_sz));
    int nb = (64/32) * (64/32) * (64/32);
    size_t expected = (size_t)(8 + 8 * nb + 4) + (size_t)nb * 4 * 32768;
    if (max_sz != expected) { printf("  FAIL: MaxOutputSize=%zu expected=%zu\n", max_sz, expected); hipCompressDestroyPlan(plan); return false; }
    printf("  MaxOutputSize=%zu: PASS\n", max_sz);
    hipCompressDestroyPlan(plan);

    return true;
}

static bool test_round_trip()
{
    printf("Test 2: Compress/decompress round-trip\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));

    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipDeviceSynchronize());

    long compressed_length = 0;
    float cr = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &compressed_length, &cr, plan));
    if (cr <= 1.0f || compressed_length <= 0) {
        printf("  FAIL: cr=%.2f compressed_length=%ld\n", cr, compressed_length);
        return false;
    }
    printf("  compress: CR=%.2f, %ld bytes\n", cr, compressed_length);

    HIPCHECK(hipDecompress(d_compressed, d_output, plan, 0));

    std::vector<float> h_in(total), h_out(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), total);
    float max_err = maxAbsError(h_in.data(), h_out.data(), total);
    float rel_err = max_err / rms;
    printf("  decompress: max_err=%.6e, rms=%.6e, rel_max_err=%.6e\n", max_err, rms, rel_err);

    // 3D wavelet inverse amplifies quantization error; bound is generous
    bool pass = (max_err < rms) && (cr > 1.0f);
    printf("  round-trip: %s\n", pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_cr_vs_cpu()
{
    printf("Test 3: CR matches CPU (within z-line gap)\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;
    const int bx = 32, by = 32, bz = 32;

    // GPU compress
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    float* d_input = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_data(total);
    HIPCHECK(hipMemcpy(h_data.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));

    long gpu_length = 0;
    float gpu_cr = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &gpu_length, &gpu_cr, plan));

    // CPU compress
    CvxCompress compressor;
    unsigned int* cpu_compressed = nullptr;
    posix_memalign((void**)&cpu_compressed, 64, total * 5);
    long cpu_length = 0;
    float cpu_cr = compressor.Compress(scale, h_data.data(), N, N, N, bx, by, bz, cpu_compressed, cpu_length);

    float gap = (cpu_cr - gpu_cr) / cpu_cr * 100.0f;
    printf("  CPU CR=%.2f  GPU CR=%.2f  gap=%.1f%%\n", cpu_cr, gpu_cr, gap);

    // Z-line encoding can't exploit cross-zline zero runs; gap grows with CR
    bool pass = (gpu_cr > 1.0f) && (gpu_cr <= cpu_cr);
    printf("  CR comparison: %s\n", pass ? "PASS" : "FAIL");

    free(cpu_compressed);
    hipFree(d_input); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_varying_scale()
{
    printf("Test 4: Varying scale (reuse plan)\n");
    const int N = 128, total = N * N * N;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipDeviceSynchronize());

    float scales[] = {1e-1f, 5e-2f, 1e-2f, 1e-3f};
    float prev_cr = 1e9f;
    bool pass = true;
    for (float s : scales) {
        long len = 0;
        float cr = 0;
        HIPCHECK(compressWithAutoRMS(s, d_input, d_compressed, &len, &cr, plan));
        HIPCHECK(hipDecompress(d_compressed, d_output, plan, 0));

        std::vector<float> h_in(total), h_out(total);
        HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));
        float max_err = maxAbsError(h_in.data(), h_out.data(), total);

        printf("  scale=%.0e: CR=%.2f, max_err=%.4e, len=%ld\n", s, cr, max_err, len);
        if (cr >= prev_cr) {
            printf("    FAIL: CR should decrease with smaller scale\n");
            pass = false;
        }
        prev_cr = cr;
    }
    printf("  varying scale: %s\n", pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_multiple_cycles()
{
    printf("Test 5: Multiple compress/decompress cycles\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    bool pass = true;
    int threads = 256, blocks = (total + threads - 1) / threads;
    for (int step = 0; step < 5; ++step) {
        float k = 10.0f + step * 5.0f;
        initSinKernel<<<blocks, threads>>>(d_input, N, N, N, k, k, k);
        HIPCHECK(hipDeviceSynchronize());

        long len = 0;
        float cr = 0;
        HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len, &cr, plan));
        HIPCHECK(hipDecompress(d_compressed, d_output, plan, 0));

        std::vector<float> h_in(total), h_out(total);
        HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));
        float max_err = maxAbsError(h_in.data(), h_out.data(), total);
        float rms = hostRMS(h_in.data(), total);

        printf("  step %d (k=%.0f): CR=%.2f, max_err=%.4e, rel=%.4e\n",
               step, k, cr, max_err, max_err / rms);
        if (cr <= 1.0f || max_err > rms) pass = false;
    }
    printf("  multiple cycles: %s\n", pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_determinism()
{
    printf("Test 6: Determinism\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    float* d_input = nullptr;
    unsigned char* d_comp1 = nullptr;
    unsigned char* d_comp2 = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_comp1, comp_size));
    HIPCHECK(hipMalloc(&d_comp2, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipDeviceSynchronize());

    long len1 = 0, len2 = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_comp1, &len1, nullptr, plan));
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_comp2, &len2, nullptr, plan));

    bool pass = (len1 == len2);
    if (pass && len1 > 0) {
        std::vector<unsigned char> h1(len1), h2(len2);
        HIPCHECK(hipMemcpy(h1.data(), d_comp1, len1, hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(h2.data(), d_comp2, len2, hipMemcpyDeviceToHost));
        pass = (memcmp(h1.data(), h2.data(), len1) == 0);
    }
    printf("  len1=%ld, len2=%ld, byte-identical=%s\n", len1, len2, pass ? "yes" : "no");
    printf("  determinism: %s\n", pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_comp1); hipFree(d_comp2);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_nondefault_stream_roundtrip()
{
    printf("Test 7: Non-default stream round-trip\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipStream_t stream, aux;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK(hipStreamCreateWithFlags(&aux, hipStreamNonBlocking));
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, aux));

    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads, 0, stream>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);

    long len = 0;
    float cr = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len, &cr, plan, stream));
    HIPCHECK(hipDecompress(d_compressed, d_output, plan, stream));
    HIPCHECK(hipStreamSynchronize(stream));

    std::vector<float> h_in(total), h_out(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), total);
    float max_err = maxAbsError(h_in.data(), h_out.data(), total);
    bool pass = (max_err < rms) && (cr > 1.0f);
    printf("  CR=%.2f, max_err=%.4e, rel=%.4e: %s\n", cr, max_err, max_err / rms, pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    hipStreamDestroy(aux);
    hipStreamDestroy(stream);
    return pass;
}

static bool test_back_to_back_compress()
{
    printf("Test 8: Back-to-back compress reuse\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;

    initSinKernel<<<blocks, threads>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipDeviceSynchronize());
    long len1 = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len1, nullptr, plan));

    initSinKernel<<<blocks, threads>>>(d_input, N, N, N, 30.0f, 30.0f, 30.0f);
    HIPCHECK(hipDeviceSynchronize());
    long len2 = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len2, nullptr, plan));

    HIPCHECK(hipDecompress(d_compressed, d_output, plan, 0));

    std::vector<float> h_in(total), h_out(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), total);
    float max_err = maxAbsError(h_in.data(), h_out.data(), total);
    bool pass = (max_err < rms) && (len1 != len2);
    printf("  len1=%ld, len2=%ld, max_err=%.4e: %s\n", len1, len2, max_err, pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_back_to_back_decompress()
{
    printf("Test 9: Back-to-back decompress (no internal sync)\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    float *d_input, *d_out1, *d_out2, *d_out3;
    unsigned char* d_compressed;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out1, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out2, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out3, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipDeviceSynchronize());

    long len = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len, nullptr, plan));

    HIPCHECK(hipDecompress(d_compressed, d_out1, plan, 0));
    HIPCHECK(hipDecompress(d_compressed, d_out2, plan, 0));
    HIPCHECK(hipDecompress(d_compressed, d_out3, plan, 0));

    std::vector<float> h_in(total), h1(total), h2(total), h3(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h1.data(), d_out1, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h2.data(), d_out2, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h3.data(), d_out3, total * sizeof(float), hipMemcpyDeviceToHost));

    float max1 = maxAbsError(h_in.data(), h1.data(), total);
    float max2 = maxAbsError(h_in.data(), h2.data(), total);
    float max3 = maxAbsError(h_in.data(), h3.data(), total);
    float rms = hostRMS(h_in.data(), total);
    bool pass = (max1 < rms) && (max1 == max2) && (max2 == max3);
    printf("  err=%.4e,%.4e,%.4e identical=%s: %s\n",
           max1, max2, max3, (max1 == max2 && max2 == max3) ? "yes" : "no",
           pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_out1); hipFree(d_out2); hipFree(d_out3); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_pipeline_ordering()
{
    printf("Test 10: Compress pipeline ordering\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipStream_t stream, aux;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK(hipStreamCreateWithFlags(&aux, hipStreamNonBlocking));
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, aux));

    float *d_inputA, *d_inputB, *d_output;
    unsigned char* d_compressed;
    HIPCHECK(hipMalloc(&d_inputA, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_inputB, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads, 0, stream>>>(d_inputA, N, N, N, 20.0f, 20.0f, 20.0f);
    initSinKernel<<<blocks, threads, 0, stream>>>(d_inputB, N, N, N, 40.0f, 40.0f, 40.0f);

    long lenA = 0, lenB = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_inputA, d_compressed, &lenA, nullptr, plan, stream));
    HIPCHECK(compressWithAutoRMS(scale, d_inputB, d_compressed, &lenB, nullptr, plan, stream));
    HIPCHECK(hipDecompress(d_compressed, d_output, plan, stream));

    HIPCHECK(hipStreamSynchronize(stream));

    std::vector<float> h_B(total), h_out(total);
    HIPCHECK(hipMemcpy(h_B.data(), d_inputB, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));
    float rms = hostRMS(h_B.data(), total);
    float max_err = maxAbsError(h_B.data(), h_out.data(), total);
    bool pass = (max_err < rms) && (lenA != lenB);
    printf("  lenA=%ld, lenB=%ld, err_vs_B=%.4e: %s\n", lenA, lenB, max_err, pass ? "PASS" : "FAIL");

    hipFree(d_inputA); hipFree(d_inputB); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    hipStreamDestroy(aux);
    hipStreamDestroy(stream);
    return pass;
}

static bool test_two_plans_two_streams()
{
    printf("Test 11: Two plans, two streams, shared aux (concurrent decompress)\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipStream_t s1, s2, aux;
    HIPCHECK(hipStreamCreate(&s1));
    HIPCHECK(hipStreamCreate(&s2));
    HIPCHECK(hipStreamCreateWithFlags(&aux, hipStreamNonBlocking));

    hipCompressPlan *plan1 = nullptr, *plan2 = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan1, N, N, N, aux));
    HIPCHECK(hipCompressCreatePlan(&plan2, N, N, N, aux));

    float *d_in1, *d_in2, *d_out1, *d_out2;
    unsigned char *d_comp1, *d_comp2;
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan1, &comp_size));
    HIPCHECK(hipMalloc(&d_in1, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_in2, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out1, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out2, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_comp1, comp_size));
    HIPCHECK(hipMalloc(&d_comp2, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads, 0, s1>>>(d_in1, N, N, N, 20.0f, 20.0f, 20.0f);
    initSinKernel<<<blocks, threads, 0, s2>>>(d_in2, N, N, N, 40.0f, 40.0f, 40.0f);

    long len1 = 0, len2 = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_in1, d_comp1, &len1, nullptr, plan1, s1));
    HIPCHECK(compressWithAutoRMS(scale, d_in2, d_comp2, &len2, nullptr, plan2, s2));

    HIPCHECK(hipDecompress(d_comp1, d_out1, plan1, s1));
    HIPCHECK(hipDecompress(d_comp2, d_out2, plan2, s2));

    HIPCHECK(hipStreamSynchronize(s1));
    HIPCHECK(hipStreamSynchronize(s2));

    std::vector<float> h_in1(total), h_in2(total), h_o1(total), h_o2(total);
    HIPCHECK(hipMemcpy(h_in1.data(), d_in1, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_in2.data(), d_in2, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_o1.data(), d_out1, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_o2.data(), d_out2, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms1 = hostRMS(h_in1.data(), total);
    float rms2 = hostRMS(h_in2.data(), total);
    float err1 = maxAbsError(h_in1.data(), h_o1.data(), total);
    float err2 = maxAbsError(h_in2.data(), h_o2.data(), total);
    bool pass = (err1 < rms1) && (err2 < rms2);
    printf("  stream1: err=%.4e  stream2: err=%.4e: %s\n", err1, err2, pass ? "PASS" : "FAIL");

    hipFree(d_in1); hipFree(d_in2); hipFree(d_out1); hipFree(d_out2);
    hipFree(d_comp1); hipFree(d_comp2);
    hipCompressDestroyPlan(plan1); hipCompressDestroyPlan(plan2);
    hipStreamDestroy(aux);
    hipStreamDestroy(s1); hipStreamDestroy(s2);
    return pass;
}

static bool test_async_decompress()
{
    printf("Test 12: Async decompress (no internal sync)\n");
    const int N = 256, total = N * N * N;
    const float scale = 5e-2f;

    hipStream_t stream, aux;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK(hipStreamCreateWithFlags(&aux, hipStreamNonBlocking));
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, aux));

    float *d_input, *d_output;
    unsigned char* d_compressed;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads, 0, stream>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipStreamSynchronize(stream));

    long len = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len, nullptr, plan, stream));
    HIPCHECK(hipDecompress(d_compressed, d_output, plan, stream));

    hipEvent_t ev;
    HIPCHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    HIPCHECK(hipEventRecord(ev, stream));
    hipError_t query = hipEventQuery(ev);
    bool async_ok = (query == hipErrorNotReady);
    printf("  event query: %s (expected NotReady)\n",
           (query == hipErrorNotReady) ? "NotReady" : "Complete");

    HIPCHECK(hipStreamSynchronize(stream));

    std::vector<float> h_in(total), h_out(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), total);
    float max_err = maxAbsError(h_in.data(), h_out.data(), total);
    bool pass = (max_err < rms);
    printf("  err=%.4e: %s (async=%s)\n", max_err, pass ? "PASS" : "FAIL",
           async_ok ? "yes" : "no");

    hipEventDestroy(ev);
    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    hipStreamDestroy(aux);
    hipStreamDestroy(stream);
    return pass;
}

static bool test_stream_ordering_no_sync()
{
    printf("Test 13: Stream ordering (init on user stream, no device sync)\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipStream_t stream, aux;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK(hipStreamCreateWithFlags(&aux, hipStreamNonBlocking));
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, aux));

    float *d_input, *d_output;
    unsigned char* d_compressed;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads, 0, stream>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);

    long len = 0;
    float cr = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len, &cr, plan, stream));
    HIPCHECK(hipDecompress(d_compressed, d_output, plan, stream));
    HIPCHECK(hipStreamSynchronize(stream));

    std::vector<float> h_in(total), h_out(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), total);
    float max_err = maxAbsError(h_in.data(), h_out.data(), total);
    bool pass = (max_err < rms) && (cr > 1.0f);
    printf("  CR=%.2f, err=%.4e: %s\n", cr, max_err, pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    hipStreamDestroy(aux);
    hipStreamDestroy(stream);
    return pass;
}

static bool test_compressed_length_accuracy()
{
    printf("Test 14: Compressed length accuracy (round-trip via host)\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    float *d_input, *d_output;
    unsigned char *d_compressed, *d_compressed2;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));
    HIPCHECK(hipMalloc(&d_compressed2, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipDeviceSynchronize());

    long len = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len, nullptr, plan));

    std::vector<unsigned char> h_comp(len);
    HIPCHECK(hipMemcpy(h_comp.data(), d_compressed, len, hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(d_compressed2, h_comp.data(), len, hipMemcpyHostToDevice));

    HIPCHECK(hipDecompress(d_compressed2, d_output, plan, 0));

    std::vector<float> h_in(total), h_out(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), total);
    float max_err = maxAbsError(h_in.data(), h_out.data(), total);
    bool pass = (max_err < rms);
    printf("  len=%ld, err=%.4e: %s\n", len, max_err, pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed); hipFree(d_compressed2);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_multithread_separate_plans()
{
    printf("Test 15: Multi-threaded, separate plans\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    struct ThreadResult { float max_err; float rms; float cr; bool ok; };

    auto worker = [&](float k, ThreadResult* result) {
        hipStream_t stream, aux;
        hipStreamCreate(&stream);
        hipStreamCreateWithFlags(&aux, hipStreamNonBlocking);
        hipCompressPlan* plan = nullptr;
        hipCompressCreatePlan(&plan, N, N, N, aux);

        float *d_in, *d_out;
        unsigned char* d_comp;
        hipMalloc(&d_in, total * sizeof(float));
        hipMalloc(&d_out, total * sizeof(float));
        size_t cs = 0;
        hipCompressMaxOutputSize(plan, &cs);
        hipMalloc(&d_comp, cs);

        int thr = 256, blk = (total + thr - 1) / thr;
        initSinKernel<<<blk, thr, 0, stream>>>(d_in, N, N, N, k, k, k);

        long len = 0;
        compressWithAutoRMS(scale, d_in, d_comp, &len, &result->cr, plan, stream);
        hipDecompress(d_comp, d_out, plan, stream);
        hipStreamSynchronize(stream);

        std::vector<float> h_in(total), h_out(total);
        hipMemcpy(h_in.data(), d_in, total * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(h_out.data(), d_out, total * sizeof(float), hipMemcpyDeviceToHost);

        result->rms = hostRMS(h_in.data(), total);
        result->max_err = maxAbsError(h_in.data(), h_out.data(), total);
        result->ok = (result->max_err < result->rms) && (result->cr > 1.0f);

        hipFree(d_in); hipFree(d_out); hipFree(d_comp);
        hipCompressDestroyPlan(plan);
        hipStreamDestroy(aux);
        hipStreamDestroy(stream);
    };

    ThreadResult r1{}, r2{};
    std::thread t1(worker, 20.0f, &r1);
    std::thread t2(worker, 40.0f, &r2);
    t1.join();
    t2.join();

    bool pass = r1.ok && r2.ok;
    printf("  thread1: CR=%.2f err=%.4e  thread2: CR=%.2f err=%.4e: %s\n",
           r1.cr, r1.max_err, r2.cr, r2.max_err, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_compress_pending_guard()
{
    printf("Test 16: compress_pending state guard\n");

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, 128, 128, 128, 0));

    if (plan->compress_pending) {
        printf("  FAIL: compress_pending=true at start\n");
        hipCompressDestroyPlan(plan);
        return false;
    }

    float* d_in;
    unsigned char* d_comp;
    int total = 128 * 128 * 128;
    HIPCHECK(hipMalloc(&d_in, total * sizeof(float)));
    size_t comp_size_guard = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size_guard));
    HIPCHECK(hipMalloc(&d_comp, comp_size_guard));
    HIPCHECK(hipMemset(d_in, 0, total * sizeof(float)));

    HIPCHECK(deviceRMS(d_in, 128, 128, 128, plan->d_rms, plan));
    HIPCHECK(hipCompress(5e-2f, plan->d_rms, d_in, d_comp, plan, 0));

    if (!plan->compress_pending) {
        printf("  FAIL: compress_pending=false after hipCompress\n");
        hipFree(d_in); hipFree(d_comp);
        hipCompressDestroyPlan(plan);
        return false;
    }

    // Second hipCompress should be rejected
    hipError_t err = hipCompress(5e-2f, plan->d_rms, d_in, d_comp, plan, 0);
    if (err != hipErrorNotReady) {
        printf("  FAIL: second hipCompress not rejected (err=%d)\n", (int)err);
        hipCompressSynchronize(plan, nullptr, nullptr);
        hipFree(d_in); hipFree(d_comp);
        hipCompressDestroyPlan(plan);
        return false;
    }

    // Synchronize should clear pending
    long len = 0;
    HIPCHECK(hipCompressSynchronize(plan, &len, nullptr));

    if (plan->compress_pending) {
        printf("  FAIL: compress_pending=true after Synchronize\n");
        hipFree(d_in); hipFree(d_comp);
        hipCompressDestroyPlan(plan);
        return false;
    }

    // Synchronize when not pending should return hipErrorNotReady
    err = hipCompressSynchronize(plan, nullptr, nullptr);
    if (err != hipErrorNotReady) {
        printf("  FAIL: Synchronize on idle plan did not return hipErrorNotReady\n");
        hipFree(d_in); hipFree(d_comp);
        hipCompressDestroyPlan(plan);
        return false;
    }

    hipFree(d_in); hipFree(d_comp);
    printf("  pending state transitions: PASS\n");

    hipCompressDestroyPlan(plan);
    return true;
}

static bool test_out_of_order_decompress()
{
    printf("Test 17: Out-of-order decompress (compress A,B,C -> decompress B,A,C)\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    float *d_inA, *d_inB, *d_inC, *d_output;
    unsigned char *d_compA, *d_compB, *d_compC;
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_inA, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_inB, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_inC, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_compA, comp_size));
    HIPCHECK(hipMalloc(&d_compB, comp_size));
    HIPCHECK(hipMalloc(&d_compC, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads>>>(d_inA, N, N, N, 20.0f, 20.0f, 20.0f);
    initSinKernel<<<blocks, threads>>>(d_inB, N, N, N, 30.0f, 30.0f, 30.0f);
    initSinKernel<<<blocks, threads>>>(d_inC, N, N, N, 40.0f, 40.0f, 40.0f);
    HIPCHECK(hipDeviceSynchronize());

    long lenA = 0, lenB = 0, lenC = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_inA, d_compA, &lenA, nullptr, plan));
    HIPCHECK(compressWithAutoRMS(scale, d_inB, d_compB, &lenB, nullptr, plan));
    HIPCHECK(compressWithAutoRMS(scale, d_inC, d_compC, &lenC, nullptr, plan));

    std::vector<float> h_in(total), h_out(total);
    bool pass = true;

    auto check = [&](const char* label, unsigned char* d_comp, float* d_ref) -> bool {
        HIPCHECK(hipDecompress(d_comp, d_output, plan, 0));
        HIPCHECK(hipMemcpy(h_in.data(), d_ref, total * sizeof(float), hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));
        float rms = hostRMS(h_in.data(), total);
        float err = maxAbsError(h_in.data(), h_out.data(), total);
        bool ok = (err < rms);
        printf("  %s: err=%.4e %s\n", label, err, ok ? "ok" : "FAIL");
        return ok;
    };

    if (!check("B", d_compB, d_inB)) pass = false;
    if (!check("A", d_compA, d_inA)) pass = false;
    if (!check("C", d_compC, d_inC)) pass = false;
    if (!check("A", d_compA, d_inA)) pass = false;
    if (!check("C", d_compC, d_inC)) pass = false;
    if (!check("B", d_compB, d_inB)) pass = false;
    if (!check("B", d_compB, d_inB)) pass = false;
    if (!check("A", d_compA, d_inA)) pass = false;
    if (!check("C", d_compC, d_inC)) pass = false;

    printf("  out-of-order (3 permutations, 9 calls): %s\n", pass ? "PASS" : "FAIL");

    hipFree(d_inA); hipFree(d_inB); hipFree(d_inC); hipFree(d_output);
    hipFree(d_compA); hipFree(d_compB); hipFree(d_compC);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_zero_input()
{
    printf("Test 18: Zero input round-trip\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    HIPCHECK(hipMemset(d_input, 0, total * sizeof(float)));
    HIPCHECK(hipMemset(d_output, 0xFF, total * sizeof(float)));

    long len = 0;
    float cr = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len, &cr, plan));
    if (len <= 0) { printf("  FAIL: compressed_length=%ld\n", len); return false; }

    HIPCHECK(hipDecompress(d_compressed, d_output, plan, 0));

    std::vector<float> h_out(total);
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float max_val = 0.0f;
    for (int i = 0; i < total; ++i) {
        float a = fabsf(h_out[i]);
        if (a > max_val) max_val = a;
    }

    bool pass = (max_val == 0.0f) && (len > 0);
    printf("  CR=%.2f, len=%ld, max_output=%.2e: %s\n", cr, len, max_val, pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    return pass;
}

__global__ void fillConstantKernel(float* data, int n, float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

static bool test_near_zero_input()
{
    printf("Test 19: Near-zero input (FLT_MIN)\n");
    const int N = 64, total = N * N * N;
    const float scale = 5e-2f;
    const float tiny_val = FLT_MIN;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    fillConstantKernel<<<blocks, threads>>>(d_input, total, tiny_val);
    HIPCHECK(hipDeviceSynchronize());

    long len = 0;
    float cr = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len, &cr, plan));
    if (len <= 0) { printf("  FAIL: compressed_length=%ld\n", len); return false; }

    HIPCHECK(hipDecompress(d_compressed, d_output, plan, 0));

    std::vector<float> h_out(total);
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    bool has_nan = false, has_inf = false;
    for (int i = 0; i < total; ++i) {
        if (std::isnan(h_out[i])) has_nan = true;
        if (std::isinf(h_out[i])) has_inf = true;
    }

    bool pass = !has_nan && !has_inf && (len > 0);
    printf("  val=%.2e, CR=%.2f, len=%ld, nan=%s, inf=%s: %s\n",
           tiny_val, cr, len, has_nan ? "yes" : "no", has_inf ? "yes" : "no",
           pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_noncubic_roundtrip(int nx, int ny, int nz, float scale, const char* tag)
{
    long total = (long)nx * ny * nz;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, nx, ny, nz, 0));

    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (int)((total + threads - 1) / threads);
    initSinKernel<<<blocks, threads>>>(d_input, nx, ny, nz, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipDeviceSynchronize());

    long len = 0;
    float cr = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len, &cr, plan));
    HIPCHECK(hipDecompress(d_compressed, d_output, plan, 0));

    std::vector<float> h_in(total), h_out(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), (int)total);
    float max_err = maxAbsError(h_in.data(), h_out.data(), (int)total);
    float inv_scale = rms * scale;
    float err_ratio = (inv_scale > 0.0f) ? max_err / inv_scale : 0.0f;

    bool pass = (max_err < rms) && (cr > 1.0f) && (err_ratio < 10.0f);
    printf("  %-20s %4dx%4dx%4d: CR=%.2f, max_err=%.4e, rms=%.4e, err/inv_scale=%.2f: %s\n",
           tag, nx, ny, nz, cr, max_err, rms, err_ratio, pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_noncubic_grids()
{
    printf("Test 20: Non-cubic grid dimensions\n");
    const float scale = 5e-2f;
    bool pass = true;
    struct { int nx, ny, nz; const char* tag; } cases[] = {
        {  32,   32,   32, "single block"},
        {  64,  128,  256, "asymmetric"},
        { 352,  416,  320, "seismic grid"},
        {  32,   32,  512, "thin slab"},
        { 256,   32,   64, "wide-x"},
    };
    for (auto& c : cases) {
        if (!test_noncubic_roundtrip(c.nx, c.ny, c.nz, scale, c.tag))
            pass = false;
    }
    return pass;
}

static bool test_error_bound()
{
    printf("Test 21: Error bound (max_err <= C * rms * scale)\n");
    const float C = 10.0f;
    bool pass = true;

    struct { int nx, ny, nz; float scale; } cases[] = {
        { 128, 128, 128, 1e-1f },
        { 128, 128, 128, 5e-2f },
        { 128, 128, 128, 1e-2f },
        { 128, 128, 128, 1e-3f },
        {  64, 128, 256, 5e-2f },
        { 352, 416, 320, 5e-2f },
    };

    float prev_err = 0.0f;
    float prev_scale = 0.0f;
    int prev_nx = 0;
    for (auto& c : cases) {
        long total = (long)c.nx * c.ny * c.nz;

        hipCompressPlan* plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&plan, c.nx, c.ny, c.nz, 0));

        float* d_input = nullptr;
        float* d_output = nullptr;
        unsigned char* d_compressed = nullptr;
        HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
        HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
        size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
        HIPCHECK(hipMalloc(&d_compressed, comp_size));

        int threads = 256, blocks = (int)((total + threads - 1) / threads);
        initSinKernel<<<blocks, threads>>>(d_input, c.nx, c.ny, c.nz, 20.0f, 20.0f, 20.0f);
        HIPCHECK(hipDeviceSynchronize());

        long len = 0;
        HIPCHECK(compressWithAutoRMS(c.scale, d_input, d_compressed, &len, nullptr, plan));
        HIPCHECK(hipDecompress(d_compressed, d_output, plan, 0));

        std::vector<float> h_in(total), h_out(total);
        HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

        float rms = hostRMS(h_in.data(), (int)total);
        float max_err = maxAbsError(h_in.data(), h_out.data(), (int)total);
        float bound = C * rms * c.scale;
        bool ok = (max_err <= bound);

        printf("  %4dx%4dx%4d scale=%.0e: max_err=%.4e, bound(C=%.0f)=%.4e, ratio=%.2f: %s\n",
               c.nx, c.ny, c.nz, c.scale, max_err, C, bound, max_err / bound, ok ? "ok" : "FAIL");

        if (!ok) pass = false;

        // Check error monotonicity for the 128^3 series
        if (c.nx == 128 && c.ny == 128 && c.nz == 128 && prev_nx == 128 && c.scale < prev_scale) {
            if (max_err > prev_err) {
                printf("    FAIL: error increased with tighter scale (%.4e > %.4e)\n", max_err, prev_err);
                pass = false;
            }
        }
        if (c.nx == 128 && c.ny == 128 && c.nz == 128) {
            prev_err = max_err; prev_scale = c.scale; prev_nx = 128;
        }

        hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
        hipCompressDestroyPlan(plan);
    }
    printf("  error bound: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ---- Tests for hipCopyToWaveletLayout / hipCopyFromWaveletLayout ----

static bool test_copy_zero_padding()
{
    printf("Test 22: CopyToWaveletLayout zero padding\n");

    const int wx = 50, wy = 45, wz = 37;
    int wnx, wny, wnz;
    hipCompressWaveletDims(wx, wy, wz, &wnx, &wny, &wnz);
    printf("  window %dx%dx%d -> wavelet %dx%dx%d\n", wx, wy, wz, wnx, wny, wnz);

    const int nx = wx, ny = wy, nz = wz;
    long src_total = (long)nx * ny * nz;
    long dst_total = (long)wnx * wny * wnz;

    std::vector<float> h_src(src_total);
    for (long i = 0; i < src_total; ++i)
        h_src[i] = (float)(i % 1000) * 0.001f;

    float *d_src, *d_dst;
    HIPCHECK(hipMalloc(&d_src, src_total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_dst, dst_total * sizeof(float)));
    HIPCHECK(hipMemcpy(d_src, h_src.data(), src_total * sizeof(float), hipMemcpyHostToDevice));
    HIPCHECK(hipMemset(d_dst, 0, dst_total * sizeof(float)));

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, wnx, wny, wnz, 0));

    HIPCHECK(hipCopyToWaveletLayout(
        d_src, nx, nx * ny,
        0, 0, 0, wx, wy, wz,
        d_dst, nullptr, plan, 0));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_dst(dst_total);
    HIPCHECK(hipMemcpy(h_dst.data(), d_dst, dst_total * sizeof(float), hipMemcpyDeviceToHost));

    bool pass = true;
    int mismatches = 0;
    for (int iz = 0; iz < wnz; ++iz)
    for (int iy = 0; iy < wny; ++iy)
    for (int ix = 0; ix < wnx; ++ix) {
        bool in_phys = (ix < wx && iy < wy && iz < wz);
        float expected = in_phys
            ? h_src[(long)iz * nx * ny + (long)iy * nx + ix]
            : 0.0f;
        float got = h_dst[(long)iz * wnx * wny + (long)iy * wnx + ix];
        if (got != expected) {
            if (mismatches < 5)
                printf("  mismatch at (%d,%d,%d): expected=%.6f got=%.6f\n",
                       ix, iy, iz, expected, got);
            mismatches++;
            pass = false;
        }
    }
    printf("  mismatches: %d / %ld: %s\n", mismatches, dst_total, pass ? "PASS" : "FAIL");

    hipFree(d_src); hipFree(d_dst);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_copy_nonzero_origin()
{
    printf("Test 23: CopyToWaveletLayout with non-zero origin and strides\n");

    const int ldimx = 100, ldimy = 80, ldimz = 60;
    const int ldimxy = ldimx * ldimy;
    const int x0 = 10, y0 = 5, z0 = 3;
    const int wx = 32, wy = 32, wz = 32;
    int wnx = 32, wny = 32, wnz = 32;
    long vol_total = (long)ldimx * ldimy * ldimz;

    std::vector<float> h_vol(vol_total);
    for (long i = 0; i < vol_total; ++i)
        h_vol[i] = sinf((float)i * 0.01f);

    float *d_vol, *d_dst;
    HIPCHECK(hipMalloc(&d_vol, vol_total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_dst, (long)wnx * wny * wnz * sizeof(float)));
    HIPCHECK(hipMemcpy(d_vol, h_vol.data(), vol_total * sizeof(float), hipMemcpyHostToDevice));

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, wnx, wny, wnz, 0));

    HIPCHECK(hipCopyToWaveletLayout(
        d_vol, ldimx, ldimxy,
        x0, y0, z0, wx, wy, wz,
        d_dst, nullptr, plan, 0));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_dst((long)wnx * wny * wnz);
    HIPCHECK(hipMemcpy(h_dst.data(), d_dst, (long)wnx * wny * wnz * sizeof(float), hipMemcpyDeviceToHost));

    bool pass = true;
    for (int iz = 0; iz < wnz; ++iz)
    for (int iy = 0; iy < wny; ++iy)
    for (int ix = 0; ix < wnx; ++ix) {
        long src_off = (long)(z0 + iz) * ldimxy + (long)(y0 + iy) * ldimx + (x0 + ix);
        float expected = h_vol[src_off];
        float got = h_dst[(long)iz * wnx * wny + (long)iy * wnx + ix];
        if (got != expected) { pass = false; break; }
    }
    printf("  non-zero origin copy: %s\n", pass ? "PASS" : "FAIL");

    hipFree(d_vol); hipFree(d_dst);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_rms_accuracy()
{
    printf("Test 24: RMS accuracy vs host reference\n");

    struct { int nx, ny, nz; const char* tag; } cases[] = {
        {128, 128, 128, "128^3"},
        { 50,  45,  37, "non-32-div"},
        { 32,  32,  32, "single block"},
    };

    bool pass = true;
    for (auto& c : cases) {
        int wx = c.nx, wy = c.ny, wz = c.nz;
        int wnx, wny, wnz;
        hipCompressWaveletDims(wx, wy, wz, &wnx, &wny, &wnz);
        long total = (long)wx * wy * wz;

        std::vector<float> h_data(total);
        for (long i = 0; i < total; ++i)
            h_data[i] = sinf((float)i * 0.01f) * 10.0f;

        float cpu_rms = hostRMS(h_data.data(), (int)total);

        float *d_src;
        HIPCHECK(hipMalloc(&d_src, total * sizeof(float)));
        HIPCHECK(hipMemcpy(d_src, h_data.data(), total * sizeof(float), hipMemcpyHostToDevice));

        hipCompressPlan* plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&plan, wnx, wny, wnz, 0));

        double h_rms = 0;
        HIPCHECK(hipCopyToWaveletLayout(
            d_src, wx, wx * wy,
            0, 0, 0, wx, wy, wz,
            nullptr, plan->d_rms, plan, 0));
        HIPCHECK(hipDeviceSynchronize());
        HIPCHECK(hipMemcpy(&h_rms, plan->d_rms, sizeof(double), hipMemcpyDeviceToHost));

        float gpu_rms = (float)h_rms;
        float rel_err = fabsf(gpu_rms - cpu_rms) / cpu_rms;
        bool ok = (rel_err < 1e-5f);
        printf("  %-15s cpu_rms=%.6e gpu_rms=%.6e rel_err=%.2e: %s\n",
               c.tag, cpu_rms, gpu_rms, rel_err, ok ? "ok" : "FAIL");
        if (!ok) pass = false;

        hipFree(d_src);
        hipCompressDestroyPlan(plan);
    }
    printf("  RMS accuracy: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_rms_zero_constant()
{
    printf("Test 25: RMS for zero and constant input\n");
    bool pass = true;

    {
        const int N = 64;
        long total = (long)N * N * N;
        float *d_src;
        HIPCHECK(hipMalloc(&d_src, total * sizeof(float)));
        HIPCHECK(hipMemset(d_src, 0, total * sizeof(float)));

        hipCompressPlan* plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

        HIPCHECK(hipCopyToWaveletLayout(
            d_src, N, N * N,
            0, 0, 0, N, N, N,
            nullptr, plan->d_rms, plan, 0));
        HIPCHECK(hipDeviceSynchronize());

        double h_rms = -1.0;
        HIPCHECK(hipMemcpy(&h_rms, plan->d_rms, sizeof(double), hipMemcpyDeviceToHost));
        bool ok = (h_rms == 0.0);
        printf("  zero input: rms=%.2e: %s\n", h_rms, ok ? "ok" : "FAIL");
        if (!ok) pass = false;

        hipFree(d_src);
        hipCompressDestroyPlan(plan);
    }

    {
        const int N = 64;
        long total = (long)N * N * N;
        const float val = 7.5f;

        float *d_src;
        HIPCHECK(hipMalloc(&d_src, total * sizeof(float)));
        int threads = 256, blocks = (int)((total + threads - 1) / threads);
        fillConstantKernel<<<blocks, threads>>>(d_src, (int)total, val);
        HIPCHECK(hipDeviceSynchronize());

        hipCompressPlan* plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

        HIPCHECK(hipCopyToWaveletLayout(
            d_src, N, N * N,
            0, 0, 0, N, N, N,
            nullptr, plan->d_rms, plan, 0));
        HIPCHECK(hipDeviceSynchronize());

        double h_rms = 0;
        HIPCHECK(hipMemcpy(&h_rms, plan->d_rms, sizeof(double), hipMemcpyDeviceToHost));
        float rel_err = fabsf((float)h_rms - val) / val;
        bool ok = (rel_err < 1e-6f);
        printf("  constant(%.1f): rms=%.6e rel_err=%.2e: %s\n", val, h_rms, rel_err, ok ? "ok" : "FAIL");
        if (!ok) pass = false;

        hipFree(d_src);
        hipCompressDestroyPlan(plan);
    }
    printf("  RMS zero/constant: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_copy_modes()
{
    printf("Test 26: CopyToWaveletLayout mode dispatch (copy+RMS, copy-only, RMS-only)\n");
    const int N = 64;
    long total = (long)N * N * N;
    bool pass = true;

    float *d_src, *d_dst;
    HIPCHECK(hipMalloc(&d_src, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_dst, total * sizeof(float)));

    int threads = 256, blocks = (int)((total + threads - 1) / threads);
    initSinKernel<<<blocks, threads>>>(d_src, N, N, N, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipDeviceSynchronize());

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

    // copy+RMS
    HIPCHECK(hipMemset(d_dst, 0, total * sizeof(float)));
    HIPCHECK(hipCopyToWaveletLayout(
        d_src, N, N * N,
        0, 0, 0, N, N, N,
        d_dst, plan->d_rms, plan, 0));
    HIPCHECK(hipDeviceSynchronize());

    double h_rms1 = 0;
    HIPCHECK(hipMemcpy(&h_rms1, plan->d_rms, sizeof(double), hipMemcpyDeviceToHost));

    std::vector<float> h_src(total), h_dst(total);
    HIPCHECK(hipMemcpy(h_src.data(), d_src, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_dst.data(), d_dst, total * sizeof(float), hipMemcpyDeviceToHost));
    bool copy_ok = (memcmp(h_src.data(), h_dst.data(), total * sizeof(float)) == 0);
    bool rms_ok = (h_rms1 > 0.0);
    printf("  copy+RMS: copy_match=%s rms=%.4e: %s\n",
           copy_ok ? "yes" : "no", h_rms1, (copy_ok && rms_ok) ? "ok" : "FAIL");
    if (!copy_ok || !rms_ok) pass = false;

    // copy-only (d_rms_out = nullptr)
    HIPCHECK(hipMemset(d_dst, 0xFF, total * sizeof(float)));
    HIPCHECK(hipCopyToWaveletLayout(
        d_src, N, N * N,
        0, 0, 0, N, N, N,
        d_dst, nullptr, plan, 0));
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(h_dst.data(), d_dst, total * sizeof(float), hipMemcpyDeviceToHost));
    copy_ok = (memcmp(h_src.data(), h_dst.data(), total * sizeof(float)) == 0);
    printf("  copy-only: copy_match=%s: %s\n", copy_ok ? "yes" : "no", copy_ok ? "ok" : "FAIL");
    if (!copy_ok) pass = false;

    // RMS-only (d_dst = nullptr)
    HIPCHECK(hipCopyToWaveletLayout(
        d_src, N, N * N,
        0, 0, 0, N, N, N,
        nullptr, plan->d_rms, plan, 0));
    HIPCHECK(hipDeviceSynchronize());
    double h_rms2 = 0;
    HIPCHECK(hipMemcpy(&h_rms2, plan->d_rms, sizeof(double), hipMemcpyDeviceToHost));
    bool rms_match = (fabsf((float)h_rms2 - (float)h_rms1) < 1e-10f);
    printf("  RMS-only: rms=%.4e (match=%s): %s\n", h_rms2, rms_match ? "yes" : "no",
           rms_match ? "ok" : "FAIL");
    if (!rms_match) pass = false;

    // Both NULL should error
    hipError_t err = hipCopyToWaveletLayout(
        d_src, N, N * N,
        0, 0, 0, N, N, N,
        nullptr, nullptr, plan, 0);
    bool err_ok = (err != hipSuccess);
    printf("  both-null: error=%s: %s\n", err_ok ? "yes" : "no", err_ok ? "ok" : "FAIL");
    if (!err_ok) pass = false;

    printf("  mode dispatch: %s\n", pass ? "PASS" : "FAIL");
    hipFree(d_src); hipFree(d_dst);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_copy_roundtrip()
{
    printf("Test 27: Full round-trip CopyTo -> Compress -> Decompress -> CopyFrom\n");
    const float scale = 5e-2f;
    bool pass = true;

    struct { int wx, wy, wz; const char* tag; } cases[] = {
        { 50,  45,  37, "non-32-div"},
        {128, 128, 128, "exact-32"},
        { 32,  32,  32, "single-block"},
        { 33,  33,  33, "minimal-pad"},
        {100, 200,  60, "asymmetric"},
    };

    for (auto& c : cases) {
        int wnx, wny, wnz;
        hipCompressWaveletDims(c.wx, c.wy, c.wz, &wnx, &wny, &wnz);
        long src_total = (long)c.wx * c.wy * c.wz;
        long wav_total = (long)wnx * wny * wnz;

        std::vector<float> h_src(src_total);
        for (long i = 0; i < src_total; ++i)
            h_src[i] = sinf((float)i * 0.01f);

        float *d_src, *d_wav, *d_wav_out, *d_dst;
        unsigned char *d_comp;
        HIPCHECK(hipMalloc(&d_src, src_total * sizeof(float)));
        HIPCHECK(hipMalloc(&d_wav, wav_total * sizeof(float)));
        HIPCHECK(hipMalloc(&d_wav_out, wav_total * sizeof(float)));
        HIPCHECK(hipMalloc(&d_dst, src_total * sizeof(float)));
        HIPCHECK(hipMemcpy(d_src, h_src.data(), src_total * sizeof(float), hipMemcpyHostToDevice));

        hipCompressPlan* plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&plan, wnx, wny, wnz, 0));
        size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
        HIPCHECK(hipMalloc(&d_comp, comp_size));

        HIPCHECK(hipCopyToWaveletLayout(
            d_src, c.wx, c.wx * c.wy,
            0, 0, 0, c.wx, c.wy, c.wz,
            d_wav, plan->d_rms, plan, 0));

        long len = 0;
        float cr = 0;
        HIPCHECK(hipCompress(scale, plan->d_rms, d_wav, d_comp, plan, 0));
        HIPCHECK(hipCompressSynchronize(plan, &len, &cr));
        HIPCHECK(hipDecompress(d_comp, d_wav_out, plan, 0));

        HIPCHECK(hipMemset(d_dst, 0, src_total * sizeof(float)));
        HIPCHECK(hipCopyFromWaveletLayout(
            d_wav_out, d_dst, c.wx, c.wx * c.wy,
            0, 0, 0, c.wx, c.wy, c.wz, plan, 0));
        HIPCHECK(hipDeviceSynchronize());

        std::vector<float> h_dst(src_total);
        HIPCHECK(hipMemcpy(h_dst.data(), d_dst, src_total * sizeof(float), hipMemcpyDeviceToHost));

        float rms = hostRMS(h_src.data(), (int)src_total);
        float max_err = maxAbsError(h_src.data(), h_dst.data(), (int)src_total);
        float bound = 10.0f * rms * scale;
        bool ok = (max_err < bound) && (cr > 1.0f);
        printf("  %-15s %3dx%3dx%3d->%3dx%3dx%3d: CR=%.2f max_err=%.4e bound=%.4e: %s\n",
               c.tag, c.wx, c.wy, c.wz, wnx, wny, wnz, cr, max_err, bound, ok ? "ok" : "FAIL");
        if (!ok) pass = false;

        hipFree(d_src); hipFree(d_wav); hipFree(d_wav_out); hipFree(d_dst); hipFree(d_comp);
        hipCompressDestroyPlan(plan);
    }
    printf("  copy round-trip: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_copy_from_wavelet()
{
    printf("Test 28: CopyFromWaveletLayout correctness\n");

    const int wx = 50, wy = 45, wz = 37;
    int wnx, wny, wnz;
    hipCompressWaveletDims(wx, wy, wz, &wnx, &wny, &wnz);
    long wav_total = (long)wnx * wny * wnz;
    long dst_total = (long)wx * wy * wz;

    std::vector<float> h_wav(wav_total);
    for (long i = 0; i < wav_total; ++i)
        h_wav[i] = (float)i * 0.001f;

    float *d_wav, *d_dst;
    HIPCHECK(hipMalloc(&d_wav, wav_total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_dst, dst_total * sizeof(float)));
    HIPCHECK(hipMemcpy(d_wav, h_wav.data(), wav_total * sizeof(float), hipMemcpyHostToDevice));
    HIPCHECK(hipMemset(d_dst, 0, dst_total * sizeof(float)));

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, wnx, wny, wnz, 0));

    HIPCHECK(hipCopyFromWaveletLayout(
        d_wav, d_dst, wx, wx * wy,
        0, 0, 0, wx, wy, wz, plan, 0));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_dst(dst_total);
    HIPCHECK(hipMemcpy(h_dst.data(), d_dst, dst_total * sizeof(float), hipMemcpyDeviceToHost));

    bool pass = true;
    int mismatches = 0;
    for (int iz = 0; iz < wz; ++iz)
    for (int iy = 0; iy < wy; ++iy)
    for (int ix = 0; ix < wx; ++ix) {
        long wav_off = (long)iz * wnx * wny + (long)iy * wnx + ix;
        long dst_off = (long)iz * wx * wy + (long)iy * wx + ix;
        if (h_dst[dst_off] != h_wav[wav_off]) {
            mismatches++;
            pass = false;
        }
    }
    printf("  mismatches: %d / %ld: %s\n", mismatches, dst_total, pass ? "PASS" : "FAIL");

    hipFree(d_wav); hipFree(d_dst);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_wavelet_dims_helper()
{
    printf("Test 29: hipCompressWaveletDims helper\n");
    bool pass = true;
    struct { int in; int expected; } cases[] = {
        {  1, 32}, { 31, 32}, { 32, 32}, { 33, 64},
        { 64, 64}, {100, 128}, {256, 256}, {257, 288},
    };
    for (auto& c : cases) {
        int out = hipCompressWaveletDim(c.in);
        if (out != c.expected) {
            printf("  FAIL: hipCompressWaveletDim(%d) = %d, expected %d\n", c.in, out, c.expected);
            pass = false;
        }
    }
    printf("  wavelet dims: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

static void bench_grid_size(int nx, int ny, int nz, float scale)
{
    long total = (long)nx * ny * nz;
    float data_MB = (float)total * sizeof(float) / (1024.0f * 1024.0f);

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, nx, ny, nz, 0));

    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (int)((total + threads - 1) / threads);
    initSinKernel<<<blocks, threads>>>(d_input, nx, ny, nz, 40.0f, 40.0f, 40.0f);
    HIPCHECK(hipDeviceSynchronize());

    const int warmup = 10, iters = 50;
    long len = 0;

    for (int i = 0; i < warmup; ++i) {
        compressWithAutoRMS(scale, d_input, d_compressed, &len, nullptr, plan);
        hipDecompress(d_compressed, d_output, plan, 0);
    }

    hipEvent_t t0, t1, t2;
    hipEventCreate(&t0); hipEventCreate(&t1); hipEventCreate(&t2);

    hipEventRecord(t0);
    for (int i = 0; i < iters; ++i)
        compressWithAutoRMS(scale, d_input, d_compressed, &len, nullptr, plan);
    hipEventRecord(t1);
    for (int i = 0; i < iters; ++i)
        hipDecompress(d_compressed, d_output, plan, 0);
    hipEventRecord(t2);
    hipEventSynchronize(t2);

    float fwd_ms = 0, inv_ms = 0;
    hipEventElapsedTime(&fwd_ms, t0, t1);
    hipEventElapsedTime(&inv_ms, t1, t2);
    fwd_ms /= iters;
    inv_ms /= iters;

    float cr = (float)(total * sizeof(float)) / (float)len;
    float fwd_bw = data_MB / fwd_ms * 1000.0f / 1024.0f;
    float inv_bw = data_MB / inv_ms * 1000.0f / 1024.0f;

    printf("  %4dx%4dx%4d  %7.1f  %5.1f:1  %8.3f  %8.1f  %8.3f  %8.1f  %5.2fx\n",
           nx, ny, nz, data_MB, cr, fwd_ms, fwd_bw, inv_ms, inv_bw, inv_ms / fwd_ms);

    hipEventDestroy(t0); hipEventDestroy(t1); hipEventDestroy(t2);
    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
}

static void bench_throughput()
{
    const float scale = 5e-2f;
    printf("Benchmark: API throughput, scale=%.0e, sin(40x)sin(40y)sin(40z)\n", scale);
    printf("  %16s  %7s  %5s  %8s  %8s  %8s  %8s  %5s\n",
           "grid", "MB", "CR", "fwd(ms)", "fwd GB/s", "inv(ms)", "inv GB/s", "ratio");

    int sizes[][3] = {
        {352, 416, 320},
        {256, 256, 512},
        {128, 512, 512},
        { 32,  32,  32},
        { 64,  64,  64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
    };

    for (auto& s : sizes)
        bench_grid_size(s[0], s[1], s[2], scale);
}

static void generateRadialSinc(float* vol, int nx, int ny, int nz,
                               double k, double omega, double t)
{
    double x0 = nx / 2.0, y0 = ny / 2.0, z0 = nz / 2.0;
    #pragma omp parallel for collapse(3)
    for (int iz = 0; iz < nz; ++iz)
        for (int iy = 0; iy < ny; ++iy)
            for (int ix = 0; ix < nx; ++ix) {
                double x = ix - x0, y = iy - y0, z = iz - z0;
                double r = sqrt(x*x + y*y + z*z);
                double arg = k * r - omega * t;
                float val = (fabs(arg) < 1e-10) ? 1.0f : (float)(sin(arg) / arg);
                vol[(size_t)iz * nx * ny + (size_t)iy * nx + ix] = val;
            }
}

struct ErrorNorms {
    double l1, l2, linf;
    double ref_rms;
};

static ErrorNorms computeErrors(const float* ref, const float* cmp, long n)
{
    double sum_abs = 0, sum_sq = 0, max_abs = 0, sum_ref_sq = 0;
    #pragma omp parallel for reduction(+:sum_abs,sum_sq,sum_ref_sq) reduction(max:max_abs)
    for (long i = 0; i < n; ++i) {
        double d = (double)cmp[i] - (double)ref[i];
        double a = fabs(d);
        sum_abs += a;
        sum_sq += d * d;
        if (a > max_abs) max_abs = a;
        sum_ref_sq += (double)ref[i] * (double)ref[i];
    }
    ErrorNorms e;
    e.l1 = sum_abs / n;
    e.l2 = sqrt(sum_sq / n);
    e.linf = max_abs;
    e.ref_rms = sqrt(sum_ref_sq / n);
    return e;
}

static void bench_radial_sinc_sweep()
{
    const float scale = 5e-2f;
    const double k = 2.0 * M_PI / 5.0, omega = k, t = 10.0;
    const int bx = 32, by = 32, bz = 32;
    const int warmup = 5, iters = 20;

    int sizes[] = {256, 512, 1024};
    const int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    struct Result {
        int N; float data_MB;
        float cpu_cr, gpu_cr, cr_overhead;
        float gpu_fwd_ms, gpu_inv_ms, cpu_fwd_ms, cpu_inv_ms;
        float fwd_speedup, inv_speedup;
        int cpu_fwd_threads, cpu_inv_threads;
        ErrorNorms gpu_err, cpu_err;
    };
    std::vector<Result> results;

    int max_threads = omp_get_max_threads();

    printf("Benchmark: Radial sinc, scale=%.0e, k=%.1f, omega=%.1f, t=%.0f, max_omp=%d\n\n",
           scale, k, omega, t, max_threads);

    for (int N : sizes) {
        long total = (long)N * N * N;
        float data_MB = (float)total * sizeof(float) / (1024.0f * 1024.0f);

        std::vector<float> h_data(total);
        generateRadialSinc(h_data.data(), N, N, N, k, omega, t);

        // --- GPU compress/decompress + timing ---
        hipCompressPlan* plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

        float* d_input = nullptr;
        float* d_output = nullptr;
        unsigned char* d_compressed = nullptr;
        HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
        HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
        size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
        HIPCHECK(hipMalloc(&d_compressed, comp_size));
        HIPCHECK(hipMemcpy(d_input, h_data.data(), total * sizeof(float), hipMemcpyHostToDevice));

        long gpu_len = 0;
        float gpu_cr = 0;

        for (int i = 0; i < warmup; ++i) {
            compressWithAutoRMS(scale, d_input, d_compressed, &gpu_len, nullptr, plan);
            hipDecompress(d_compressed, d_output, plan, 0);
        }

        hipEvent_t te0, te1, te2;
        hipEventCreate(&te0); hipEventCreate(&te1); hipEventCreate(&te2);

        hipEventRecord(te0);
        for (int i = 0; i < iters; ++i)
            compressWithAutoRMS(scale, d_input, d_compressed, &gpu_len, &gpu_cr, plan);
        hipEventRecord(te1);
        for (int i = 0; i < iters; ++i)
            hipDecompress(d_compressed, d_output, plan, 0);
        hipEventRecord(te2);
        hipEventSynchronize(te2);

        float gpu_fwd_ms = 0, gpu_inv_ms = 0;
        hipEventElapsedTime(&gpu_fwd_ms, te0, te1); gpu_fwd_ms /= iters;
        hipEventElapsedTime(&gpu_inv_ms, te1, te2); gpu_inv_ms /= iters;
        hipEventDestroy(te0); hipEventDestroy(te1); hipEventDestroy(te2);

        // GPU error: decompress one more time, copy back
        compressWithAutoRMS(scale, d_input, d_compressed, &gpu_len, &gpu_cr, plan);
        
        hipDecompress(d_compressed, d_output, plan, 0);
        std::vector<float> h_gpu_out(total);
        HIPCHECK(hipMemcpy(h_gpu_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));
        ErrorNorms gpu_err = computeErrors(h_data.data(), h_gpu_out.data(), total);

        hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
        hipCompressDestroyPlan(plan);

        // --- CPU compress/decompress + timing (sweep thread counts) ---
        CvxCompress compressor;
        std::vector<float> h_cpu(h_data);
        unsigned int* cpu_compressed = nullptr;
        posix_memalign((void**)&cpu_compressed, 64, total * 5);
        long cpu_len = 0;

        float cpu_fwd_ms = 1e30f, cpu_inv_ms = 1e30f;
        float cpu_cr = 0;
        int best_fwd_threads = 1, best_inv_threads = 1;

        std::vector<int> thread_sweep;
        for (int t = 1; t <= max_threads; t *= 2) thread_sweep.push_back(t);
        if (thread_sweep.back() != max_threads) thread_sweep.push_back(max_threads);

        for (int nt : thread_sweep) {
            // warmup
            std::copy(h_data.begin(), h_data.end(), h_cpu.begin());
            compressor.Compress(scale, h_cpu.data(), N, N, N, bx, by, bz, cpu_compressed, nt, cpu_len);

            double fwd_total = 0, inv_total = 0;
            float cr_tmp = 0;
            for (int i = 0; i < iters; ++i) {
                std::copy(h_data.begin(), h_data.end(), h_cpu.begin());
                double wt0 = omp_get_wtime();
                cr_tmp = compressor.Compress(scale, h_cpu.data(), N, N, N, bx, by, bz, cpu_compressed, nt, cpu_len);
                double wt1 = omp_get_wtime();
                compressor.Decompress(h_cpu.data(), N, N, N, cpu_compressed, nt, cpu_len);
                double wt2 = omp_get_wtime();
                fwd_total += (wt1 - wt0);
                inv_total += (wt2 - wt1);
            }
            float fwd = (float)(fwd_total / iters * 1000.0);
            float inv = (float)(inv_total / iters * 1000.0);
            printf("  N=%d  threads=%3d  fwd=%.2f ms  inv=%.2f ms\n", N, nt, fwd, inv);
            if (fwd < cpu_fwd_ms) { cpu_fwd_ms = fwd; best_fwd_threads = nt; cpu_cr = cr_tmp; }
            if (inv < cpu_inv_ms) { cpu_inv_ms = inv; best_inv_threads = nt; }
        }

        // CPU error: run once more at best thread count
        std::copy(h_data.begin(), h_data.end(), h_cpu.begin());
        compressor.Compress(scale, h_cpu.data(), N, N, N, bx, by, bz, cpu_compressed, best_fwd_threads, cpu_len);
        compressor.Decompress(h_cpu.data(), N, N, N, cpu_compressed, best_inv_threads, cpu_len);
        ErrorNorms cpu_err = computeErrors(h_data.data(), h_cpu.data(), total);

        free(cpu_compressed);

        Result r;
        r.N = N; r.data_MB = data_MB;
        r.cpu_cr = cpu_cr; r.gpu_cr = gpu_cr;
        r.cr_overhead = (cpu_cr > 0) ? (cpu_cr - gpu_cr) / cpu_cr * 100.0f : 0.0f;
        r.gpu_fwd_ms = gpu_fwd_ms; r.gpu_inv_ms = gpu_inv_ms;
        r.cpu_fwd_ms = cpu_fwd_ms; r.cpu_inv_ms = cpu_inv_ms;
        r.fwd_speedup = (gpu_fwd_ms > 0) ? cpu_fwd_ms / gpu_fwd_ms : 0.0f;
        r.inv_speedup = (gpu_inv_ms > 0) ? cpu_inv_ms / gpu_inv_ms : 0.0f;
        r.cpu_fwd_threads = best_fwd_threads;
        r.cpu_inv_threads = best_inv_threads;
        r.gpu_err = gpu_err; r.cpu_err = cpu_err;
        results.push_back(r);
    }

    // --- Table 1: CR and throughput ---
    printf("Table 1: Compression ratio and throughput (CPU = best of 1..%d threads)\n", max_threads);
    printf("  %6s  %7s  %7s  %7s  %7s  %9s  %9s  %9s  %4s  %9s  %4s  %7s  %7s\n",
           "N", "MB", "CPU_CR", "GPU_CR", "CR_oh%",
           "GPU_fwd", "GPU_inv", "CPU_fwd", "thr", "CPU_inv", "thr",
           "fwd_su", "inv_su");
    printf("  %6s  %7s  %7s  %7s  %7s  %9s  %9s  %9s  %4s  %9s  %4s  %7s  %7s\n",
           "", "", "", "", "",
           "ms", "ms", "ms", "", "ms", "", "x", "x");
    for (auto& r : results) {
        printf("  %6d  %7.1f  %7.1f  %7.1f  %6.1f%%  %8.3f  %8.3f  %8.3f  %4d  %8.3f  %4d  %6.1fx  %6.1fx\n",
               r.N, r.data_MB, r.cpu_cr, r.gpu_cr, r.cr_overhead,
               r.gpu_fwd_ms, r.gpu_inv_ms,
               r.cpu_fwd_ms, r.cpu_fwd_threads,
               r.cpu_inv_ms, r.cpu_inv_threads,
               r.fwd_speedup, r.inv_speedup);
    }

    // --- Table 2: Error norms ---
    printf("\nTable 2: Error norms (GPU vs CPU)\n");
    printf("  %6s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
           "N", "GPU_L1", "CPU_L1", "GPU_L2", "CPU_L2",
           "GPU_Linf", "CPU_Linf", "GPU_L2%%", "CPU_L2%%");
    for (auto& r : results) {
        float gpu_l2_pct = (r.gpu_err.ref_rms > 0) ? r.gpu_err.l2 / r.gpu_err.ref_rms * 100.0 : 0;
        float cpu_l2_pct = (r.cpu_err.ref_rms > 0) ? r.cpu_err.l2 / r.cpu_err.ref_rms * 100.0 : 0;
        printf("  %6d  %10.2e  %10.2e  %10.2e  %10.2e  %10.2e  %10.2e  %9.2f%%  %9.2f%%\n",
               r.N,
               r.gpu_err.l1, r.cpu_err.l1,
               r.gpu_err.l2, r.cpu_err.l2,
               r.gpu_err.linf, r.cpu_err.linf,
               gpu_l2_pct, cpu_l2_pct);
    }
}

static bool test_pipeline_overlap()
{
    printf("Test 31: Pipeline overlap (user_stream free while aux compacts)\n");
    const int N = 256, total = N * N * N;
    const float scale = 5e-2f;

    hipStream_t stream, aux;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK(hipStreamCreateWithFlags(&aux, hipStreamNonBlocking));
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, aux));

    float *d_input, *d_output;
    unsigned char* d_compressed;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads, 0, stream>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipStreamSynchronize(stream));

    // RMS + compress (async — no Synchronize yet)
    HIPCHECK(deviceRMS(d_input, N, N, N, plan->d_rms, plan, stream));
    HIPCHECK(hipCompress(scale, plan->d_rms, d_input, d_compressed, plan, stream));

    // user_stream is free — enqueue work immediately
    hipEvent_t user_done;
    HIPCHECK(hipEventCreateWithFlags(&user_done, hipEventDisableTiming));
    initSinKernel<<<blocks, threads, 0, stream>>>(d_input, N, N, N, 30.0f, 30.0f, 30.0f);
    HIPCHECK(hipEventRecord(user_done, stream));
    HIPCHECK(hipEventSynchronize(user_done));

    // Now collect the compress result
    long len = 0;
    HIPCHECK(hipCompressSynchronize(plan, &len, nullptr));

    // Decompress and verify correctness
    initSinKernel<<<blocks, threads, 0, stream>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);
    HIPCHECK(hipDecompress(d_compressed, d_output, plan, stream));
    HIPCHECK(hipStreamSynchronize(stream));

    std::vector<float> h_in(total), h_out(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), total);
    float max_err = maxAbsError(h_in.data(), h_out.data(), total);
    bool pass = (max_err < rms) && (len > 0);
    printf("  len=%ld, err=%.4e: %s\n", len, max_err, pass ? "PASS" : "FAIL");

    hipEventDestroy(user_done);
    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    hipStreamDestroy(aux);
    hipStreamDestroy(stream);
    return pass;
}

static bool test_different_user_stream_per_call()
{
    printf("Test 32: Same plan, different user_stream per call\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipStream_t sA, sB, aux;
    HIPCHECK(hipStreamCreate(&sA));
    HIPCHECK(hipStreamCreate(&sB));
    HIPCHECK(hipStreamCreateWithFlags(&aux, hipStreamNonBlocking));
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, aux));

    float *d_input, *d_output;
    unsigned char* d_compressed;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    // Init and compress on stream A
    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads, 0, sA>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);

    long len = 0;
    float cr = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len, &cr, plan, sA));
    HIPCHECK(hipStreamSynchronize(sA));

    // Decompress on stream B (different stream, same plan)
    hipEvent_t bridge;
    HIPCHECK(hipEventCreateWithFlags(&bridge, hipEventDisableTiming));
    HIPCHECK(hipEventRecord(bridge, sA));
    HIPCHECK(hipStreamWaitEvent(sB, bridge, 0));
    HIPCHECK(hipDecompress(d_compressed, d_output, plan, sB));
    HIPCHECK(hipStreamSynchronize(sB));

    std::vector<float> h_in(total), h_out(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), total);
    float max_err = maxAbsError(h_in.data(), h_out.data(), total);
    bool pass = (max_err < rms) && (cr > 1.0f);
    printf("  CR=%.2f, err=%.4e (compress on sA, decompress on sB): %s\n",
           cr, max_err, pass ? "PASS" : "FAIL");

    hipEventDestroy(bridge);
    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    hipStreamDestroy(aux);
    hipStreamDestroy(sA); hipStreamDestroy(sB);
    return pass;
}

static bool test_user_stream_equals_aux_stream()
{
    printf("Test 33: user_stream == aux_stream (degenerate serialized)\n");
    const int N = 128, total = N * N * N;
    const float scale = 5e-2f;

    hipStream_t shared;
    HIPCHECK(hipStreamCreate(&shared));
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, shared));

    float *d_input, *d_output;
    unsigned char* d_compressed;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSinKernel<<<blocks, threads, 0, shared>>>(d_input, N, N, N, 20.0f, 20.0f, 20.0f);

    long len = 0;
    float cr = 0;
    HIPCHECK(compressWithAutoRMS(scale, d_input, d_compressed, &len, &cr, plan, shared));
    HIPCHECK(hipDecompress(d_compressed, d_output, plan, shared));
    HIPCHECK(hipStreamSynchronize(shared));

    std::vector<float> h_in(total), h_out(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), total);
    float max_err = maxAbsError(h_in.data(), h_out.data(), total);
    bool pass = (max_err < rms) && (cr > 1.0f);
    printf("  CR=%.2f, err=%.4e: %s\n", cr, max_err, pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    hipStreamDestroy(shared);
    return pass;
}

static bool test_full_pipeline_nondefault_stream()
{
    printf("Test 34: Full pipeline (CopyTo->Compress->Decompress->CopyFrom) non-default stream\n");
    const int wx = 100, wy = 100, wz = 60;
    int wnx, wny, wnz;
    hipCompressWaveletDims(wx, wy, wz, &wnx, &wny, &wnz);
    const long src_total = (long)wx * wy * wz;
    const long wav_total = (long)wnx * wny * wnz;
    const float scale = 5e-2f;

    hipStream_t stream, aux;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK(hipStreamCreateWithFlags(&aux, hipStreamNonBlocking));
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, wnx, wny, wnz, aux));

    float *d_src, *d_wav, *d_wav_out, *d_dst;
    unsigned char* d_comp;
    HIPCHECK(hipMalloc(&d_src, src_total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_wav, wav_total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_wav_out, wav_total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_dst, src_total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_comp, comp_size));

    std::vector<float> h_src(src_total);
    for (long i = 0; i < src_total; ++i)
        h_src[i] = sinf((float)i * 0.01f) * 10.0f;
    HIPCHECK(hipMemcpyAsync(d_src, h_src.data(), src_total * sizeof(float),
                            hipMemcpyHostToDevice, stream));

    HIPCHECK(hipCopyToWaveletLayout(
        d_src, wx, wx * wy,
        0, 0, 0, wx, wy, wz,
        d_wav, plan->d_rms, plan, stream));

    long len = 0;
    float cr = 0;
    HIPCHECK(hipCompress(scale, plan->d_rms, d_wav, d_comp, plan, stream));
    HIPCHECK(hipCompressSynchronize(plan, &len, &cr));
    HIPCHECK(hipDecompress(d_comp, d_wav_out, plan, stream));

    HIPCHECK(hipCopyFromWaveletLayout(
        d_wav_out, d_dst, wx, wx * wy,
        0, 0, 0, wx, wy, wz, plan, stream));
    HIPCHECK(hipStreamSynchronize(stream));

    std::vector<float> h_dst(src_total);
    HIPCHECK(hipMemcpy(h_dst.data(), d_dst, src_total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_src.data(), (int)src_total);
    float max_err = maxAbsError(h_src.data(), h_dst.data(), (int)src_total);
    float bound = 10.0f * rms * scale;
    bool pass = (max_err < bound) && (cr > 1.0f);
    printf("  %dx%dx%d -> %dx%dx%d: CR=%.2f err=%.4e bound=%.4e: %s\n",
           wx, wy, wz, wnx, wny, wnz, cr, max_err, bound, pass ? "PASS" : "FAIL");

    hipFree(d_src); hipFree(d_wav); hipFree(d_wav_out); hipFree(d_dst); hipFree(d_comp);
    hipCompressDestroyPlan(plan);
    hipStreamDestroy(aux);
    hipStreamDestroy(stream);
    return pass;
}

static void bench_copy_kernels()
{
    printf("Benchmark: Copy kernel throughput\n");
    printf("  %20s  %7s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n",
           "grid", "MB",
           "copy(ms)", "copy GB/s",
           "c+r(ms)", "c+r GB/s",
           "rms(ms)", "rms GB/s",
           "from(ms)", "from GB/s");

    struct { int wx, wy, wz; const char* tag; } cases[] = {
        {512, 512, 512, "512^3"},
        {352, 416, 320, "352x416x320"},
        {256, 256, 256, "256^3"},
        {256, 256, 512, "256x256x512"},
        {128, 128, 128, "128^3"},
        { 50,  45,  37, "50x45x37 (pad)"},
        {100, 200,  60, "100x200x60 (pad)"},
    };

    const int warmup = 5, iters = 20;

    for (auto& c : cases) {
        int wnx, wny, wnz;
        hipCompressWaveletDims(c.wx, c.wy, c.wz, &wnx, &wny, &wnz);
        long src_total = (long)c.wx * c.wy * c.wz;
        long wav_total = (long)wnx * wny * wnz;
        float rw_bytes = (float)(wav_total + wav_total) * sizeof(float);
        float rw_MB = rw_bytes / (1024.0f * 1024.0f);
        float rd_bytes = (float)wav_total * sizeof(float);
        float rd_MB = rd_bytes / (1024.0f * 1024.0f);
        float from_bytes = (float)(wav_total + src_total) * sizeof(float);
        float from_MB = from_bytes / (1024.0f * 1024.0f);

        float *d_src, *d_wav, *d_dst;
        HIPCHECK(hipMalloc(&d_src, src_total * sizeof(float)));
        HIPCHECK(hipMalloc(&d_wav, wav_total * sizeof(float)));
        HIPCHECK(hipMalloc(&d_dst, src_total * sizeof(float)));
        HIPCHECK(hipMemset(d_src, 0x42, src_total * sizeof(float)));

        hipCompressPlan* plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&plan, wnx, wny, wnz, 0));

        for (int i = 0; i < warmup; ++i) {
            hipCopyToWaveletLayout(d_src, c.wx, c.wx * c.wy,
                0, 0, 0, c.wx, c.wy, c.wz,
                d_wav, nullptr, plan, 0);
            hipCopyToWaveletLayout(d_src, c.wx, c.wx * c.wy,
                0, 0, 0, c.wx, c.wy, c.wz,
                d_wav, plan->d_rms, plan, 0);
            hipCopyToWaveletLayout(d_src, c.wx, c.wx * c.wy,
                0, 0, 0, c.wx, c.wy, c.wz,
                nullptr, plan->d_rms, plan, 0);
            hipCopyFromWaveletLayout(d_wav,
                d_dst, c.wx, c.wx * c.wy, 0, 0, 0, c.wx, c.wy, c.wz, plan, 0);
        }

        hipEvent_t t0, t1, t2, t3, t4;
        hipEventCreate(&t0); hipEventCreate(&t1); hipEventCreate(&t2);
        hipEventCreate(&t3); hipEventCreate(&t4);

        hipEventRecord(t0);
        for (int i = 0; i < iters; ++i)
            hipCopyToWaveletLayout(d_src, c.wx, c.wx * c.wy,
                0, 0, 0, c.wx, c.wy, c.wz,
                d_wav, nullptr, plan, 0);
        hipEventRecord(t1);
        for (int i = 0; i < iters; ++i)
            hipCopyToWaveletLayout(d_src, c.wx, c.wx * c.wy,
                0, 0, 0, c.wx, c.wy, c.wz,
                d_wav, plan->d_rms, plan, 0);
        hipEventRecord(t2);
        for (int i = 0; i < iters; ++i)
            hipCopyToWaveletLayout(d_src, c.wx, c.wx * c.wy,
                0, 0, 0, c.wx, c.wy, c.wz,
                nullptr, plan->d_rms, plan, 0);
        hipEventRecord(t3);
        for (int i = 0; i < iters; ++i)
            hipCopyFromWaveletLayout(d_wav,
                d_dst, c.wx, c.wx * c.wy, 0, 0, 0, c.wx, c.wy, c.wz, plan, 0);
        hipEventRecord(t4);
        hipEventSynchronize(t4);

        float copy_ms = 0, copyrms_ms = 0, rms_ms = 0, from_ms = 0;
        hipEventElapsedTime(&copy_ms, t0, t1); copy_ms /= iters;
        hipEventElapsedTime(&copyrms_ms, t1, t2); copyrms_ms /= iters;
        hipEventElapsedTime(&rms_ms, t2, t3); rms_ms /= iters;
        hipEventElapsedTime(&from_ms, t3, t4); from_ms /= iters;

        float copy_bw = rw_MB / copy_ms * 1000.0f / 1024.0f;
        float copyrms_bw = rw_MB / copyrms_ms * 1000.0f / 1024.0f;
        float rms_bw = rd_MB / rms_ms * 1000.0f / 1024.0f;
        float from_bw = from_MB / from_ms * 1000.0f / 1024.0f;

        printf("  %20s  %7.1f  %8.3f  %7.1f  %8.3f  %7.1f  %8.3f  %7.1f  %8.3f  %7.1f\n",
               c.tag, rw_MB,
               copy_ms, copy_bw, copyrms_ms, copyrms_bw,
               rms_ms, rms_bw, from_ms, from_bw);

        hipEventDestroy(t0); hipEventDestroy(t1); hipEventDestroy(t2);
        hipEventDestroy(t3); hipEventDestroy(t4);
        hipFree(d_src); hipFree(d_wav); hipFree(d_dst);
        hipCompressDestroyPlan(plan);
    }
}

static bool check_error(const char* label, hipCompressPlan* plan,
                        hipError_t got_hip, hipError_t expected_hip,
                        hipCompressError_t expected_lib)
{
    if (got_hip != expected_hip) {
        printf("  FAIL [%s]: hipError_t=%d expected=%d\n", label, (int)got_hip, (int)expected_hip);
        return false;
    }
    hipCompressError_t lib_err = hipCompressGetLastError(plan);
    if (lib_err != expected_lib) {
        printf("  FAIL [%s]: hipCompressError_t=%d expected=%d (%s)\n",
               label, (int)lib_err, (int)expected_lib, hipCompressErrorString(lib_err));
        return false;
    }
    printf("  ok [%s]: %s\n", label, hipCompressErrorString(expected_lib));
    return true;
}

static bool test_error_codes()
{
    printf("Test 35: Error codes for all validation paths\n");
    bool pass = true;

    // --- CreatePlan errors ---
    {
        hipCompressPlan* plan = nullptr;
        hipError_t err = hipCompressCreatePlan(&plan, -1, 128, 128, 0);
        if (!check_error("CreatePlan negative dim", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_INVALID_DIMENSIONS))
            pass = false;
        if (plan) hipCompressDestroyPlan(plan);
    }
    {
        hipCompressPlan* plan = nullptr;
        hipError_t err = hipCompressCreatePlan(&plan, 100, 128, 128, 0);
        if (!check_error("CreatePlan not mult 32", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_NOT_MULTIPLE_OF_32))
            pass = false;
        if (plan) hipCompressDestroyPlan(plan);
    }
    {
        hipCompressPlan* plan = nullptr;
        hipError_t err = hipCompressCreatePlan(&plan, 32800, 32768, 32, 0);
        if (!check_error("CreatePlan plane too large", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_PLANE_TOO_LARGE))
            pass = false;
        if (plan) hipCompressDestroyPlan(plan);
    }
    {
        hipError_t err = hipCompressCreatePlan(nullptr, 128, 128, 128, 0);
        if (err != hipErrorInvalidValue) {
            printf("  FAIL [CreatePlan null plan**]: hipError_t=%d\n", (int)err);
            pass = false;
        } else {
            printf("  ok [CreatePlan null plan**]\n");
        }
    }
    {
        hipCompressPlan* plan = nullptr;
        hipError_t err = hipCompressCreatePlan(&plan, 128, 128, 128, 0);
        if (err != hipSuccess || !plan) {
            printf("  FAIL [CreatePlan success]: could not create plan\n");
            pass = false;
        } else {
            if (!check_error("CreatePlan success", plan, err,
                              hipSuccess, HIP_COMPRESS_SUCCESS))
                pass = false;
            hipCompressDestroyPlan(plan);
        }
    }

    // --- DestroyPlan null ---
    {
        hipError_t err = hipCompressDestroyPlan(nullptr);
        if (err != hipErrorInvalidValue) {
            printf("  FAIL [DestroyPlan null]\n");
            pass = false;
        } else {
            printf("  ok [DestroyPlan null]\n");
        }
    }

    // --- Null plan for all other functions ---
    {
        hipError_t err = hipCompress(5e-2f, nullptr, nullptr, nullptr, nullptr, 0);
        if (err != hipErrorInvalidValue) { printf("  FAIL [Compress null plan]\n"); pass = false; }
        else printf("  ok [Compress null plan]\n");
    }
    {
        hipError_t err = hipCompressSynchronize(nullptr, nullptr, nullptr);
        if (err != hipErrorInvalidValue) { printf("  FAIL [Synchronize null plan]\n"); pass = false; }
        else printf("  ok [Synchronize null plan]\n");
    }
    {
        hipError_t err = hipCopyToWaveletLayout(nullptr, 0, 0,
            0, 0, 0,
            0, 0, 0,
            nullptr, nullptr, nullptr, 0);
        if (err != hipErrorInvalidValue) { printf("  FAIL [CopyTo null plan]\n"); pass = false; }
        else printf("  ok [CopyTo null plan]\n");
    }
    {
        hipError_t err = hipCopyFromWaveletLayout(nullptr,
            nullptr, 0, 0, 0, 0, 0, 0, 0, 0, nullptr, 0);
        if (err != hipErrorInvalidValue) { printf("  FAIL [CopyFrom null plan]\n"); pass = false; }
        else printf("  ok [CopyFrom null plan]\n");
    }
    {
        hipError_t err = hipDecompress(nullptr, nullptr, nullptr, 0);
        if (err != hipErrorInvalidValue) { printf("  FAIL [Decompress null plan]\n"); pass = false; }
        else printf("  ok [Decompress null plan]\n");
    }
    {
        size_t sz = 0;
        hipError_t err = hipCompressMaxOutputSize(nullptr, &sz);
        if (err != hipErrorInvalidValue) { printf("  FAIL [MaxOutputSize null plan]\n"); pass = false; }
        else printf("  ok [MaxOutputSize null plan]\n");
    }

    // Allocate a valid plan for the remaining tests
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, 128, 128, 128, 0));
    int total = 128 * 128 * 128;
    float* d_buf;
    unsigned char* d_comp;
    HIPCHECK(hipMalloc(&d_buf, total * sizeof(float)));
    HIPCHECK(hipMemset(d_buf, 0, total * sizeof(float)));
    size_t cs = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &cs));
    HIPCHECK(hipMalloc(&d_comp, cs));

    // --- Compress errors ---
    {
        hipError_t err = hipCompress(5e-2f, nullptr, nullptr, d_comp, plan, 0);
        if (!check_error("Compress null input", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_NULL_INPUT))
            pass = false;
    }
    {
        hipError_t err = hipCompress(5e-2f, nullptr, (const float*)d_buf, nullptr, plan, 0);
        if (!check_error("Compress null output", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_NULL_OUTPUT))
            pass = false;
    }
    {
        hipError_t err = hipCompress(-1.0f, nullptr, (const float*)d_buf, d_comp, plan, 0);
        if (!check_error("Compress negative scale", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_INVALID_SCALE))
            pass = false;
    }
    {
        hipError_t err = hipCompress(INFINITY, nullptr, (const float*)d_buf, d_comp, plan, 0);
        if (!check_error("Compress infinite scale", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_INVALID_SCALE))
            pass = false;
    }
    {
        hipError_t err = hipCompress(NAN, nullptr, (const float*)d_buf, d_comp, plan, 0);
        if (!check_error("Compress NaN scale", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_INVALID_SCALE))
            pass = false;
    }
    {
        // Force compress_pending
        HIPCHECK(hipCompress(5e-2f, nullptr, (const float*)d_buf, d_comp, plan, 0));
        hipError_t err = hipCompress(5e-2f, nullptr, (const float*)d_buf, d_comp, plan, 0);
        if (!check_error("Compress while pending", plan, err,
                          hipErrorNotReady, HIP_COMPRESS_ERROR_COMPRESS_PENDING))
            pass = false;
        HIPCHECK(hipCompressSynchronize(plan, nullptr, nullptr));
    }

    // --- Synchronize errors ---
    {
        hipError_t err = hipCompressSynchronize(plan, nullptr, nullptr);
        if (!check_error("Synchronize no pending", plan, err,
                          hipErrorNotReady, HIP_COMPRESS_ERROR_NO_COMPRESS_PENDING))
            pass = false;
    }

    // --- CopyToWaveletLayout errors ---
    {
        hipError_t err = hipCopyToWaveletLayout(
            nullptr, 128, 128*128,
            0, 0, 0,
            128, 128, 128,
            d_buf, plan->d_rms, plan, 0);
        if (!check_error("CopyTo null src", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_NULL_INPUT))
            pass = false;
    }
    {
        hipError_t err = hipCopyToWaveletLayout(
            d_buf, 128, 128*128,
            0, 0, 0,
            128, 128, 128,
            nullptr, nullptr, plan, 0);
        if (!check_error("CopyTo both outputs null", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_BOTH_OUTPUTS_NULL))
            pass = false;
    }
    {
        hipError_t err = hipCopyToWaveletLayout(
            d_buf, 128, 128*128,
            0, 0, 0,
            16, 128, 128,
            d_buf, nullptr, plan, 0);
        if (!check_error("CopyTo window too small", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_WINDOW_TOO_SMALL))
            pass = false;
    }
    {
        hipCompressPlan* small_plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&small_plan, 64, 64, 64, 0));
        hipError_t err = hipCopyToWaveletLayout(
            d_buf, 128, 128*128,
            0, 0, 0,
            65, 65, 65,
            d_buf, nullptr, small_plan, 0);
        if (!check_error("CopyTo extraction exceeds plan", small_plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_EXTRACTION_DIMS_MISMATCH))
            pass = false;
        hipCompressDestroyPlan(small_plan);
    }
    {
        hipError_t err = hipCopyToWaveletLayout(
            d_buf, 128, 128*128,
            0, 0, 0,
            50, 50, 50,
            d_buf, nullptr, plan, 0);
        if (!check_error("CopyTo extraction smaller than plan", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_EXTRACTION_DIMS_MISMATCH))
            pass = false;
    }

    // --- CopyFromWaveletLayout errors ---
    {
        hipError_t err = hipCopyFromWaveletLayout(
            nullptr, d_buf, 128, 128*128,
            0, 0, 0, 128, 128, 128, plan, 0);
        if (!check_error("CopyFrom null src", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_NULL_INPUT))
            pass = false;
    }
    {
        hipError_t err = hipCopyFromWaveletLayout(
            d_buf, nullptr, 128, 128*128,
            0, 0, 0, 128, 128, 128, plan, 0);
        if (!check_error("CopyFrom null dst", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_NULL_OUTPUT))
            pass = false;
    }
    {
        hipError_t err = hipCopyFromWaveletLayout(
            d_buf, d_buf, 128, 128*128,
            0, 0, 0, 16, 128, 128, plan, 0);
        if (!check_error("CopyFrom window too small", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_WINDOW_TOO_SMALL))
            pass = false;
    }
    {
        hipCompressPlan* small_plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&small_plan, 64, 64, 64, 0));
        hipError_t err = hipCopyFromWaveletLayout(
            d_buf, d_buf, 128, 128*128,
            0, 0, 0, 65, 65, 65, small_plan, 0);
        if (!check_error("CopyFrom extraction exceeds plan", small_plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_EXTRACTION_DIMS_MISMATCH))
            pass = false;
        hipCompressDestroyPlan(small_plan);
    }
    {
        hipError_t err = hipCopyFromWaveletLayout(
            d_buf, d_buf, 128, 128*128,
            0, 0, 0, 50, 50, 50, plan, 0);
        if (!check_error("CopyFrom extraction smaller than plan", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_EXTRACTION_DIMS_MISMATCH))
            pass = false;
    }

    // --- Plane too large ---
    {
        hipError_t err = hipCopyToWaveletLayout(
            d_buf, 32768, 1073741825,
            0, 0, 0,
            128, 128, 128,
            d_buf, nullptr, plan, 0);
        if (!check_error("CopyTo plane too large", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_PLANE_TOO_LARGE))
            pass = false;
    }
    {
        hipError_t err = hipCopyFromWaveletLayout(
            d_buf, d_buf, 32768, 1073741825,
            0, 0, 0, 128, 128, 128, plan, 0);
        if (!check_error("CopyFrom plane too large", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_PLANE_TOO_LARGE))
            pass = false;
    }

    // --- Decompress errors ---
    {
        hipError_t err = hipDecompress(nullptr, d_buf, plan, 0);
        if (!check_error("Decompress null input", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_NULL_INPUT))
            pass = false;
    }
    {
        hipError_t err = hipDecompress(d_comp, nullptr, plan, 0);
        if (!check_error("Decompress null output", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_NULL_OUTPUT))
            pass = false;
    }

    // --- MaxOutputSize errors ---
    {
        hipError_t err = hipCompressMaxOutputSize(plan, nullptr);
        if (!check_error("MaxOutputSize null size", plan, err,
                          hipErrorInvalidValue, HIP_COMPRESS_ERROR_NULL_OUTPUT))
            pass = false;
    }

    // --- GetLastError / ErrorString ---
    {
        hipCompressError_t e = hipCompressGetLastError(nullptr);
        if (e != HIP_COMPRESS_ERROR_NULL_PLAN) {
            printf("  FAIL [GetLastError null plan]\n");
            pass = false;
        } else {
            printf("  ok [GetLastError null plan]\n");
        }
    }
    {
        const char* s = hipCompressErrorString(HIP_COMPRESS_SUCCESS);
        if (!s || s[0] == '\0') {
            printf("  FAIL [ErrorString success]\n");
            pass = false;
        } else {
            printf("  ok [ErrorString success]: \"%s\"\n", s);
        }
    }
    {
        const char* s = hipCompressErrorString((hipCompressError_t)9999);
        if (!s || s[0] == '\0') {
            printf("  FAIL [ErrorString unknown]\n");
            pass = false;
        } else {
            printf("  ok [ErrorString unknown]: \"%s\"\n", s);
        }
    }

    // --- Success clears error ---
    {
        // Trigger an error first
        hipCompress(5e-2f, nullptr, nullptr, d_comp, plan, 0);
        if (hipCompressGetLastError(plan) != HIP_COMPRESS_ERROR_NULL_INPUT) {
            printf("  FAIL [error set before clear test]\n");
            pass = false;
        }
        // Successful call should clear it
        size_t sz = 0;
        HIPCHECK(hipCompressMaxOutputSize(plan, &sz));
        if (!check_error("success clears error", plan,
                          hipSuccess, hipSuccess, HIP_COMPRESS_SUCCESS))
            pass = false;
    }

    hipFree(d_buf); hipFree(d_comp);
    hipCompressDestroyPlan(plan);

    printf("  error codes: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ---------------------------------------------------------------------------
// Test 37: Demonstrate soffset overflow with large stride + ez0.
//
// Allocates ~5.3 GB source buffer with large ldimxy and ez0 so that the
// z-plane byte offset (ez0 + iz_src) * ldimxy * 4 exceeds UINT32_MAX.
// Writes sentinel values at correct source locations.  If the kernel reads
// from the right addresses the wavelet output contains sentinels; if soffset
// wraps it reads zeros → detectable mismatch.
// ---------------------------------------------------------------------------
bool test_soffset_overflow()
{
    printf("Test 37: soffset overflow with large stride + ez0\n");

    const int ldimx  = 10000;
    const int ldimy  = 1000;
    const int ldimxy = ldimx * ldimy;          // 10 M elements per plane
    const int ez0    = 100;
    const int ex = 32, ey = 32, ez = 32;
    const int wnx = 32, wny = 32, wnz = 32;

    // Verify this configuration actually overflows uint32 soffset
    long max_z_byte = (long)(ez0 + ez - 1) * ldimxy * sizeof(float);
    printf("  max z-plane byte offset: %ld (%.2f GB, uint32 max = %u)\n",
           max_z_byte, max_z_byte / 1.0e9, (unsigned)0xFFFFFFFF);
    if (max_z_byte <= (long)0xFFFFFFFF) {
        printf("  SKIP: config does not overflow uint32 soffset\n");
        return true;
    }

    // Total source allocation: (ez0 + ez) planes
    long src_elems = (long)(ez0 + ez) * ldimxy;
    long src_bytes = src_elems * sizeof(float);
    printf("  source allocation: %ld elements (%.2f GB)\n", src_elems, src_bytes / 1.0e9);

    float* d_src = nullptr;
    hipError_t alloc_err = hipMalloc(&d_src, src_bytes);
    if (alloc_err != hipSuccess) {
        printf("  SKIP: cannot allocate %.2f GB (%s)\n",
               src_bytes / 1.0e9, hipGetErrorString(alloc_err));
        return true;
    }
    HIPCHECK(hipMemset(d_src, 0, src_bytes));

    // Write sentinels: value = iz + 1 at the first element of each
    // extraction row (ex0=0, ey0=0) for each z-plane in [ez0, ez0+ez).
    std::vector<float> h_sentinels(ez);
    for (int iz = 0; iz < ez; ++iz) {
        h_sentinels[iz] = (float)(iz + 1);
        long offset = (long)(ez0 + iz) * ldimxy;  // first element of plane
        HIPCHECK(hipMemcpy(d_src + offset, &h_sentinels[iz],
                           sizeof(float), hipMemcpyHostToDevice));
    }

    // Wavelet output buffer
    long wav_elems = (long)wnx * wny * wnz;
    float* d_wav = nullptr;
    HIPCHECK(hipMalloc(&d_wav, wav_elems * sizeof(float)));
    HIPCHECK(hipMemset(d_wav, 0, wav_elems * sizeof(float)));

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, wnx, wny, wnz, 0));

    HIPCHECK(hipCopyToWaveletLayout(
        d_src, ldimx, ldimxy,
        0, 0, ez0,
        ex, ey, ez,
        d_wav, nullptr, plan, 0));
    HIPCHECK(hipDeviceSynchronize());

    // Read back wavelet buffer and check sentinels at position (0,0,iz)
    std::vector<float> h_wav(wav_elems);
    HIPCHECK(hipMemcpy(h_wav.data(), d_wav, wav_elems * sizeof(float),
                       hipMemcpyDeviceToHost));

    bool pass = true;
    int mismatches = 0;
    for (int iz = 0; iz < ez; ++iz) {
        long idx = (long)iz * wnx * wny;  // element (0,0,iz) in wavelet layout
        float expected = (float)(iz + 1);
        float got = h_wav[idx];
        if (fabsf(got - expected) > 1e-6f) {
            if (mismatches < 4)
                printf("  MISMATCH iz=%d: expected=%.1f got=%.1f\n", iz, expected, got);
            ++mismatches;
            pass = false;
        }
    }

    if (mismatches > 0)
        printf("  %d/%d z-planes read wrong data (soffset overflow)\n", mismatches, ez);
    else
        printf("  all %d sentinels correct\n", ez);

    hipFree(d_src);
    hipFree(d_wav);
    hipCompressDestroyPlan(plan);

    printf("  soffset overflow: %s\n", pass ? "PASS" : "FAIL (overflow confirmed)");
    return pass;
}

int main(int argc, char**)
{
    printf("=== hipCompress API Tests ===\n\n");

    int passed = 0, total = 35;
    if (test_plan_lifecycle())              ++passed;
    if (test_round_trip())                  ++passed;
    if (test_cr_vs_cpu())                   ++passed;
    if (test_varying_scale())               ++passed;
    if (test_multiple_cycles())             ++passed;
    if (test_determinism())                 ++passed;
    if (test_nondefault_stream_roundtrip()) ++passed;
    if (test_back_to_back_compress())       ++passed;
    if (test_back_to_back_decompress())     ++passed;
    if (test_pipeline_ordering())           ++passed;
    if (test_two_plans_two_streams())       ++passed;
    if (test_async_decompress())            ++passed;
    if (test_stream_ordering_no_sync())     ++passed;
    if (test_compressed_length_accuracy())  ++passed;
    if (test_multithread_separate_plans())  ++passed;
    if (test_compress_pending_guard())       ++passed;
    if (test_out_of_order_decompress())     ++passed;
    if (test_zero_input())                  ++passed;
    if (test_near_zero_input())             ++passed;
    if (test_noncubic_grids())             ++passed;
    if (test_error_bound())                 ++passed;
    if (test_copy_zero_padding())            ++passed;
    if (test_copy_nonzero_origin())         ++passed;
    if (test_rms_accuracy())                ++passed;
    if (test_rms_zero_constant())           ++passed;
    if (test_copy_modes())                  ++passed;
    if (test_copy_roundtrip())              ++passed;
    if (test_copy_from_wavelet())           ++passed;
    if (test_wavelet_dims_helper())         ++passed;
    if (test_pipeline_overlap())                  ++passed;
    if (test_different_user_stream_per_call())    ++passed;
    if (test_user_stream_equals_aux_stream())     ++passed;
    if (test_full_pipeline_nondefault_stream())   ++passed;
    if (test_error_codes())                       ++passed;
    if (test_soffset_overflow())                  ++passed;

    printf("\n%d/%d TESTS PASSED\n\n", passed, total);

    if (argc < 2) {
        bench_copy_kernels();
        printf("\n");
        bench_throughput();
        printf("\n");
        bench_radial_sinc_sweep();
    }

    return (passed == total) ? 0 : 1;
}
