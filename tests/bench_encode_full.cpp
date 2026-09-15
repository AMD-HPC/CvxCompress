// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// Full-encode throughput: times hipCompress + hipCompressSynchronize
// end-to-end (wavelet transform, entropy coding, scan, compact, D2H readback).
// RMS is computed once outside the loop so quantization is realistic.
// Set HIP_CVX_SADDR to route the forward kernel through the saddr variant.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include "hipCompress.h"

#define HIPCHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__global__ void initKernel(float* d, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = sinf(0.001f * (float)(i % 9973));
}

int main(int argc, char** argv)
{
    int nx = 512, ny = 512, nz = 512;
    if (argc >= 4) { nx = atoi(argv[1]); ny = atoi(argv[2]); nz = atoi(argv[3]); }
    int iters = (argc >= 5) ? atoi(argv[4]) : 200;
    const float scale = 5e-2f;
    // argv[5] selects the codec (default AUTO) so the octree path can be timed
    // on its own rather than through whatever AUTO happens to pick.
    hipCompressKernel kern = (argc >= 6)
        ? (hipCompressKernel)atoi(argv[5]) : HIP_COMPRESS_KERNEL_AUTO;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, nx, ny, nz, 0, kern));

    size_t bufsz = 0, maxout = 0;
    HIPCHECK(hipCompressBufferSize(plan, &bufsz));
    HIPCHECK(hipCompressMaxOutputSize(plan, &maxout));

    float* d_in = nullptr;
    unsigned char* d_out = nullptr;
    double* d_rms = nullptr;
    HIPCHECK(hipMalloc(&d_in, bufsz));
    HIPCHECK(hipMalloc(&d_out, maxout));
    HIPCHECK(hipMalloc(&d_rms, sizeof(double)));

    long total = (long)nx * ny * nz;
    initKernel<<<(total + 255) / 256, 256>>>(d_in, total);
    HIPCHECK(hipDeviceSynchronize());

    // Realistic mulfac: compute RMS once, reuse across iterations.
    HIPCHECK(hipComputeRMS(d_in, nx, nx * ny, 0, 0, 0, nx, ny, nz, d_rms, plan, 0));
    HIPCHECK(hipDeviceSynchronize());

    long len = 0; float cr = 0;
    for (int i = 0; i < 10; ++i) {
        HIPCHECK(hipCompress(scale, d_rms, d_in, d_out, plan, 0));
        HIPCHECK(hipCompressSynchronize(plan, &len, &cr));
    }
    HIPCHECK(hipDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        HIPCHECK(hipCompress(scale, d_rms, d_in, d_out, plan, 0));
        HIPCHECK(hipCompressSynchronize(plan, &len, &cr));
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double gbps = (double)total * sizeof(float) / (ms * 1.0e-3) / 1.0e9;
    const char* mode = getenv("HIP_CVX_SADDR") ? "saddr" : "buffer";

    printf("codec=%d ", (int)kern);
    printf("mode=%-6s dims=%dx%dx%d  encode: %.4f ms  %.1f GB/s  len=%ld CR=%.2f\n",
           mode, nx, ny, nz, ms, gbps, len, cr);

    hipFree(d_in); hipFree(d_out); hipFree(d_rms);
    hipCompressDestroyPlan(plan);
    return 0;
}
