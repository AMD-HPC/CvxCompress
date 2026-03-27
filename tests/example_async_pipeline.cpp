// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>
#include "hipCompress.h"

#define HIPCHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Dummy simulation kernel: stencil-like read/write pattern to burn GPU time
__global__ void simulationKernel(float* data, int n, int iter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = data[idx];
    // Cheap arithmetic that depends on neighbors to prevent elimination
    int left  = (idx > 0)     ? idx - 1 : n - 1;
    int right = (idx < n - 1) ? idx + 1 : 0;
    data[idx] = 0.999f * v + 0.0005f * (data[left] + data[right])
                + 1e-6f * (float)iter;
}

// Init with sinusoidal pattern
__global__ void initKernel(float* data, int nx, int ny, int nz)
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
    data[idx] = sinf(20.0f * x) * sinf(20.0f * y) * sinf(20.0f * z);
}

int main()
{
    const int N = 256;
    const int total = N * N * N;
    const float scale = 5e-2f;
    const int num_iters = 50;
    const int compress_interval = 10;

    hipStream_t user_stream, aux;
    HIPCHECK(hipStreamCreate(&user_stream));
    HIPCHECK(hipStreamCreateWithFlags(&aux, hipStreamNonBlocking));

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, aux));

    float* d_data;
    unsigned char* d_compressed;
    HIPCHECK(hipMalloc(&d_data, total * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    initKernel<<<blocks, threads, 0, user_stream>>>(d_data, N, N, N);

    printf("Running %d iterations, compressing every %d\n", num_iters, compress_interval);

    long len = 0;
    float cr = 0;
    int compress_count = 0;

    for (int iter = 0; iter < num_iters; iter++) {
        // Simulation step on user_stream
        simulationKernel<<<blocks, threads, 0, user_stream>>>(d_data, total, iter);

        if (iter % compress_interval == 0) {
            // Drain previous compress if pending
            hipCompressSynchronize(plan, &len, &cr);
            if (compress_count > 0)
                printf("  iter %3d: collected compress %d: len=%ld CR=%.2f\n",
                       iter, compress_count, len, cr);

            // RMS on user_stream, then async compress
            HIPCHECK(hipCopyToWaveletLayout(
                d_data, N, N * N,
                0, 0, 0, N, N, N,
                nullptr, plan->d_rms, plan, user_stream));
            HIPCHECK(hipCompress(scale, plan->d_rms, d_data, d_compressed,
                                 plan, user_stream));
            compress_count++;
            // user_stream is free — next iteration's simulation proceeds immediately
        }
    }

    // Drain final compress
    if (hipCompressSynchronize(plan, &len, &cr) == hipSuccess)
        printf("  final: collected compress %d: len=%ld CR=%.2f\n",
               compress_count, len, cr);

    HIPCHECK(hipStreamSynchronize(user_stream));

    hipFree(d_data);
    hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    hipStreamDestroy(aux);
    hipStreamDestroy(user_stream);

    printf("Done. %d compresses over %d iterations.\n", compress_count, num_iters);
    return 0;
}
