// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// Unit test: fused forward encode → fused inverse decode round-trip.
// Verifies that the fused compress→decompress pipeline matches a reference
// unfused path: forward wavelet → quantize/dequant (CPU) → inverse wavelet.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "hipWaveletRLE.h"
#include "hipWaveletRLEInverse.h"
#include "hipWaveletTransformBuffer.h"
#include "quantize_rle_ref.h"

#define HIPCHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void initSinData(float* data, int NX, int NY, int NZ)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long N = (long)NX * NY * NZ;
    if (idx >= N) return;
    int ix = idx % NX;
    int iy = (idx / NX) % NY;
    int iz = idx / (NX * NY);
    float x = (float)ix / NX;
    float y = (float)iy / NY;
    float z = (float)iz / NZ;
    data[idx] = sinf(6.283f * x) * cosf(6.283f * y) * sinf(3.141f * z) * 100.0f;
}

__global__ void initRandom(float* data, long N, unsigned seed)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    unsigned h = seed ^ (unsigned)idx;
    h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
    float u = (float)(h & 0x7FFFFF) / (float)0x7FFFFF;
    float sign = (h & 0x800000) ? -1.0f : 1.0f;
    data[idx] = sign * u * u * u * 500.0f;
}

__global__ void computeRMS(const float* data, double* partial_sums, long N, int chunk_size)
{
    long start = (long)blockIdx.x * chunk_size;
    long end = start + chunk_size;
    if (end > N) end = N;
    double sum = 0.0;
    for (long i = start; i < end; ++i) sum += (double)data[i] * data[i];
    partial_sums[blockIdx.x] = sum;
}

float getGpuRMS(const float* d_data, long N)
{
    int chunk = 65536;
    int nblocks = (N + chunk - 1) / chunk;
    double* d_partial;
    HIPCHECK(hipMalloc(&d_partial, nblocks * sizeof(double)));
    computeRMS<<<nblocks, 1>>>(d_data, d_partial, N, chunk);
    HIPCHECK(hipDeviceSynchronize());
    std::vector<double> h_partial(nblocks);
    HIPCHECK(hipMemcpy(h_partial.data(), d_partial, nblocks * sizeof(double), hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(d_partial));
    double total = 0.0;
    for (auto v : h_partial) total += v;
    return (float)sqrt(total / N);
}

// Reference path: GPU forward wavelet → CPU quantize/dequantize → GPU inverse wavelet
void reference_unfused(float* d_data, float* d_out, float scale, float inv_scale,
                       int NX, int NY, int NZ, int ldimx, int ldimxy, long N)
{
    // Forward wavelet
    HIPCHECK(hipMemcpy(d_out, d_data, N * sizeof(float), hipMemcpyDeviceToDevice));
    HIPCHECK((hipWaveletTransformBufferZYX<256, 2>(d_out, NX, NY, NZ, 32, 32, 32, ldimx, ldimxy)));
    HIPCHECK(hipDeviceSynchronize());

    // CPU quantize/dequantize (truncation, same as encoder)
    std::vector<float> h_wav(N);
    HIPCHECK(hipMemcpy(h_wav.data(), d_out, N * sizeof(float), hipMemcpyDeviceToHost));
    for (long i = 0; i < N; ++i) {
        int ival = (int)(h_wav[i] * scale);
        h_wav[i] = (float)ival * inv_scale;
    }
    HIPCHECK(hipMemcpy(d_out, h_wav.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Inverse wavelet (using standalone kernel)
    dim3 grid((NX + 31) / 32, (NY + 31) / 32, (NZ + 31) / 32);
    extern __global__ void waveletInverseZYXKernel(float*, int, int);
    // We need the inverse kernel. Use the one from test_inverse_wavelet.
    // But it's in a separate TU. Instead, let's do CPU inverse.
    HIPCHECK(hipMemcpy(h_wav.data(), d_out, N * sizeof(float), hipMemcpyDeviceToHost));

    int bx = 32, by = 32, bz = 32;
    float work[32];
    int nbx = NX / bx, nby = NY / by, nbz = NZ / bz;
    for (int biz = 0; biz < nbz; ++biz) {
        for (int biy = 0; biy < nby; ++biy) {
            for (int bix = 0; bix < nbx; ++bix) {
                int x0 = bix * bx, y0 = biy * by, z0 = biz * bz;
                // Inverse X
                for (int iz = 0; iz < bz; ++iz)
                    for (int iy = 0; iy < by; ++iy) {
                        float* row = &h_wav[((iz + z0) * NY + (iy + y0)) * NX + x0];
                        us79_inverse(row, bx);
                    }
                // Inverse Y
                for (int iz = 0; iz < bz; ++iz)
                    for (int ix = 0; ix < bx; ++ix) {
                        float col[32];
                        for (int iy = 0; iy < by; ++iy)
                            col[iy] = h_wav[((iz + z0) * NY + (iy + y0)) * NX + (ix + x0)];
                        us79_inverse(col, by);
                        for (int iy = 0; iy < by; ++iy)
                            h_wav[((iz + z0) * NY + (iy + y0)) * NX + (ix + x0)] = col[iy];
                    }
                // Inverse Z
                for (int iy = 0; iy < by; ++iy)
                    for (int ix = 0; ix < bx; ++ix) {
                        float col[32];
                        for (int iz = 0; iz < bz; ++iz)
                            col[iz] = h_wav[((iz + z0) * NY + (iy + y0)) * NX + (ix + x0)];
                        us79_inverse(col, bz);
                        for (int iz = 0; iz < bz; ++iz)
                            h_wav[((iz + z0) * NY + (iy + y0)) * NX + (ix + x0)] = col[iz];
                    }
            }
        }
    }
    HIPCHECK(hipMemcpy(d_out, h_wav.data(), N * sizeof(float), hipMemcpyHostToDevice));
}

bool test_roundtrip(const char* tag, int NX, int NY, int NZ, float user_scale, bool use_sin)
{
    int ldimx = NX, ldimxy = NX * NY;
    long N = (long)NX * NY * NZ;
    int nblocks = (NX / 32) * (NY / 32) * (NZ / 32);

    float *d_input, *d_fused_out, *d_ref_out;
    unsigned char *d_compressed;
    int *d_sizes;
    HIPCHECK(hipMalloc(&d_input, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_fused_out, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_ref_out, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_compressed, (long)nblocks * 4 * WRLE_LDS_BYTES));
    HIPCHECK(hipMalloc(&d_sizes, nblocks * sizeof(int)));

    { int thr = 256, blk = (N + thr - 1) / thr;
      if (use_sin) initSinData<<<blk, thr>>>(d_input, NX, NY, NZ);
      else         initRandom<<<blk, thr>>>(d_input, N, 42);
      HIPCHECK(hipDeviceSynchronize());
    }

    float rms = getGpuRMS(d_input, N);
    float scale = 1.0f / (rms * user_scale);
    float inv_scale = rms * user_scale;

    // Path 1: Fused encode (meta embedded) → fused decode
    HIPCHECK(hipWaveletRLEFused(d_input, d_compressed, d_sizes, scale,
                                NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipWaveletRLEInverseFusedFixedStride(
        d_compressed, d_sizes, d_fused_out, inv_scale,
        NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    // Path 2: Reference unfused (GPU wavelet → CPU quant/dequant → CPU inverse wavelet)
    reference_unfused(d_input, d_ref_out, scale, inv_scale,
                      NX, NY, NZ, ldimx, ldimxy, N);

    // Compare
    std::vector<float> h_input(N), h_fused(N), h_ref(N);
    HIPCHECK(hipMemcpy(h_input.data(), d_input, N * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_fused.data(), d_fused_out, N * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_ref.data(), d_ref_out, N * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(d_input)); HIPCHECK(hipFree(d_fused_out)); HIPCHECK(hipFree(d_ref_out));
    HIPCHECK(hipFree(d_compressed)); HIPCHECK(hipFree(d_sizes));

    float max_err_input = 0.0f, max_err_ref = 0.0f;
    double sum_sq_input = 0.0, sum_sq_ref = 0.0;
    for (long i = 0; i < N; ++i) {
        float ei = fabsf(h_fused[i] - h_input[i]);
        float er = fabsf(h_fused[i] - h_ref[i]);
        if (ei > max_err_input) max_err_input = ei;
        if (er > max_err_ref) max_err_ref = er;
        sum_sq_input += (double)ei * ei;
        sum_sq_ref += (double)er * er;
    }
    float rms_input = (float)sqrt(sum_sq_input / N);
    float rms_ref = (float)sqrt(sum_sq_ref / N);

    // The fused vs ref comparison tells us if the pipeline is correct.
    // Small differences expected from FP non-determinism in wavelet.
    bool pass = max_err_ref < inv_scale * 2.0f;
    printf("  %-30s: inv_scale=%.4f  vs_input(max=%.4f rms=%.6f)  vs_ref(max=%.4f rms=%.6f)  %s\n",
           tag, inv_scale, max_err_input, rms_input, max_err_ref, rms_ref,
           pass ? "PASS" : "FAIL");

    return pass;
}

int main()
{
    printf("=== Fused Forward+Inverse Round-trip Tests ===\n\n");
    bool all_pass = true;

    for (float us : {1e-2f, 1e-3f, 1e-4f}) {
        char tag[64];
        snprintf(tag, sizeof(tag), "sin 128^3 scale=%.0e", us);
        if (!test_roundtrip(tag, 128, 128, 128, us, true)) all_pass = false;
    }
    printf("\n");
    for (float us : {1e-2f, 1e-3f, 1e-4f}) {
        char tag[64];
        snprintf(tag, sizeof(tag), "random 128^3 scale=%.0e", us);
        if (!test_roundtrip(tag, 128, 128, 128, us, false)) all_pass = false;
    }

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
