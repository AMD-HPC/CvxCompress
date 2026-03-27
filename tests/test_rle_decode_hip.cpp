// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// Unit test for GPU z-line RLE decoder.
// Tests: (1) CPU reference encode→GPU decode round-trip with integer data,
//        (2) GPU encode→GPU decode round-trip with float data (scale=1),
//        (3) GPU encode→GPU decode with quantization (scale≠1).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "hipRLEDecode.h"
#include "quantize_rle_ref.h"

// Pull in wrle_zline from the fused kernel header for encoding.
// We only use the encoder device function, not the full fused kernel.
#include "Run_Length_Escape_Codes.hxx"

#define HIPCHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Minimal GPU z-line encoder (matches wrle_zline<true,true> logic from hipWaveletRLE.h)
// Encodes 32 floats → byte stream. Returns encoded byte count.
__device__ __forceinline__
int encode_zline_gpu(const float* vals, float scale, unsigned char* dst)
{
    int bp = 0, rle = 0;
    #pragma unroll
    for (int z = 0; z < 32; ++z) {
        float fval = scale * vals[z];
        int ival = (int)fval;
        if (ival == 0) { ++rle; } else {
            if (rle == 1) {
                dst[bp] = 0; bp += 1;
            } else if (rle >= 2 && rle < 256) {
                dst[bp] = (unsigned char)(RLESC1 & 0xFF);
                dst[bp+1] = (unsigned char)rle; bp += 2;
            } else if (rle >= 256) {
                dst[bp] = (unsigned char)(RLESC3 & 0xFF);
                dst[bp+1] = (unsigned char)(rle & 0xFF);
                dst[bp+2] = (unsigned char)((rle >> 8) & 0xFF);
                dst[bp+3] = (unsigned char)((rle >> 16) & 0xFF); bp += 4;
            }
            rle = 0;
            bool ib = (ival > VLESC2) && (ival < RLESC3);
            bool i16 = (ival >= -32768) && (ival <= 32767);
            bool i24 = (ival >= -8388608) && (ival <= 8388607);
            bool f32 = !ib && !i16 && !i24;
            unsigned u; __builtin_memcpy(&u, &fval, 4);
            unsigned pay = f32 ? u : (unsigned)ival;
            if (ib) {
                dst[bp] = (unsigned char)(signed char)ival;
                bp += 1;
            } else if (i16) {
                dst[bp] = (unsigned char)(signed char)VLESC2;
                dst[bp+1] = (unsigned char)(pay & 0xFF);
                dst[bp+2] = (unsigned char)((pay >> 8) & 0xFF);
                bp += 3;
            } else if (i24) {
                dst[bp] = (unsigned char)(signed char)VLESC3;
                dst[bp+1] = (unsigned char)(pay & 0xFF);
                dst[bp+2] = (unsigned char)((pay >> 8) & 0xFF);
                dst[bp+3] = (unsigned char)((pay >> 16) & 0xFF);
                bp += 4;
            } else {
                dst[bp] = (unsigned char)(signed char)VLESC4;
                dst[bp+1] = (unsigned char)(pay & 0xFF);
                dst[bp+2] = (unsigned char)((pay >> 8) & 0xFF);
                dst[bp+3] = (unsigned char)((pay >> 16) & 0xFF);
                dst[bp+4] = (unsigned char)((pay >> 24) & 0xFF);
                bp += 5;
            }
        }
    }
    // Trailing zeros
    if (rle == 1) {
        dst[bp] = 0; bp += 1;
    } else if (rle >= 2 && rle < 256) {
        dst[bp] = (unsigned char)(RLESC1 & 0xFF);
        dst[bp+1] = (unsigned char)rle; bp += 2;
    } else if (rle >= 256) {
        dst[bp] = (unsigned char)(RLESC3 & 0xFF);
        dst[bp+1] = (unsigned char)(rle & 0xFF);
        dst[bp+2] = (unsigned char)((rle >> 8) & 0xFF);
        dst[bp+3] = (unsigned char)((rle >> 16) & 0xFF); bp += 4;
    }
    return bp;
}

// Kernel: encode N z-lines, store encoded bytes + sizes.
// Input: vals[N*32], scale. Output: encoded[N*MAX_ENC], sizes[N].
static constexpr int MAX_ENC = 32 * 5;

__global__ void kernel_encode(const float* vals, float scale,
                              unsigned char* encoded, int* sizes, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    sizes[tid] = encode_zline_gpu(vals + tid * 32, scale, encoded + tid * MAX_ENC);
}

// Kernel: decode N z-lines. Input: encoded[N*MAX_ENC], sizes[N], inv_scale.
// Output: decoded[N*32].
__global__ void kernel_decode(const unsigned char* encoded, const int* sizes,
                              float inv_scale, float* decoded, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    wrle_zline_decode(encoded + tid * MAX_ENC, sizes[tid], inv_scale, decoded + tid * 32);
}

// ---- Test 1: CPU encode → GPU decode (integer data) ----
bool test_cpu_encode_gpu_decode() {
    printf("Test 1: CPU encode -> GPU decode (integer data, scale=1)\n");
    const int N = 4096;
    std::vector<float> h_vals(N * 32);
    unsigned h = 0xCAFEBABE;
    for (int i = 0; i < N * 32; ++i) {
        h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
        int ival = (int)(h % 201) - 100;
        if ((h >> 24) < 128) ival = 0; // ~50% zeros
        h_vals[i] = (float)ival;
    }

    // CPU encode
    std::vector<unsigned char> h_encoded(N * MAX_ENC, 0);
    std::vector<int> h_sizes(N);
    for (int i = 0; i < N; ++i) {
        int quantized[32];
        for (int j = 0; j < 32; ++j) quantized[j] = (int)h_vals[i * 32 + j];
        h_sizes[i] = encode_zline(quantized, 32, h_encoded.data() + i * MAX_ENC);
    }

    // GPU decode
    unsigned char* d_encoded;
    int* d_sizes;
    float* d_decoded;
    HIPCHECK(hipMalloc(&d_encoded, N * MAX_ENC));
    HIPCHECK(hipMalloc(&d_sizes, N * sizeof(int)));
    HIPCHECK(hipMalloc(&d_decoded, N * 32 * sizeof(float)));
    HIPCHECK(hipMemcpy(d_encoded, h_encoded.data(), N * MAX_ENC, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_sizes, h_sizes.data(), N * sizeof(int), hipMemcpyHostToDevice));

    kernel_decode<<<(N + 255) / 256, 256>>>(d_encoded, d_sizes, 1.0f, d_decoded, N);
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_decoded(N * 32);
    HIPCHECK(hipMemcpy(h_decoded.data(), d_decoded, N * 32 * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(d_encoded)); HIPCHECK(hipFree(d_sizes)); HIPCHECK(hipFree(d_decoded));

    int mismatches = 0;
    for (int i = 0; i < N * 32; ++i) {
        if (h_decoded[i] != h_vals[i]) {
            if (mismatches < 5)
                printf("  MISMATCH at %d: expected %.0f, got %.0f\n", i, h_vals[i], h_decoded[i]);
            mismatches++;
        }
    }
    printf("  %d mismatches out of %d values %s\n", mismatches, N * 32, mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

// ---- Test 2: GPU encode → GPU decode (integer data, scale=1) ----
bool test_gpu_roundtrip_lossless() {
    printf("Test 2: GPU encode -> GPU decode round-trip (integer data, scale=1)\n");
    const int N = 4096;
    std::vector<float> h_vals(N * 32);
    unsigned h = 0xBAADF00D;
    for (int i = 0; i < N * 32; ++i) {
        h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
        int ival = (int)(h % 201) - 100;
        if ((h >> 24) < 128) ival = 0;
        h_vals[i] = (float)ival;
    }

    float *d_vals, *d_decoded;
    unsigned char* d_encoded;
    int* d_sizes;
    HIPCHECK(hipMalloc(&d_vals, N * 32 * sizeof(float)));
    HIPCHECK(hipMalloc(&d_decoded, N * 32 * sizeof(float)));
    HIPCHECK(hipMalloc(&d_encoded, N * MAX_ENC));
    HIPCHECK(hipMalloc(&d_sizes, N * sizeof(int)));
    HIPCHECK(hipMemcpy(d_vals, h_vals.data(), N * 32 * sizeof(float), hipMemcpyHostToDevice));

    kernel_encode<<<(N + 255) / 256, 256>>>(d_vals, 1.0f, d_encoded, d_sizes, N);
    HIPCHECK(hipDeviceSynchronize());
    kernel_decode<<<(N + 255) / 256, 256>>>(d_encoded, d_sizes, 1.0f, d_decoded, N);
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_decoded(N * 32);
    HIPCHECK(hipMemcpy(h_decoded.data(), d_decoded, N * 32 * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(d_vals)); HIPCHECK(hipFree(d_decoded));
    HIPCHECK(hipFree(d_encoded)); HIPCHECK(hipFree(d_sizes));

    int mismatches = 0;
    for (int i = 0; i < N * 32; ++i) {
        if (h_decoded[i] != h_vals[i]) {
            if (mismatches < 5)
                printf("  MISMATCH at %d: expected %.0f, got %.0f\n", i, h_vals[i], h_decoded[i]);
            mismatches++;
        }
    }
    printf("  %d mismatches out of %d values %s\n", mismatches, N * 32, mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

// ---- Test 3: GPU encode → GPU decode with quantization ----
bool test_gpu_roundtrip_quantized() {
    printf("Test 3: GPU encode -> GPU decode with quantization (scale=100)\n");
    const int N = 4096;
    const float scale = 100.0f;
    const float inv_scale = 1.0f / scale;
    std::vector<float> h_vals(N * 32);
    unsigned h = 0x12345678;
    for (int i = 0; i < N * 32; ++i) {
        h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
        float u = (float)(h & 0xFFFFFF) / (float)0xFFFFFF;
        float sign = (h & 0x1000000) ? -1.0f : 1.0f;
        h_vals[i] = sign * u * 10.0f;
    }

    float *d_vals, *d_decoded;
    unsigned char* d_encoded;
    int* d_sizes;
    HIPCHECK(hipMalloc(&d_vals, N * 32 * sizeof(float)));
    HIPCHECK(hipMalloc(&d_decoded, N * 32 * sizeof(float)));
    HIPCHECK(hipMalloc(&d_encoded, N * MAX_ENC));
    HIPCHECK(hipMalloc(&d_sizes, N * sizeof(int)));
    HIPCHECK(hipMemcpy(d_vals, h_vals.data(), N * 32 * sizeof(float), hipMemcpyHostToDevice));

    kernel_encode<<<(N + 255) / 256, 256>>>(d_vals, scale, d_encoded, d_sizes, N);
    HIPCHECK(hipDeviceSynchronize());
    kernel_decode<<<(N + 255) / 256, 256>>>(d_encoded, d_sizes, inv_scale, d_decoded, N);
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_decoded(N * 32);
    HIPCHECK(hipMemcpy(h_decoded.data(), d_decoded, N * 32 * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(d_vals)); HIPCHECK(hipFree(d_decoded));
    HIPCHECK(hipFree(d_encoded)); HIPCHECK(hipFree(d_sizes));

    // With quantization, we expect: decoded ≈ round(val * scale) / scale
    // Max error should be ≤ 0.5 * inv_scale = 0.005
    float max_err = 0.0f;
    for (int i = 0; i < N * 32; ++i) {
        float expected = (float)((int)(h_vals[i] * scale)) * inv_scale;
        float err = fabsf(h_decoded[i] - expected);
        if (err > max_err) max_err = err;
    }
    bool pass = max_err < 1e-5f;
    printf("  max_err vs expected = %.2e %s\n", max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ---- Test 4: Edge cases (all zeros, all nonzero, large values) ----
bool test_edge_cases() {
    printf("Test 4: Edge cases\n");
    const int N = 4;
    std::vector<float> h_vals(N * 32, 0.0f);

    // Line 0: all zeros
    // Line 1: all ones
    for (int i = 0; i < 32; ++i) h_vals[32 + i] = 1.0f;
    // Line 2: alternating 0, nonzero
    for (int i = 0; i < 32; i += 2) h_vals[64 + i + 1] = (float)(i + 1);
    // Line 3: large values requiring VLESC2/VLESC3
    h_vals[96] = 200.0f; h_vals[97] = -200.0f;
    h_vals[98] = 30000.0f; h_vals[99] = -30000.0f;

    float *d_vals, *d_decoded;
    unsigned char* d_encoded;
    int* d_sizes;
    HIPCHECK(hipMalloc(&d_vals, N * 32 * sizeof(float)));
    HIPCHECK(hipMalloc(&d_decoded, N * 32 * sizeof(float)));
    HIPCHECK(hipMalloc(&d_encoded, N * MAX_ENC));
    HIPCHECK(hipMalloc(&d_sizes, N * sizeof(int)));
    HIPCHECK(hipMemcpy(d_vals, h_vals.data(), N * 32 * sizeof(float), hipMemcpyHostToDevice));

    kernel_encode<<<1, N>>>(d_vals, 1.0f, d_encoded, d_sizes, N);
    HIPCHECK(hipDeviceSynchronize());
    kernel_decode<<<1, N>>>(d_encoded, d_sizes, 1.0f, d_decoded, N);
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_decoded(N * 32);
    HIPCHECK(hipMemcpy(h_decoded.data(), d_decoded, N * 32 * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(d_vals)); HIPCHECK(hipFree(d_decoded));
    HIPCHECK(hipFree(d_encoded)); HIPCHECK(hipFree(d_sizes));

    int mismatches = 0;
    const char* names[] = {"all-zeros", "all-ones", "alternating", "large-values"};
    for (int line = 0; line < N; ++line) {
        int line_mm = 0;
        for (int j = 0; j < 32; ++j) {
            if (h_decoded[line * 32 + j] != h_vals[line * 32 + j]) line_mm++;
        }
        printf("  %-15s: %s\n", names[line], line_mm == 0 ? "PASS" : "FAIL");
        mismatches += line_mm;
    }
    return mismatches == 0;
}

int main() {
    printf("=== Z-line RLE Decoder Tests ===\n\n");
    bool all_pass = true;
    if (!test_cpu_encode_gpu_decode()) all_pass = false;
    printf("\n");
    if (!test_gpu_roundtrip_lossless()) all_pass = false;
    printf("\n");
    if (!test_gpu_roundtrip_quantized()) all_pass = false;
    printf("\n");
    if (!test_edge_cases()) all_pass = false;
    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
