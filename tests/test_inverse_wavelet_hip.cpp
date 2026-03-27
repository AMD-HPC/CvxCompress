// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// Unit test: forward wavelet → inverse wavelet round-trip on GPU.
// Verifies us79 inverse implementation matches ds79 forward.
// Tests: (1) scalar CPU round-trip, (2) unrolled reg32 GPU round-trip,
//        (3) full 3D ZYX kernel round-trip.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "ds79.h"

#define HIPCHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---- Test 1: CPU scalar round-trip ----
bool test_cpu_scalar() {
    printf("Test 1: CPU scalar forward->inverse round-trip\n");
    bool pass = true;
    for (int dim : {2, 4, 8, 16, 32}) {
        float data[32], orig[32];
        unsigned h = 0xDEADBEEF ^ dim;
        for (int i = 0; i < dim; ++i) {
            h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
            orig[i] = data[i] = ((float)(h & 0xFFFFFF) / (float)0xFFFFFF - 0.5f) * 100.0f;
        }
        ds79_forward(data, dim);
        us79_inverse(data, dim);
        float max_err = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float err = fabsf(data[i] - orig[i]);
            if (err > max_err) max_err = err;
        }
        printf("  dim=%2d: max_err = %.2e %s\n", dim, max_err, max_err < 1e-4f ? "PASS" : "FAIL");
        if (max_err >= 1e-4f) pass = false;
    }
    return pass;
}

// ---- Test 2: GPU reg32 kernel round-trip ----
using float4_vec = ds79_float4_vec;

__global__ void kernel_forward_inverse_reg32(float* data, int n_lines) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_lines) return;
    float line[32];
    for (int i = 0; i < 32; ++i) line[i] = data[tid * 32 + i];
    ds79_forward_reg32(line);
    us79_inverse_reg32(line);
    for (int i = 0; i < 32; ++i) data[tid * 32 + i] = line[i];
}

bool test_gpu_reg32() {
    printf("Test 2: GPU reg32 forward->inverse round-trip\n");
    const int N_LINES = 1024;
    const int N = N_LINES * 32;
    std::vector<float> h_orig(N), h_result(N);
    unsigned h = 42;
    for (int i = 0; i < N; ++i) {
        h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
        h_orig[i] = ((float)(h & 0xFFFFFF) / (float)0xFFFFFF - 0.5f) * 100.0f;
    }

    float* d_data;
    HIPCHECK(hipMalloc(&d_data, N * sizeof(float)));
    HIPCHECK(hipMemcpy(d_data, h_orig.data(), N * sizeof(float), hipMemcpyHostToDevice));
    kernel_forward_inverse_reg32<<<(N_LINES + 255) / 256, 256>>>(d_data, N_LINES);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(h_result.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(d_data));

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = fabsf(h_result[i] - h_orig[i]);
        if (err > max_err) max_err = err;
    }
    printf("  %d lines: max_err = %.2e %s\n", N_LINES, max_err, max_err < 1e-3f ? "PASS" : "FAIL");
    return max_err < 1e-3f;
}

// ---- Test 3: Full 3D ZYX kernel round-trip ----
// Forward ZYX kernel: same structure as waveletRLEFusedKernel but writes back to global.
// Uses buffer load, Z-transform (regs), Y+X-transform (LDS), then stores back.

__launch_bounds__(256, 2)
__global__ void waveletForwardZYXKernel(
    float* __restrict__ data,
    int ldimx, int ldimxy)
{
    constexpr int PLANES = 32;
    constexpr int BATCH  = 8;
    constexpr int SLC    = 2;

    __shared__ float lds[BATCH * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(block_base, 0, -1, 0x00027000);
    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    int byte_off    = (gx + gy * ldimx) * (int)sizeof(float);
    int byte_stride = ldimxy * (int)sizeof(float);

    float4_vec regs[PLANES];
    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_off, p * byte_stride, SLC));

    ds79_forward_f4_scalar_tmp(regs, PLANES);

    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v = regs[pb + dp];
            int x0 = xg * 4;
            lds[dp * 1024 + (x0+0) * 32 + (yr ^ (x0+0))] = v[0];
            lds[dp * 1024 + (x0+1) * 32 + (yr ^ (x0+1))] = v[1];
            lds[dp * 1024 + (x0+2) * 32 + (yr ^ (x0+2))] = v[2];
            lds[dp * 1024 + (x0+3) * 32 + (yr ^ (x0+3))] = v[3];
        }
        __syncthreads();

        int pl  = tid / 32;
        int pos = tid % 32;
        float line[32];
        for (int y = 0; y < 32; y++)
            line[y] = lds[pl * 1024 + pos * 32 + (y ^ pos)];
        ds79_forward_reg32(line);
        for (int y = 0; y < 32; y++)
            lds[pl * 1024 + pos * 32 + (y ^ pos)] = line[y];
        __syncthreads();

        for (int x = 0; x < 32; x++)
            line[x] = lds[pl * 1024 + x * 32 + (pos ^ x)];
        ds79_forward_reg32(line);
        for (int x = 0; x < 32; x++)
            lds[pl * 1024 + x * 32 + (pos ^ x)] = line[x];
        __syncthreads();

        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v;
            int x0 = xg * 4;
            v[0] = lds[dp * 1024 + (x0+0) * 32 + (yr ^ (x0+0))];
            v[1] = lds[dp * 1024 + (x0+1) * 32 + (yr ^ (x0+1))];
            v[2] = lds[dp * 1024 + (x0+2) * 32 + (yr ^ (x0+2))];
            v[3] = lds[dp * 1024 + (x0+3) * 32 + (yr ^ (x0+3))];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    // Write back: buffer store
    #pragma unroll
    for (int p = 0; p < PLANES; p++) {
        auto v = __builtin_bit_cast(__attribute__((__vector_size__(16))) int, regs[p]);
        __builtin_amdgcn_raw_buffer_store_b128(v, rsrc, byte_off, p * byte_stride, SLC);
    }
}

// Inverse: X+Y in LDS (inverse order), then Z in registers.
__launch_bounds__(256, 2)
__global__ void waveletInverseZYXKernel(
    float* __restrict__ data,
    int ldimx, int ldimxy)
{
    constexpr int PLANES = 32;
    constexpr int BATCH  = 8;
    constexpr int SLC    = 2;

    __shared__ float lds[BATCH * 1024];

    int tid = threadIdx.x;
    int xg  = tid % 8;
    int yr  = tid / 8;

    float* block_base = data + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(block_base, 0, -1, 0x00027000);
    int gx = blockIdx.x * 32 + xg * 4;
    int gy = blockIdx.y * 32 + yr;
    int byte_off    = (gx + gy * ldimx) * (int)sizeof(float);
    int byte_stride = ldimxy * (int)sizeof(float);

    // Load wavelet coefficients from global
    float4_vec regs[PLANES];
    #pragma unroll
    for (int p = 0; p < PLANES; p++)
        regs[p] = __builtin_bit_cast(float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_off, p * byte_stride, SLC));

    // Inverse X+Y transform in LDS (reverse order: X first, then Y)
    for (int pb = 0; pb < PLANES; pb += BATCH) {
        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v = regs[pb + dp];
            int x0 = xg * 4;
            lds[dp * 1024 + (x0+0) * 32 + (yr ^ (x0+0))] = v[0];
            lds[dp * 1024 + (x0+1) * 32 + (yr ^ (x0+1))] = v[1];
            lds[dp * 1024 + (x0+2) * 32 + (yr ^ (x0+2))] = v[2];
            lds[dp * 1024 + (x0+3) * 32 + (yr ^ (x0+3))] = v[3];
        }
        __syncthreads();

        int pl  = tid / 32;
        int pos = tid % 32;
        float line[32];

        // Inverse X
        for (int x = 0; x < 32; x++)
            line[x] = lds[pl * 1024 + x * 32 + (pos ^ x)];
        us79_inverse_reg32(line);
        for (int x = 0; x < 32; x++)
            lds[pl * 1024 + x * 32 + (pos ^ x)] = line[x];
        __syncthreads();

        // Inverse Y
        for (int y = 0; y < 32; y++)
            line[y] = lds[pl * 1024 + pos * 32 + (y ^ pos)];
        us79_inverse_reg32(line);
        for (int y = 0; y < 32; y++)
            lds[pl * 1024 + pos * 32 + (y ^ pos)] = line[y];
        __syncthreads();

        for (int dp = 0; dp < BATCH; dp++) {
            float4_vec v;
            int x0 = xg * 4;
            v[0] = lds[dp * 1024 + (x0+0) * 32 + (yr ^ (x0+0))];
            v[1] = lds[dp * 1024 + (x0+1) * 32 + (yr ^ (x0+1))];
            v[2] = lds[dp * 1024 + (x0+2) * 32 + (yr ^ (x0+2))];
            v[3] = lds[dp * 1024 + (x0+3) * 32 + (yr ^ (x0+3))];
            regs[pb + dp] = v;
        }
        __syncthreads();
    }

    // Inverse Z in registers
    us79_inverse_f4_scalar_tmp(regs, PLANES);

    // Write back
    #pragma unroll
    for (int p = 0; p < PLANES; p++) {
        auto v = __builtin_bit_cast(__attribute__((__vector_size__(16))) int, regs[p]);
        __builtin_amdgcn_raw_buffer_store_b128(v, rsrc, byte_off, p * byte_stride, SLC);
    }
}

__global__ void initRandom(float* data, long N, unsigned seed)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    unsigned h = seed ^ (unsigned)idx;
    h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
    float u = (float)(h & 0x7FFFFF) / (float)0x7FFFFF;
    float sign = (h & 0x800000) ? -1.0f : 1.0f;
    data[idx] = sign * u * 500.0f;
}

bool test_gpu_zyx_roundtrip(int NX, int NY, int NZ) {
    printf("Test 3: GPU ZYX forward->inverse round-trip (%dx%dx%d)\n", NX, NY, NZ);
    int ldimx = NX, ldimxy = NX * NY;
    long N = (long)NX * NY * NZ;

    float *d_data, *d_orig;
    HIPCHECK(hipMalloc(&d_data, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_orig, N * sizeof(float)));

    { int thr=256, blk=(N+thr-1)/thr; initRandom<<<blk,thr>>>(d_data, N, 12345); HIPCHECK(hipDeviceSynchronize()); }
    HIPCHECK(hipMemcpy(d_orig, d_data, N * sizeof(float), hipMemcpyDeviceToDevice));

    dim3 grid((NX+31)/32, (NY+31)/32, (NZ+31)/32);

    // Forward
    waveletForwardZYXKernel<<<grid, dim3(256)>>>(d_data, ldimx, ldimxy);
    HIPCHECK(hipDeviceSynchronize());

    // Inverse
    waveletInverseZYXKernel<<<grid, dim3(256)>>>(d_data, ldimx, ldimxy);
    HIPCHECK(hipDeviceSynchronize());

    // Compare
    std::vector<float> h_orig(N), h_result(N);
    HIPCHECK(hipMemcpy(h_orig.data(), d_orig, N * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_result.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(d_data)); HIPCHECK(hipFree(d_orig));

    float max_err = 0.0f;
    double sum_sq_err = 0.0;
    for (long i = 0; i < N; ++i) {
        float err = fabsf(h_result[i] - h_orig[i]);
        if (err > max_err) max_err = err;
        sum_sq_err += (double)err * err;
    }
    float rms_err = (float)sqrt(sum_sq_err / N);
    bool pass = max_err < 0.1f;
    printf("  max_err = %.2e, rms_err = %.2e %s\n", max_err, rms_err, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("=== Inverse Wavelet Transform Tests ===\n\n");
    bool all_pass = true;
    if (!test_cpu_scalar()) all_pass = false;
    printf("\n");
    if (!test_gpu_reg32()) all_pass = false;
    printf("\n");
    if (!test_gpu_zyx_roundtrip(64, 64, 64)) all_pass = false;
    if (!test_gpu_zyx_roundtrip(128, 128, 128)) all_pass = false;
    if (!test_gpu_zyx_roundtrip(256, 256, 256)) all_pass = false;
    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
