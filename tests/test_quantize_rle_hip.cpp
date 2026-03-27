// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// GPU quantize+RLE encode test (two-pass compacted kernel).
// Validates GPU kernel output against CPU reference (quantize_rle_ref.h).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "hipQuantizeRLE.h"
#include "quantize_rle_ref.h"

#define HIPCHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static constexpr int BX = 32, BY = 32, BZ = 32;
static constexpr int BLOCK_SIZE = BX * BY * BZ;

static void extract_zline(const float* block, int ix, int iy, float* zline)
{
    for (int iz = 0; iz < BZ; ++iz)
        zline[iz] = block[iz * BY * BX + iy * BX + ix];
}

// tid % 8 -> x-group (float4), tid / 8 -> y-row
static void zline_id_to_xy(int local_zline_id, int& ix, int& iy)
{
    int tid   = local_zline_id / 4;
    int x_off = local_zline_id % 4;
    ix = (tid % 8) * 4 + x_off;
    iy = tid / 8;
}

struct TestResult { int tested; int passed; };

static TestResult run_test(const char* name, const float* h_block, float scale)
{
    TestResult result = {0, 0};

    float* d_input;
    unsigned char* d_output;
    int* d_block_sizes;
    int ldimx = BX, ldimxy = BX * BY;
    long out_bytes = 4L * QRLE_LDS_BYTES;

    HIPCHECK(hipMalloc(&d_input, BLOCK_SIZE * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, out_bytes));
    HIPCHECK(hipMalloc(&d_block_sizes, sizeof(int)));
    HIPCHECK(hipMemcpy(d_input, h_block, BLOCK_SIZE * sizeof(float), hipMemcpyHostToDevice));
    HIPCHECK(hipMemset(d_output, 0, out_bytes));

    HIPCHECK(hipQuantizeRLEEncode(
        d_input, d_output, d_block_sizes, scale,
        BX, BY, BZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<unsigned char> h_output(out_bytes);
    int h_block_size = 0;
    HIPCHECK(hipMemcpy(h_output.data(), d_output, out_bytes, hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(&h_block_size, d_block_sizes, sizeof(int), hipMemcpyDeviceToHost));

    // The compacted output order is: x_off=0 (tids 0..255), x_off=1, x_off=2, x_off=3.
    // Within each x_off, z-lines are ordered by tid in prefix-sum order.
    unsigned char cpu_buf[256];
    float zline[BZ];
    int mismatches = 0;
    int offset = 0;
    int cpu_total = 0;

    for (int x_off = 0; x_off < 4; ++x_off) {
        for (int tid = 0; tid < 256; ++tid) {
            int zl = tid * 4 + x_off;
            int ix, iy;
            zline_id_to_xy(zl, ix, iy);
            extract_zline(h_block, ix, iy, zline);

            int cpu_bytes = quantize_encode_zline(zline, scale, BZ, cpu_buf);
            cpu_total += cpu_bytes;
            result.tested++;

            if (offset + cpu_bytes > h_block_size) {
                if (mismatches < 5)
                    printf("  [%s] zline %d (ix=%d,iy=%d): stream overflow at offset %d + %d > %d\n",
                           name, zl, ix, iy, offset, cpu_bytes, h_block_size);
                mismatches++;
                offset += cpu_bytes;
                continue;
            }

            const unsigned char* gpu_data = h_output.data() + offset;
            bool match = (memcmp(cpu_buf, gpu_data, cpu_bytes) == 0);
            if (!match) {
                if (mismatches < 5) {
                    printf("  [%s] zline %d (ix=%d,iy=%d): data mismatch at offset %d (%d bytes)\n",
                           name, zl, ix, iy, offset, cpu_bytes);
                    printf("    CPU:");
                    for (int b = 0; b < cpu_bytes && b < 16; ++b) printf(" %02x", cpu_buf[b]);
                    printf("\n    GPU:");
                    for (int b = 0; b < cpu_bytes && b < 16; ++b) printf(" %02x", gpu_data[b]);
                    printf("\n");
                }
                mismatches++;
            } else {
                result.passed++;
            }
            offset += cpu_bytes;
        }
    }

    if (cpu_total != h_block_size) {
        printf("  [%s] WARNING: block_size mismatch: cpu_total=%d gpu_block_size=%d\n",
               name, cpu_total, h_block_size);
    }

    printf("  [%s] %d/%d z-lines match (%d mismatches, block_size=%d)\n",
           name, result.passed, result.tested, mismatches, h_block_size);

    HIPCHECK(hipFree(d_input));
    HIPCHECK(hipFree(d_output));
    HIPCHECK(hipFree(d_block_sizes));
    return result;
}

int main()
{
    printf("=== GPU Quantize+RLE Encode Test (two-pass compacted) ===\n\n");
    int total_tested = 0, total_passed = 0;

    {
        std::vector<float> block(BLOCK_SIZE, 0.0f);
        auto r = run_test("all_zeros", block.data(), 10.0f);
        total_tested += r.tested; total_passed += r.passed;
    }
    {
        std::vector<float> block(BLOCK_SIZE, 3.0f);
        auto r = run_test("constant_3", block.data(), 1.0f);
        total_tested += r.tested; total_passed += r.passed;
    }
    {
        std::vector<float> block(BLOCK_SIZE);
        for (int i = 0; i < BLOCK_SIZE; ++i)
            block[i] = (i % 2 == 0) ? 0.0f : 5.0f;
        auto r = run_test("alternating", block.data(), 1.0f);
        total_tested += r.tested; total_passed += r.passed;
    }
    {
        std::vector<float> block(BLOCK_SIZE);
        for (int i = 0; i < BLOCK_SIZE; ++i)
            block[i] = (float)(i - BLOCK_SIZE / 2);
        auto r = run_test("ramp_s1", block.data(), 1.0f);
        total_tested += r.tested; total_passed += r.passed;
        auto r2 = run_test("ramp_s10", block.data(), 10.0f);
        total_tested += r2.tested; total_passed += r2.passed;
    }
    {
        std::vector<float> block(BLOCK_SIZE);
        srand(42);
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            float u = (float)rand() / (float)RAND_MAX;
            float sign = (rand() % 2) ? 1.0f : -1.0f;
            block[i] = sign * powf(u, 3.0f) * 1000.0f;
        }
        auto r = run_test("random_s01", block.data(), 0.1f);
        total_tested += r.tested; total_passed += r.passed;
        auto r2 = run_test("random_s1", block.data(), 1.0f);
        total_tested += r2.tested; total_passed += r2.passed;
        auto r3 = run_test("random_s100", block.data(), 100.0f);
        total_tested += r3.tested; total_passed += r3.passed;
    }
    {
        std::vector<float> block(BLOCK_SIZE, 0.0f);
        for (int iy = 0; iy < BY; ++iy)
            for (int ix = 0; ix < BX; ++ix)
                block[15 * BY * BX + iy * BX + ix] = 42.0f;
        auto r = run_test("single_nz_z15", block.data(), 1.0f);
        total_tested += r.tested; total_passed += r.passed;
    }

    printf("\n=== Total: %d/%d z-lines passed ===\n", total_passed, total_tested);
    return (total_passed == total_tested) ? 0 : 1;
}
