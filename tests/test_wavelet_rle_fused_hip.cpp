// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// Correctness and performance test for fused wavelet ZYX + quantize RLE kernel.
// Correctness: verifies determinism (two fused runs byte-identical) and
// compares compression ratio against unfused reference.
// FP differences between fused/unfused are expected (different compilation
// contexts → different FMA contraction), so we compare sizes, not bytes.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "hipWaveletRLE.h"
#include "hipWaveletRLEInverse.h"
#include "hipWaveletTransformBuffer.h"
#include "hipQuantizeRLE.h"
#include "CvxCompress.hxx"
#include "quantize_rle_ref.h"

#define HIPCHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

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

__global__ void initSinKernel(float* data, int NX, int NY, int NZ, float kx, float ky, float kz)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long N = (long)NX * NY * NZ;
    if (idx >= N) return;
    int ix = idx % NX;
    int iy = (idx / NX) % NY;
    int iz = idx / ((long)NX * NY);
    float x = (float)ix / NX;
    float y = (float)iy / NY;
    float z = (float)iz / NZ;
    data[idx] = sinf(kx * 6.2831853f * x) * sinf(ky * 6.2831853f * y) * sinf(kz * 6.2831853f * z);
}

__global__ void computeRMS_kernel(const float* data, double* partial, long N, int chunk)
{
    long start = (long)blockIdx.x * chunk;
    long end = start + chunk;
    if (end > N) end = N;
    double sum = 0.0;
    for (long i = start; i < end; ++i) sum += (double)data[i] * data[i];
    partial[blockIdx.x] = sum;
}

float gpuRMS(const float* d_data, long N)
{
    int chunk = 65536;
    int nblk = (N + chunk - 1) / chunk;
    double* d_p;
    HIPCHECK(hipMalloc(&d_p, nblk * sizeof(double)));
    computeRMS_kernel<<<nblk, 1>>>(d_data, d_p, N, chunk);
    HIPCHECK(hipDeviceSynchronize());
    std::vector<double> h_p(nblk);
    HIPCHECK(hipMemcpy(h_p.data(), d_p, nblk * sizeof(double), hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(d_p));
    double tot = 0.0;
    for (auto v : h_p) tot += v;
    return (float)sqrt(tot / N);
}

typedef hipError_t (*FusedLauncher)(const float*, unsigned char*, size_t*, float, int, int, int, int, int);

bool test_correctness(int NX, int NY, int NZ, float scale, FusedLauncher launcher, const char* tag)
{
    int ldimx = NX, ldimxy = NX * NY;
    long N = (long)NX * NY * NZ;
    long nblocks = (long)(NX/32) * (NY/32) * (NZ/32);
    long out_cap = nblocks * 4L * WRLE_LDS_BYTES;

    float *d_raw, *d_wavelet;
    unsigned char *d_out1, *d_out2, *d_out_ref;
    size_t *d_sizes1, *d_sizes2;
    int *d_sizes_ref;

    HIPCHECK(hipMalloc(&d_raw, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_wavelet, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out1, out_cap));
    HIPCHECK(hipMalloc(&d_out2, out_cap));
    HIPCHECK(hipMalloc(&d_out_ref, out_cap));
    HIPCHECK(hipMalloc(&d_sizes1, nblocks * sizeof(size_t)));
    HIPCHECK(hipMalloc(&d_sizes2, nblocks * sizeof(size_t)));
    HIPCHECK(hipMalloc(&d_sizes_ref, nblocks * sizeof(int)));

    { int thr=256, blk=(N+thr-1)/thr; initRandom<<<blk,thr>>>(d_raw,N,12345); HIPCHECK(hipDeviceSynchronize()); }

    HIPCHECK(hipMemset(d_out1, 0, out_cap));
    HIPCHECK(launcher(d_raw, d_out1, d_sizes1, scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipMemset(d_out2, 0, out_cap));
    HIPCHECK(launcher(d_raw, d_out2, d_sizes2, scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipMemcpy(d_wavelet, d_raw, N * sizeof(float), hipMemcpyDeviceToDevice));
    HIPCHECK((hipWaveletTransformBufferZYX<256, 2>(d_wavelet, NX, NY, NZ, 32, 32, 32, ldimx, ldimxy)));
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemset(d_out_ref, 0, out_cap));
    HIPCHECK(hipQuantizeRLEEncode(d_wavelet, d_out_ref, d_sizes_ref, scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<size_t> h1(nblocks), h2(nblocks);
    std::vector<int> href(nblocks);
    HIPCHECK(hipMemcpy(h1.data(), d_sizes1, nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h2.data(), d_sizes2, nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(href.data(), d_sizes_ref, nblocks*sizeof(int), hipMemcpyDeviceToHost));

    bool pass = true;

    for (long i = 0; i < nblocks; ++i) {
        if (h1[i] != h2[i]) { printf("  FAIL [%s]: non-deterministic size at block %ld\n", tag, i); pass = false; break; }
    }

    if (pass) {
        std::vector<unsigned char> ho1(out_cap), ho2(out_cap);
        HIPCHECK(hipMemcpy(ho1.data(), d_out1, out_cap, hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(ho2.data(), d_out2, out_cap, hipMemcpyDeviceToHost));
        for (long b = 0; b < nblocks; ++b) {
            long off = b * 4L * WRLE_LDS_BYTES;
            if (memcmp(&ho1[off], &ho2[off], h1[b]) != 0) {
                printf("  FAIL [%s]: non-deterministic bytes at block %ld\n", tag, b); pass = false; break;
            }
        }
    }

    long total_fused = 0, total_ref = 0;
    for (long i = 0; i < nblocks; ++i) {
        total_fused += h1[i]; total_ref += href[i];
        if (h1[i] == 0 || h1[i] > 4*WRLE_LDS_BYTES) { printf("  FAIL [%s]: bad size %zu at block %ld\n", tag, h1[i], i); pass = false; }
    }

    float cr_f = (float)(N*4) / (float)total_fused;
    float cr_r = (float)(N*4) / (float)total_ref;
    float ratio = (float)total_fused / (float)total_ref;

    if (pass)
        printf("  PASS  CR %.2f:1 (unfused %.2f:1, size ratio %.4f)\n", cr_f, cr_r, ratio);

    HIPCHECK(hipFree(d_raw)); HIPCHECK(hipFree(d_wavelet));
    HIPCHECK(hipFree(d_out1)); HIPCHECK(hipFree(d_out2)); HIPCHECK(hipFree(d_out_ref));
    HIPCHECK(hipFree(d_sizes1)); HIPCHECK(hipFree(d_sizes2)); HIPCHECK(hipFree(d_sizes_ref));
    return pass;
}

bool test_compaction(int NX, int NY, int NZ, float scale)
{
    int ldimx = NX, ldimxy = NX * NY;
    long N = (long)NX * NY * NZ;
    int nblocks = (NX/32) * (NY/32) * (NZ/32);
    long out_cap = (long)nblocks * 4L * WRLE_LDS_BYTES;

    float *d_raw;
    unsigned char *d_scratch, *d_compact;
    size_t *d_sizes, *d_offsets;
    void *d_scan_temp;

    HIPCHECK(hipMalloc(&d_raw, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_scratch, out_cap));
    HIPCHECK(hipMalloc(&d_compact, out_cap));
    HIPCHECK(hipMalloc(&d_sizes, nblocks * sizeof(size_t)));
    HIPCHECK(hipMalloc(&d_offsets, nblocks * sizeof(size_t)));

    size_t scan_temp_bytes = 0;
    HIPCHECK(hipWaveletRLECompactScanTempSize(nblocks, &scan_temp_bytes));
    HIPCHECK(hipMalloc(&d_scan_temp, scan_temp_bytes));

    { int thr=256, blk=(N+thr-1)/thr; initRandom<<<blk,thr>>>(d_raw,N,12345); HIPCHECK(hipDeviceSynchronize()); }

    // Run non-compact (reference)
    unsigned char *d_ref;
    size_t *d_ref_sizes;
    HIPCHECK(hipMalloc(&d_ref, out_cap));
    HIPCHECK(hipMalloc(&d_ref_sizes, nblocks * sizeof(size_t)));
    HIPCHECK(hipWaveletRLEFused(d_raw, d_ref, d_ref_sizes, scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    // Run compact
    HIPCHECK(hipWaveletRLEFusedCompact(
        d_raw, d_scratch, d_compact, d_sizes, d_offsets,
        d_scan_temp, scan_temp_bytes, scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<size_t> h_sizes(nblocks), h_offsets(nblocks), h_ref_sizes(nblocks);
    HIPCHECK(hipMemcpy(h_sizes.data(), d_sizes, nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_offsets.data(), d_offsets, nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_ref_sizes.data(), d_ref_sizes, nblocks*sizeof(size_t), hipMemcpyDeviceToHost));

    bool pass = true;
    for (int i = 0; i < nblocks; ++i) {
        if (h_sizes[i] != h_ref_sizes[i]) {
            printf("  FAIL: compact size mismatch at block %d (%zu vs %zu)\n", i, h_sizes[i], h_ref_sizes[i]);
            pass = false; break;
        }
    }

    long total = (long)h_offsets[nblocks-1] + (long)h_sizes[nblocks-1];

    // Verify offsets are a valid prefix sum
    if (pass) {
        size_t expected = 0;
        for (int i = 0; i < nblocks; ++i) {
            if (h_offsets[i] != expected) {
                printf("  FAIL: offset mismatch at block %d (got %zu, expected %zu)\n", i, h_offsets[i], expected);
                pass = false; break;
            }
            expected += h_sizes[i];
        }
    }

    // Byte-compare compacted data vs reference (fixed-stride)
    if (pass) {
        std::vector<unsigned char> h_compact(total), h_ref(out_cap);
        HIPCHECK(hipMemcpy(h_compact.data(), d_compact, total, hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(h_ref.data(), d_ref, out_cap, hipMemcpyDeviceToHost));
        for (int b = 0; b < nblocks; ++b) {
            long ref_off = (long)b * 4L * WRLE_LDS_BYTES;
            if (memcmp(&h_compact[h_offsets[b]], &h_ref[ref_off], h_sizes[b]) != 0) {
                printf("  FAIL: byte mismatch in compacted block %d\n", b);
                pass = false; break;
            }
        }
    }

    float cr = (float)(N*4) / (float)total;
    float scratch_MB = (float)out_cap / 1e6;
    float compact_MB = (float)total / 1e6;
    if (pass)
        printf("  PASS  CR %.2f:1  compact %.1f MB (scratch %.0f MB, %.1f%% util)\n",
               cr, compact_MB, scratch_MB, 100.0 * compact_MB / scratch_MB);

    HIPCHECK(hipFree(d_raw)); HIPCHECK(hipFree(d_scratch)); HIPCHECK(hipFree(d_compact));
    HIPCHECK(hipFree(d_sizes)); HIPCHECK(hipFree(d_offsets)); HIPCHECK(hipFree(d_scan_temp));
    HIPCHECK(hipFree(d_ref)); HIPCHECK(hipFree(d_ref_sizes));
    return pass;
}

void bench(int NX, int NY, int NZ, float scale, int warmup, int runs)
{
    int ldimx = NX, ldimxy = NX * NY;
    long N = (long)NX * NY * NZ;
    int nblocks = (NX/32) * (NY/32) * (NZ/32);
    long out_cap = (long)nblocks * 4L * WRLE_LDS_BYTES;
    long input_bytes = N * sizeof(float);

    float *d_raw, *d_wavelet;
    unsigned char *d_out, *d_compact;
    size_t *d_sizes, *d_offsets;
    int *d_sizes_unfused;
    void *d_scan_temp;

    HIPCHECK(hipMalloc(&d_raw, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_wavelet, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out, out_cap));
    HIPCHECK(hipMalloc(&d_compact, out_cap));
    HIPCHECK(hipMalloc(&d_sizes, nblocks * sizeof(size_t)));
    HIPCHECK(hipMalloc(&d_offsets, nblocks * sizeof(size_t)));
    HIPCHECK(hipMalloc(&d_sizes_unfused, nblocks * sizeof(int)));

    size_t scan_temp_bytes = 0;
    HIPCHECK(hipWaveletRLECompactScanTempSize(nblocks, &scan_temp_bytes));
    HIPCHECK(hipMalloc(&d_scan_temp, scan_temp_bytes));

    { int thr=256, blk=(N+thr-1)/thr; initRandom<<<blk,thr>>>(d_raw,N,42); HIPCHECK(hipDeviceSynchronize()); }

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0)); HIPCHECK(hipEventCreate(&t1));

    // Unfused: wavelet + RLE
    for (int i = 0; i < warmup; ++i) {
        HIPCHECK(hipMemcpy(d_wavelet, d_raw, N * sizeof(float), hipMemcpyDeviceToDevice));
        HIPCHECK((hipWaveletTransformBufferZYX<256, 2>(d_wavelet, NX, NY, NZ, 32, 32, 32, ldimx, ldimxy)));
        HIPCHECK(hipQuantizeRLEEncode(d_wavelet, d_out, d_sizes_unfused, scale, NX, NY, NZ, ldimx, ldimxy));
    }
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < runs; ++i) {
        HIPCHECK(hipMemcpy(d_wavelet, d_raw, N * sizeof(float), hipMemcpyDeviceToDevice));
        HIPCHECK((hipWaveletTransformBufferZYX<256, 2>(d_wavelet, NX, NY, NZ, 32, 32, 32, ldimx, ldimxy)));
        HIPCHECK(hipQuantizeRLEEncode(d_wavelet, d_out, d_sizes_unfused, scale, NX, NY, NZ, ldimx, ldimxy));
    }
    HIPCHECK(hipEventRecord(t1)); HIPCHECK(hipEventSynchronize(t1));
    float ms_unfused = 0;
    HIPCHECK(hipEventElapsedTime(&ms_unfused, t0, t1));
    ms_unfused /= runs;

    std::vector<int> h_sizes_unfused(nblocks);
    HIPCHECK(hipMemcpy(h_sizes_unfused.data(), d_sizes_unfused, nblocks*sizeof(int), hipMemcpyDeviceToHost));
    long total_enc_u = 0; for (long i = 0; i < nblocks; ++i) total_enc_u += h_sizes_unfused[i];

    float bw_u = (float)input_bytes / (ms_unfused * 1e6);
    float cr_u = (float)input_bytes / (float)total_enc_u;
    printf("  unfused          %7.3f ms  %7.1f GB/s  CR %.1f:1\n", ms_unfused, bw_u, cr_u);

    // Fused (no compaction)
    FusedLauncher launchers[] = { hipWaveletRLEFused, hipWaveletRLEFusedSaddr };
    const char* names[]       = { "fused-buf     ", "fused-saddr   " };

    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < warmup; ++i)
            HIPCHECK(launchers[k](d_raw, d_out, d_sizes, scale, NX, NY, NZ, ldimx, ldimxy));
        HIPCHECK(hipDeviceSynchronize());

        HIPCHECK(hipEventRecord(t0));
        for (int i = 0; i < runs; ++i)
            HIPCHECK(launchers[k](d_raw, d_out, d_sizes, scale, NX, NY, NZ, ldimx, ldimxy));
        HIPCHECK(hipEventRecord(t1)); HIPCHECK(hipEventSynchronize(t1));
        float ms = 0;
        HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
        ms /= runs;

        std::vector<size_t> h_sizes(nblocks);
        HIPCHECK(hipMemcpy(h_sizes.data(), d_sizes, nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
        long total_enc = 0; for (long i = 0; i < nblocks; ++i) total_enc += h_sizes[i];

        float bw = (float)input_bytes / (ms * 1e6);
        float cr = (float)input_bytes / (float)total_enc;
        printf("  %s %7.3f ms  %7.1f GB/s  CR %.1f:1  (%.2fx)\n",
               names[k], ms, bw, cr, ms_unfused / ms);
    }

    // Fused + compact (buffer variant)
    for (int i = 0; i < warmup; ++i)
        HIPCHECK(hipWaveletRLEFusedCompact(d_raw, d_out, d_compact, d_sizes, d_offsets,
            d_scan_temp, scan_temp_bytes, scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < runs; ++i)
        HIPCHECK(hipWaveletRLEFusedCompact(d_raw, d_out, d_compact, d_sizes, d_offsets,
            d_scan_temp, scan_temp_bytes, scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipEventRecord(t1)); HIPCHECK(hipEventSynchronize(t1));
    float ms_compact = 0;
    HIPCHECK(hipEventElapsedTime(&ms_compact, t0, t1));
    ms_compact /= runs;

    {
        std::vector<size_t> h_sizes_c(nblocks);
        HIPCHECK(hipMemcpy(h_sizes_c.data(), d_sizes, nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
        long total_enc_c = 0; for (long i = 0; i < nblocks; ++i) total_enc_c += h_sizes_c[i];
        float bw_c = (float)input_bytes / (ms_compact * 1e6);
        float cr_c = (float)input_bytes / (float)total_enc_c;
        printf("  fused+compact   %7.3f ms  %7.1f GB/s  CR %.1f:1  (%.2fx)\n",
               ms_compact, bw_c, cr_c, ms_unfused / ms_compact);
    }

    // ---- Inverse pipeline benchmarks ----
    float inv_scale = 1.0f / scale;
    float *d_decoded;
    HIPCHECK(hipMalloc(&d_decoded, N * sizeof(float)));

    // Forward encode (meta embedded in output)
    HIPCHECK(hipWaveletRLEFused(d_raw, d_out, d_sizes, scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    printf("  --- inverse ---\n");

    // Inverse fused (fixed-stride)
    for (int i = 0; i < warmup; ++i)
        HIPCHECK(hipWaveletRLEInverseFusedFixedStride(
            d_out, d_sizes, d_decoded, inv_scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < runs; ++i)
        HIPCHECK(hipWaveletRLEInverseFusedFixedStride(
            d_out, d_sizes, d_decoded, inv_scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipEventRecord(t1)); HIPCHECK(hipEventSynchronize(t1));
    float ms_inv = 0;
    HIPCHECK(hipEventElapsedTime(&ms_inv, t0, t1));
    ms_inv /= runs;
    float bw_inv = (float)input_bytes / (ms_inv * 1e6);
    printf("  inv-fused       %7.3f ms  %7.1f GB/s (output write BW)\n", ms_inv, bw_inv);

    HIPCHECK(hipFree(d_decoded));

    HIPCHECK(hipEventDestroy(t0)); HIPCHECK(hipEventDestroy(t1));
    HIPCHECK(hipFree(d_raw)); HIPCHECK(hipFree(d_wavelet));
    HIPCHECK(hipFree(d_out)); HIPCHECK(hipFree(d_compact));
    HIPCHECK(hipFree(d_sizes)); HIPCHECK(hipFree(d_offsets)); HIPCHECK(hipFree(d_scan_temp));
    HIPCHECK(hipFree(d_sizes_unfused));
}

void bench_sin(int NX, int NY, int NZ, float user_scale, int warmup, int runs,
               float* d_raw, float rms,
               float* d_wavelet, unsigned char* d_out, unsigned char* d_compact,
               size_t* d_sizes, size_t* d_offsets, void* d_scan_temp, size_t scan_temp_bytes)
{
    int ldimx = NX, ldimxy = NX * NY;
    long N = (long)NX * NY * NZ;
    int nblocks = (NX/32) * (NY/32) * (NZ/32);
    long input_bytes = N * sizeof(float);

    float scale = 1.0f / (rms * user_scale);

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0)); HIPCHECK(hipEventCreate(&t1));

    // Forward fused-buf
    for (int i = 0; i < warmup; ++i)
        HIPCHECK(hipWaveletRLEFused(d_raw, d_out, d_sizes, scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < runs; ++i)
        HIPCHECK(hipWaveletRLEFused(d_raw, d_out, d_sizes, scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipEventRecord(t1)); HIPCHECK(hipEventSynchronize(t1));
    float ms_fwd = 0;
    HIPCHECK(hipEventElapsedTime(&ms_fwd, t0, t1));
    ms_fwd /= runs;

    std::vector<size_t> h_sizes(nblocks);
    HIPCHECK(hipMemcpy(h_sizes.data(), d_sizes, nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
    long total_enc = 0; for (int i = 0; i < nblocks; ++i) total_enc += h_sizes[i];
    float cr = (float)input_bytes / (float)total_enc;
    float bw_fwd = (float)input_bytes / (ms_fwd * 1e6);

    // Inverse fused (fixed-stride, meta embedded in output)
    float inv_scale = 1.0f / scale;
    float *d_decoded;
    HIPCHECK(hipMalloc(&d_decoded, N * sizeof(float)));
    HIPCHECK(hipWaveletRLEFused(d_raw, d_out, d_sizes, scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    for (int i = 0; i < warmup; ++i)
        HIPCHECK(hipWaveletRLEInverseFusedFixedStride(
            d_out, d_sizes, d_decoded, inv_scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < runs; ++i)
        HIPCHECK(hipWaveletRLEInverseFusedFixedStride(
            d_out, d_sizes, d_decoded, inv_scale, NX, NY, NZ, ldimx, ldimxy));
    HIPCHECK(hipEventRecord(t1)); HIPCHECK(hipEventSynchronize(t1));
    float ms_inv = 0;
    HIPCHECK(hipEventElapsedTime(&ms_inv, t0, t1));
    ms_inv /= runs;
    float bw_inv = (float)input_bytes / (ms_inv * 1e6);

    printf("  user_scale=%.1e  CR=%5.1f:1  fwd=%6.3f ms (%7.1f GB/s)  inv=%6.3f ms (%7.1f GB/s)  inv/fwd=%.2fx\n",
           user_scale, cr, ms_fwd, bw_fwd, ms_inv, bw_inv, ms_inv / ms_fwd);

    HIPCHECK(hipFree(d_decoded));
    HIPCHECK(hipEventDestroy(t0)); HIPCHECK(hipEventDestroy(t1));
}

void test_cr_sin()
{
    printf("\n=== Compression Ratio: Sin Data ===\n");

    // Match Test_With_Generated_Input.cpp (itries=1):
    //   nx_slow=320, ny_mid=416, nz_fast=352
    // CvxCompress: "nx is fast, nz is slow", called as Compress(..., nz_fast, ny, nx_slow, ...)
    // GPU: data[ix + iy*NX + iz*NX*NY], NX=fast=352, NY=mid=416, NZ=slow=320
    int NX = 352, NY = 416, NZ = 320;
    int bx = 32, by = 32, bz = 32;
    long N = (long)NX * NY * NZ;

    float* vol;
    posix_memalign((void**)&vol, 64, N * sizeof(float));

    // sin along slow dim (Z), constant X-Y slices.
    // Pattern: sin(iz * PI / NZ * 10) → 10 periods over NZ=320 → 1 period per 32-sample block.
    for (int iz = 0; iz < NZ; ++iz) {
        float zval = sinf((float)iz * (float)M_PI / (float)NZ * 10.0f);
        long base = (long)iz * NY * NX;
        for (long i = 0; i < (long)NY * NX; ++i)
            vol[base + i] = zval;
    }

    double acc = 0.0;
    for (long i = 0; i < N; ++i) acc += (double)vol[i] * (double)vol[i];
    float global_rms = (float)sqrt(acc / (double)N);

    printf("Volume: %dx%dx%d (%.1f MB), RMS: %e\n\n", NX, NY, NZ, N*4.0f/1e6f, global_rms);

    int ldimx = NX, ldimxy = NX * NY;
    int nblocks = (NX/32) * (NY/32) * (NZ/32);
    long out_cap = (long)nblocks * 4L * WRLE_LDS_BYTES;

    float *d_raw, *d_wav;
    unsigned char *d_out;
    size_t *d_sizes;
    HIPCHECK(hipMalloc(&d_raw, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_wav, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out, out_cap));
    HIPCHECK(hipMalloc(&d_sizes, nblocks * sizeof(size_t)));
    HIPCHECK(hipMemcpy(d_raw, vol, N * sizeof(float), hipMemcpyHostToDevice));

    CvxCompress compressor;
    unsigned int* compressed;
    posix_memalign((void**)&compressed, 64, N * sizeof(float));
    float* vol_copy;
    posix_memalign((void**)&vol_copy, 64, N * sizeof(float));
    float* wav_cpu = (float*)malloc(N * sizeof(float));
    unsigned char* zl_buf = (unsigned char*)malloc(32 * 5);

    float scales[] = {1e-1f, 1e-2f, 1e-3f, 1e-4f};

    long meta_total = (long)nblocks * WRLE_META_PER_BLOCK;

    printf("%-10s %14s %14s %14s %14s %10s\n",
           "scale", "GPU(w/meta)", "GPU(no meta)", "CPU(zline)", "CPU(full)", "meta%%");
    printf("%-10s %14s %14s %14s %14s %10s\n",
           "-----", "-----------", "------------", "----------", "---------", "-----");

    for (float user_scale : scales) {
        float mulfac = 1.0f / (global_rms * user_scale);

        // (1) GPU fused z-line (block_sizes include 1024B meta per block)
        HIPCHECK(hipWaveletRLEFused(d_raw, d_out, d_sizes, mulfac, NX, NY, NZ, ldimx, ldimxy));
        HIPCHECK(hipDeviceSynchronize());

        std::vector<size_t> h_sizes(nblocks);
        HIPCHECK(hipMemcpy(h_sizes.data(), d_sizes, nblocks * sizeof(size_t), hipMemcpyDeviceToHost));
        long total_gpu = 0;
        for (int i = 0; i < nblocks; ++i) total_gpu += h_sizes[i];
        float cr_gpu = (float)(N * 4) / (float)total_gpu;
        float cr_gpu_no_meta = (float)(N * 4) / (float)(total_gpu - meta_total);
        float meta_pct = 100.0f * (float)meta_total / (float)total_gpu;

        // (2) CPU z-line ref (GPU wavelet → copy back → scalar z-line encode)
        HIPCHECK(hipMemcpy(d_wav, d_raw, N * sizeof(float), hipMemcpyDeviceToDevice));
        HIPCHECK((hipWaveletTransformBufferZYX<256, 2>(d_wav, NX, NY, NZ, 32, 32, 32, ldimx, ldimxy)));
        HIPCHECK(hipDeviceSynchronize());
        HIPCHECK(hipMemcpy(wav_cpu, d_wav, N * sizeof(float), hipMemcpyDeviceToHost));

        long total_cpu_zline = 0;
        for (int ibz = 0; ibz < NZ/32; ++ibz)
            for (int iby = 0; iby < NY/32; ++iby)
                for (int ibx = 0; ibx < NX/32; ++ibx) {
                    int bx0 = ibx*32, by0 = iby*32, bz0 = ibz*32;
                    for (int iy = 0; iy < 32; ++iy)
                        for (int ix = 0; ix < 32; ++ix) {
                            float zl[32];
                            for (int iz = 0; iz < 32; ++iz)
                                zl[iz] = wav_cpu[(bx0+ix) + (by0+iy)*NX + (bz0+iz)*(long)NX*NY];
                            total_cpu_zline += quantize_encode_zline(zl, mulfac, 32, zl_buf);
                        }
                }
        float cr_cpu_zline = (float)(N * 4) / (float)total_cpu_zline;

        // (3) CPU CvxCompress full-block
        memcpy(vol_copy, vol, N * sizeof(float));
        long comp_len = 0;
        compressor.Compress(user_scale, vol_copy, NX, NY, NZ, bx, by, bz, false, compressed, comp_len);
        float cr_cvx = (float)(N * 4) / (float)comp_len;

        printf("%-10.0e %11.1f:1 %11.1f:1 %11.1f:1 %11.1f:1 %8.1f%%\n",
               user_scale, cr_gpu, cr_gpu_no_meta, cr_cpu_zline, cr_cvx, meta_pct);
    }

    printf("\nNotes: GPU(w/meta) = fused kernel output including 1024B meta per block\n"
           "       GPU(no meta) = same minus metadata (should match CPU zline)\n"
           "       CPU(zline) = GPU wavelet coefficients + scalar z-line encode\n"
           "       CPU(full)  = CvxCompress (full-block RLE, global RMS)\n"
           "       meta%%     = fraction of compressed output that is metadata\n");

    free(vol); free(vol_copy); free(compressed); free(wav_cpu); free(zl_buf);
    HIPCHECK(hipFree(d_raw)); HIPCHECK(hipFree(d_wav));
    HIPCHECK(hipFree(d_out)); HIPCHECK(hipFree(d_sizes));
}

int main()
{
    printf("=== Fused Wavelet+RLE Kernel Test ===\n\n");

    float scales[] = {0.01f, 0.1f, 1.0f, 10.0f, 100.0f};
    struct { FusedLauncher fn; const char* tag; } variants[] = {
        { hipWaveletRLEFused,      "buffer" },
        { hipWaveletRLEFusedSaddr, "saddr"  },
    };

    printf("-- Correctness (fused) --\n");
    bool all_pass = true;
    for (auto& v : variants) {
        for (float s : scales) {
            printf("[%s] scale=%e (64^3): ", v.tag, s);
            if (!test_correctness(64, 64, 64, s, v.fn, v.tag)) all_pass = false;
        }
        printf("[%s] scale=1.0 (128^3): ", v.tag);
        if (!test_correctness(128, 128, 128, 1.0f, v.fn, v.tag)) all_pass = false;
    }

    printf("\n-- Correctness (compaction) --\n");
    for (float s : scales) {
        printf("scale=%e (64^3): ", s);
        if (!test_compaction(64, 64, 64, s)) all_pass = false;
    }
    printf("scale=1.0 (128^3): ");
    if (!test_compaction(128, 128, 128, 1.0f)) all_pass = false;

    if (!all_pass) { printf("\nCorrectness FAILED.\n"); return 1; }

    test_cr_sin();

    printf("\n-- Benchmark: random data (128^3, %.0f MB) --\n", 128.0*128*128*4/1e6);
    bench(128, 128, 128, 1.0f, 3, 10);
    printf("\n");

    printf("\n-- Benchmark: random data (512^3, %.0f MB) --\n", 512.0*512*512*4/1e6);
    for (float s : scales) {
        printf("scale = %e\n", s);
        bench(512, 512, 512, s, 3, 10);
        printf("\n");
    }

    // CR comparison: GPU (no meta) vs CPU (full-block) on sin(kx)sin(ky)sin(kz) k=40
    {
        int NX = 512, NY = 512, NZ = 512;
        int bx = 32, by = 32, bz = 32;
        long N = (long)NX * NY * NZ;
        int nblocks = (NX/32) * (NY/32) * (NZ/32);
        long meta_total = (long)nblocks * WRLE_META_PER_BLOCK;
        long out_cap = (long)nblocks * 4L * WRLE_LDS_BYTES;

        float* h_vol;
        posix_memalign((void**)&h_vol, 64, N * sizeof(float));
        for (long idx = 0; idx < N; ++idx) {
            int ix = idx % NX;
            int iy = (idx / NX) % NY;
            int iz = idx / ((long)NX * NY);
            float x = (float)ix / NX, y = (float)iy / NY, z = (float)iz / NZ;
            h_vol[idx] = sinf(40.0f * 6.2831853f * x) * sinf(40.0f * 6.2831853f * y)
                       * sinf(40.0f * 6.2831853f * z);
        }
        double acc = 0.0;
        for (long i = 0; i < N; ++i) acc += (double)h_vol[i] * (double)h_vol[i];
        float rms = (float)sqrt(acc / (double)N);

        float *d_raw;
        unsigned char *d_out;
        size_t *d_sizes;
        HIPCHECK(hipMalloc(&d_raw, N * sizeof(float)));
        HIPCHECK(hipMalloc(&d_out, out_cap));
        HIPCHECK(hipMalloc(&d_sizes, nblocks * sizeof(size_t)));
        HIPCHECK(hipMemcpy(d_raw, h_vol, N * sizeof(float), hipMemcpyHostToDevice));

        CvxCompress compressor;
        unsigned int* cpu_comp;
        posix_memalign((void**)&cpu_comp, 64, N * sizeof(float));
        float* vol_copy = (float*)malloc(N * sizeof(float));

        printf("\n-- CR comparison: sin(kx)sin(ky)sin(kz) k=40, 512^3, RMS=%.4e --\n", rms);
        printf("  %-12s %10s %10s %10s\n", "user_scale", "GPU(zline)", "CPU(full)", "overhead");
        printf("  %-12s %10s %10s %10s\n", "----------", "----------", "---------", "--------");

        float user_scales[] = {2e-4f, 5e-4f, 1e-3f, 2e-3f, 5e-3f, 1e-2f, 2e-2f, 5e-2f, 1e-1f};
        for (float us : user_scales) {
            float mulfac = 1.0f / (rms * us);

            HIPCHECK(hipWaveletRLEFused(d_raw, d_out, d_sizes, mulfac, NX, NY, NZ, NX, NX*NY));
            HIPCHECK(hipDeviceSynchronize());
            std::vector<size_t> h_sizes(nblocks);
            HIPCHECK(hipMemcpy(h_sizes.data(), d_sizes, nblocks*sizeof(size_t), hipMemcpyDeviceToHost));
            long total_gpu = 0;
            for (int i = 0; i < nblocks; ++i) total_gpu += h_sizes[i];
            float cr_gpu = (float)(N * 4) / (float)(total_gpu - meta_total);

            memcpy(vol_copy, h_vol, N * sizeof(float));
            long comp_len = 0;
            compressor.Compress(us, vol_copy, NX, NY, NZ, bx, by, bz, false, cpu_comp, comp_len);
            float cr_cpu = (float)(N * 4) / (float)comp_len;

            float overhead = (cr_cpu / cr_gpu - 1.0f) * 100.0f;
            printf("  %-12.0e %7.1f:1 %7.1f:1 %8.1f%%\n", us, cr_gpu, cr_cpu, overhead);
        }

        free(h_vol); free(vol_copy); free(cpu_comp);
        HIPCHECK(hipFree(d_raw)); HIPCHECK(hipFree(d_out)); HIPCHECK(hipFree(d_sizes));
    }

    // Sin data benchmark: forward vs inverse across CR 4-20 range
    {
        int NX = 512, NY = 512, NZ = 512;
        long N = (long)NX * NY * NZ;
        int nblocks = (NX/32) * (NY/32) * (NZ/32);
        long out_cap = (long)nblocks * 4L * WRLE_LDS_BYTES;

        float *d_raw, *d_wavelet;
        unsigned char *d_out, *d_compact;
        size_t *d_sizes, *d_offsets;
        void *d_scan_temp;

        HIPCHECK(hipMalloc(&d_raw, N * sizeof(float)));
        HIPCHECK(hipMalloc(&d_wavelet, N * sizeof(float)));
        HIPCHECK(hipMalloc(&d_out, out_cap));
        HIPCHECK(hipMalloc(&d_compact, out_cap));
        HIPCHECK(hipMalloc(&d_sizes, nblocks * sizeof(size_t)));
        HIPCHECK(hipMalloc(&d_offsets, nblocks * sizeof(size_t)));
        size_t scan_temp_bytes = 0;
        HIPCHECK(hipWaveletRLECompactScanTempSize(nblocks, &scan_temp_bytes));
        HIPCHECK(hipMalloc(&d_scan_temp, scan_temp_bytes));

        { int thr=256, blk=(N+thr-1)/thr;
          initSinKernel<<<blk,thr>>>(d_raw, NX, NY, NZ, 40.0f, 40.0f, 40.0f);
          HIPCHECK(hipDeviceSynchronize()); }

        float rms = gpuRMS(d_raw, N);
        printf("\n-- Benchmark: sin(kx)sin(ky)sin(kz) k=40, (512^3, %.0f MB, RMS=%.4e) --\n",
               N*4.0f/1e6, rms);
        printf("  %-12s %8s %8s %20s %20s %10s\n",
               "user_scale", "CR", "", "forward", "inverse", "inv/fwd");

        float user_scales[] = {2e-4f, 5e-4f, 1e-3f, 2e-3f, 5e-3f, 1e-2f, 2e-2f, 5e-2f, 1e-1f};
        for (float us : user_scales) {
            bench_sin(NX, NY, NZ, us, 3, 10,
                      d_raw, rms, d_wavelet, d_out, d_compact,
                      d_sizes, d_offsets, d_scan_temp, scan_temp_bytes);
        }

        HIPCHECK(hipFree(d_raw)); HIPCHECK(hipFree(d_wavelet));
        HIPCHECK(hipFree(d_out)); HIPCHECK(hipFree(d_compact));
        HIPCHECK(hipFree(d_sizes)); HIPCHECK(hipFree(d_offsets)); HIPCHECK(hipFree(d_scan_temp));
    }

    return 0;
}
