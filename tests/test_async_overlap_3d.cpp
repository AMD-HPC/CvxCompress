// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// Async / aux-stream behaviour of hipCompress on 3D data.
//
// hipCompress splits its pipeline across two streams: k1 (fused wavelet +
// quantize + significance) stays on user_stream because it reads the caller's
// live input, and everything downstream -- entropy coding, the size scan,
// compaction, D2H readback -- runs on plan->aux_stream via an internal event
// bridge. Nothing tested that split. test_compress_api_hip passes an aux stream
// but synchronizes before every check, so it would pass with the bridge removed;
// example_async_pipeline has the right shape but never decompresses, never
// compares, and returns 0 unconditionally.
//
// The four properties that actually matter, all on 3D volumes:
//
//   1. EQUIVALENCE   splitting across streams must not change a single output
//                    byte, for every 3D codec.
//   2. INPUT LIFETIME  d_input must be fully consumed by the time hipCompress
//                    returns control of user_stream. If any aux-stream stage
//                    ever reads d_input, the caller's next kernel races it.
//   3. OVERLAP       the aux tail must actually run concurrently with work the
//                    caller enqueues on user_stream -- that is the entire point
//                    of the split.
//   4. NO DEADLOCK   the bridge must not stall when aux_stream == user_stream
//                    (the degenerate case: an event recorded on a stream that
//                    then waits on itself).
//
// Property 3 is measured, not asserted at a threshold: how much overlap is
// achievable is a property of the GPU, not of the library. The test asserts
// only that concurrent execution is not *slower* than serial by more than a
// tolerance, and prints the efficiency so a regression is visible.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <hip/hip_runtime.h>
#include "hipCompress.h"

#define HIPCHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static int g_fail = 0;
static void check(bool ok, const char* what)
{
    printf("    %-56s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
}

// A field with real 3D structure and a sharp interface, so the codecs see a
// mix of dense and near-empty blocks rather than smooth noise. Sparsity is what
// drives the octree/two-level paths down their interesting branches.
__global__ void initKernel(float* d, int nx, int ny, int nz)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * nz;
    if (idx >= total) return;
    int iz = (int)(idx / ((size_t)nx * ny));
    int iy = (int)((idx - (size_t)iz * nx * ny) / nx);
    int ix = (int)(idx - (size_t)iz * nx * ny - (size_t)iy * nx);
    float x = (float)ix / nx, y = (float)iy / ny, z = (float)iz / nz;
    float v = sinf(12.0f * x) * cosf(9.0f * y) * sinf(15.0f * z)
            + 0.3f * sinf(40.0f * x + 30.0f * z);
    // zero out a quadrant so a good fraction of blocks are entirely empty
    if (x > 0.6f && y > 0.6f) v = 0.0f;
    d[idx] = v;
}

// Poisons the whole buffer. Used to prove d_input is dead after hipCompress
// hands user_stream back.
__global__ void clobberKernel(float* d, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = 1e30f;
}

// Stand-in for caller work on user_stream: a 3D 7-point stencil, i.e. the same
// bandwidth-heavy shape as a wave-prop kernel, so the overlap measurement is
// against a realistic competitor rather than an idle GPU.
__global__ void stencilKernel(float* out, const float* in, int nx, int ny, int nz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix < 1 || iy < 1 || iz < 1 || ix >= nx-1 || iy >= ny-1 || iz >= nz-1)
        return;
    size_t c = (size_t)ix + (size_t)iy * nx + (size_t)iz * nx * ny;
    size_t sy = nx, sz = (size_t)nx * ny;
    out[c] = 0.5f * in[c]
           + 0.0833f * (in[c-1] + in[c+1] + in[c-sy] + in[c+sy]
                        + in[c-sz] + in[c+sz]);
}

struct Codec { hipCompressKernel k; const char* name; };
static const Codec CODECS[] = {
    { HIP_COMPRESS_KERNEL_OCTREE,   "octree"   },
    { HIP_COMPRESS_KERNEL_TWOLEVEL, "twolevel" },
};
static const int NCODEC = sizeof(CODECS) / sizeof(CODECS[0]);

static const int N = 256;                       // 3D, 256^3 = 16.8 M samples
static const size_t TOTAL = (size_t)N * N * N;
static const float SCALE = 5e-2f;

// Compress d_in through a plan whose aux_stream is `aux`, then decompress.
// Returns the compressed length; fills h_out with the reconstruction.
static long compressRoundTrip(hipCompressKernel kern, hipStream_t aux_for_plan,
                              bool aux_is_user, hipStream_t user,
                              const float* d_in, float* h_out, float* d_wav,
                              float* d_rec, unsigned char* d_comp)
{
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, N, N, N,
                                   aux_is_user ? user : aux_for_plan, kern));
    HIPCHECK(hipCopyToWaveletLayout(d_in, N, (size_t)N * N, 0, 0, 0, N, N, N,
                                    d_wav, plan->d_rms, plan, user));
    HIPCHECK(hipCompress(SCALE, plan->d_rms, d_wav, d_comp, plan, user));
    long len = 0; float cr = 0;
    HIPCHECK(hipCompressSynchronize(plan, &len, &cr));
    HIPCHECK(hipDecompress(d_comp, d_rec, plan, user));
    HIPCHECK(hipStreamSynchronize(user));
    HIPCHECK(hipMemcpy(h_out, d_rec, TOTAL * sizeof(float),
                       hipMemcpyDeviceToHost));
    HIPCHECK(hipCompressDestroyPlan(plan));
    return len;
}

// ---------------------------------------------------------------------------
// 1 + 4. Splitting across streams changes nothing; the degenerate case works.
// ---------------------------------------------------------------------------
static void testEquivalence(hipStream_t user, hipStream_t aux,
                            const float* d_in, float* d_wav, float* d_rec,
                            unsigned char* d_comp)
{
    printf("  Test 1: split vs serial equivalence (3D %d^3)\n", N);
    std::vector<float> a(TOTAL), b(TOTAL);
    for (int c = 0; c < NCODEC; ++c) {
        // aux == user: the bridge degenerates to a stream waiting on itself.
        // Must not deadlock (property 4) and defines the reference bytes.
        long len_ser = compressRoundTrip(CODECS[c].k, aux, true, user,
                                         d_in, a.data(), d_wav, d_rec, d_comp);
        // distinct aux: the real split.
        long len_spl = compressRoundTrip(CODECS[c].k, aux, false, user,
                                         d_in, b.data(), d_wav, d_rec, d_comp);
        char msg[128];
        snprintf(msg, sizeof msg, "%s: length %ld == %ld", CODECS[c].name,
                 len_ser, len_spl);
        check(len_ser == len_spl && len_ser > 0, msg);
        snprintf(msg, sizeof msg, "%s: reconstruction bit-identical",
                 CODECS[c].name);
        check(memcmp(a.data(), b.data(), TOTAL * sizeof(float)) == 0, msg);
    }
}

// ---------------------------------------------------------------------------
// 2. d_input must be dead once hipCompress returns control of user_stream.
// ---------------------------------------------------------------------------
static void testInputLifetime(hipStream_t user, hipStream_t aux,
                              const float* d_in, float* d_wav, float* d_rec,
                              unsigned char* d_comp)
{
    printf("  Test 2: d_input consumed before caller's next kernel\n");
    std::vector<float> ref(TOTAL), got(TOTAL);
    int threads = 256;
    size_t wav_total = TOTAL;   // 256 is 32-divisible, so wavelet dims == N
    for (int c = 0; c < NCODEC; ++c) {
        compressRoundTrip(CODECS[c].k, aux, false, user, d_in, ref.data(),
                          d_wav, d_rec, d_comp);

        // Same compress, but the instant hipCompress hands user_stream back we
        // enqueue a kernel that destroys the wavelet buffer it was reading. If
        // any aux-stream stage still needed d_input, this corrupts the output.
        hipCompressPlan* plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, aux, CODECS[c].k));
        HIPCHECK(hipCopyToWaveletLayout(d_in, N, (size_t)N * N, 0, 0, 0,
                                        N, N, N, d_wav, plan->d_rms, plan, user));
        HIPCHECK(hipCompress(SCALE, plan->d_rms, d_wav, d_comp, plan, user));
        clobberKernel<<<(wav_total + threads - 1) / threads, threads, 0, user>>>(
            d_wav, wav_total);
        long len = 0; float cr = 0;
        HIPCHECK(hipCompressSynchronize(plan, &len, &cr));
        HIPCHECK(hipDecompress(d_comp, d_rec, plan, user));
        HIPCHECK(hipStreamSynchronize(user));
        HIPCHECK(hipMemcpy(got.data(), d_rec, TOTAL * sizeof(float),
                           hipMemcpyDeviceToHost));
        HIPCHECK(hipCompressDestroyPlan(plan));

        char msg[128];
        snprintf(msg, sizeof msg, "%s: output survives clobber of d_input",
                 CODECS[c].name);
        check(memcmp(ref.data(), got.data(), TOTAL * sizeof(float)) == 0, msg);
    }
}

// ---------------------------------------------------------------------------
// 3. The aux tail must actually overlap caller work on user_stream.
// ---------------------------------------------------------------------------
// Parameters are not named `e`: the HIPCHECK macro declares its own
// `hipError_t e` and would shadow it.
static float timeMs(hipEvent_t beg, hipEvent_t end)
{
    float ms = 0; HIPCHECK(hipEventElapsedTime(&ms, beg, end)); return ms;
}

static void testOverlap(hipStream_t user, hipStream_t aux, const float* d_in,
                        float* d_wav, float* d_rec, unsigned char* d_comp)
{
    printf("  Test 3: aux tail overlaps caller work on user_stream\n");
    const int STENCIL_ITERS = 12;
    dim3 blk(64, 4, 1);
    dim3 grd((N + blk.x - 1) / blk.x, (N + blk.y - 1) / blk.y, N);

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0)); HIPCHECK(hipEventCreate(&t1));

    auto runStencil = [&]() {
        for (int i = 0; i < STENCIL_ITERS; ++i)
            stencilKernel<<<grd, blk, 0, user>>>(d_rec, d_wav, N, N, N);
    };

    printf("    %-10s %10s %10s %10s %10s\n",
           "codec", "stencil", "compress", "together", "hidden");
    for (int c = 0; c < NCODEC; ++c) {
        hipCompressPlan* plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, aux, CODECS[c].k));
        HIPCHECK(hipCopyToWaveletLayout(d_in, N, (size_t)N * N, 0, 0, 0,
                                        N, N, N, d_wav, plan->d_rms, plan, user));
        HIPCHECK(hipStreamSynchronize(user));

        // warm up both paths so JIT / first-touch costs are not in the numbers
        for (int w = 0; w < 2; ++w) {
            runStencil();
            HIPCHECK(hipCompress(SCALE, plan->d_rms, d_wav, d_comp, plan, user));
            long l; float r; HIPCHECK(hipCompressSynchronize(plan, &l, &r));
            HIPCHECK(hipStreamSynchronize(user));
        }

        // (a) stencil alone
        HIPCHECK(hipEventRecord(t0, user)); runStencil();
        HIPCHECK(hipEventRecord(t1, user)); HIPCHECK(hipEventSynchronize(t1));
        float t_bg = timeMs(t0, t1);

        // (b) compress alone, host-timed: the chain ends on aux, not user
        HIPCHECK(hipDeviceSynchronize());
        HIPCHECK(hipEventRecord(t0, user));
        HIPCHECK(hipCompress(SCALE, plan->d_rms, d_wav, d_comp, plan, user));
        long len = 0; float cr = 0;
        HIPCHECK(hipCompressSynchronize(plan, &len, &cr));
        HIPCHECK(hipEventRecord(t1, user)); HIPCHECK(hipEventSynchronize(t1));
        float t_c = timeMs(t0, t1);

        // (c) both: compress first, so its aux tail is in flight while the
        //     stencil runs on user_stream. This is exactly the solver's shape.
        HIPCHECK(hipDeviceSynchronize());
        HIPCHECK(hipEventRecord(t0, user));
        HIPCHECK(hipCompress(SCALE, plan->d_rms, d_wav, d_comp, plan, user));
        runStencil();
        HIPCHECK(hipCompressSynchronize(plan, &len, &cr));
        HIPCHECK(hipStreamSynchronize(user));
        HIPCHECK(hipEventRecord(t1, user)); HIPCHECK(hipEventSynchronize(t1));
        float t_both = timeMs(t0, t1);

        HIPCHECK(hipCompressDestroyPlan(plan));

        float serial = t_bg + t_c;
        float hidden = serial > 0 ? 100.0f * (serial - t_both) / serial : 0.0f;
        printf("    %-10s %9.3f %9.3f %9.3f %9.1f%%\n",
               CODECS[c].name, t_bg, t_c, t_both, hidden);

        // Not a threshold on how much overlaps -- that is a property of the
        // GPU. Only that concurrency never costs more than doing it serially,
        // with 15% of slack for contention and launch jitter.
        char msg[128];
        snprintf(msg, sizeof msg, "%s: concurrent not slower than serial",
                 CODECS[c].name);
        check(t_both <= serial * 1.15f, msg);
    }
    HIPCHECK(hipEventDestroy(t0)); HIPCHECK(hipEventDestroy(t1));
}

int main()
{
    printf("hipCompress async / aux-stream tests on 3D data (%d^3)\n\n", N);

    hipStream_t user, aux;
    HIPCHECK(hipStreamCreate(&user));
    HIPCHECK(hipStreamCreateWithFlags(&aux, hipStreamNonBlocking));

    float *d_in, *d_wav, *d_rec;
    HIPCHECK(hipMalloc(&d_in,  TOTAL * sizeof(float)));
    HIPCHECK(hipMalloc(&d_wav, TOTAL * sizeof(float)));
    HIPCHECK(hipMalloc(&d_rec, TOTAL * sizeof(float)));

    // size the output buffer from the largest max-output over the codecs
    size_t maxout = 0;
    for (int c = 0; c < NCODEC; ++c) {
        hipCompressPlan* p = nullptr;
        HIPCHECK(hipCompressCreatePlan(&p, N, N, N, aux, CODECS[c].k));
        size_t m = 0; HIPCHECK(hipCompressMaxOutputSize(p, &m));
        if (m > maxout) maxout = m;
        HIPCHECK(hipCompressDestroyPlan(p));
    }
    unsigned char* d_comp;
    HIPCHECK(hipMalloc(&d_comp, maxout));

    int threads = 256;
    initKernel<<<(TOTAL + threads - 1) / threads, threads, 0, user>>>(
        d_in, N, N, N);
    HIPCHECK(hipStreamSynchronize(user));

    testEquivalence(user, aux, d_in, d_wav, d_rec, d_comp);
    testInputLifetime(user, aux, d_in, d_wav, d_rec, d_comp);
    testOverlap(user, aux, d_in, d_wav, d_rec, d_comp);

    (void)hipFree(d_in); (void)hipFree(d_wav); (void)hipFree(d_rec);
    (void)hipFree(d_comp);
    (void)hipStreamDestroy(aux); (void)hipStreamDestroy(user);

    printf("\n%s\n", g_fail ? "OVERALL: FAIL" : "OVERALL: PASS");
    return g_fail;
}
