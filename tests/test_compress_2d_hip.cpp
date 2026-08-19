// 2D compression round-trip test.
// Tests hipCompress API with nz=1 (2D mode):
//   1. Plan lifecycle with nz=1
//   2. Compress/decompress round-trip on a 2D sinusoidal field
//   3. Copy-to/from wavelet layout with ez=1
//   4. Various 2D grid sizes (including non-32-aligned extraction)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#include "hipCompress.h"

#define HIPCHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__global__ void initSin2DKernel(float* data, int nx, int ny, float kx, float ky)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny;
    if (idx >= total) return;
    int iy = idx / nx;
    int ix = idx - iy * nx;
    float x = (float)ix / (float)nx;
    float y = (float)iy / (float)ny;
    data[idx] = sinf(kx * x) * sinf(ky * y);
}

static float hostRMS(const float* data, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)data[i] * (double)data[i];
    return (float)sqrt(sum / n);
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

static hipError_t deviceRMS2D(const float* d_input, int nx, int ny,
                               double* d_rms, hipCompressPlan* plan,
                               hipStream_t s = 0)
{
    return hipCopyToWaveletLayout(
        d_input, nx, nx,
        0, 0, 0, nx, ny, 1,
        nullptr, d_rms, plan, s);
}

static hipError_t compressWithAutoRMS2D(float scale, const float* d_input,
                                         unsigned char* d_output, long* len,
                                         float* cr, hipCompressPlan* plan,
                                         hipStream_t s = 0)
{
    int nx = plan->nx, ny = plan->ny;
    hipError_t err = deviceRMS2D(d_input, nx, ny, plan->d_rms, plan, s);
    if (err != hipSuccess) return err;
    err = hipCompress(scale, plan->d_rms, d_input, d_output, plan, s);
    if (err != hipSuccess) return err;
    return hipCompressSynchronize(plan, len, cr);
}

static bool test_plan_lifecycle_2d()
{
    printf("Test 1: 2D plan lifecycle\n");
    hipCompressPlan* plan = nullptr;

    hipError_t err = hipCompressCreatePlan(&plan, 128, 128, 1, 0);
    if (err != hipSuccess || !plan) {
        printf("  FAIL: create 128x128 2D plan: %s\n", hipGetErrorString(err));
        return false;
    }
    if (!plan->is_2d) { printf("  FAIL: is_2d not set\n"); hipCompressDestroyPlan(plan); return false; }
    if (plan->num_blocks != 4 * 4) {
        printf("  FAIL: num_blocks=%d expected=%d\n", plan->num_blocks, 16);
        hipCompressDestroyPlan(plan); return false;
    }
    printf("  128x128: num_blocks=%d: PASS\n", plan->num_blocks);

    size_t max_sz = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &max_sz));
    printf("  MaxOutputSize=%zu\n", max_sz);
    hipCompressDestroyPlan(plan);

    // Reject nx not multiple of 32
    plan = nullptr;
    err = hipCompressCreatePlan(&plan, 100, 128, 1, 0);
    if (err == hipSuccess) { printf("  FAIL: should reject nx=100\n"); hipCompressDestroyPlan(plan); return false; }
    printf("  reject non-32-multiple: PASS\n");

    // 256x256
    plan = nullptr;
    err = hipCompressCreatePlan(&plan, 256, 256, 1, 0);
    if (err != hipSuccess) { printf("  FAIL: create 256x256\n"); return false; }
    if (plan->num_blocks != 8 * 8) { printf("  FAIL: num_blocks\n"); hipCompressDestroyPlan(plan); return false; }
    printf("  256x256: num_blocks=%d: PASS\n", plan->num_blocks);
    hipCompressDestroyPlan(plan);

    return true;
}

static bool test_round_trip_2d()
{
    printf("Test 2: 2D compress/decompress round-trip\n");
    const int NX = 128, NY = 128, total = NX * NY;
    const float scale = 5e-2f;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, NX, NY, 1, 0));

    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));

    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSin2DKernel<<<blocks, threads>>>(d_input, NX, NY, 20.0f, 20.0f);
    HIPCHECK(hipDeviceSynchronize());

    long compressed_length = 0;
    float cr = 0;
    HIPCHECK(compressWithAutoRMS2D(scale, d_input, d_compressed, &compressed_length, &cr, plan));
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

    bool pass = (max_err < rms) && (cr > 1.0f);
    printf("  round-trip: %s\n", pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_round_trip_2d_large()
{
    printf("Test 3: 2D round-trip 512x512\n");
    const int NX = 512, NY = 512, total = NX * NY;
    const float scale = 5e-2f;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, NX, NY, 1, 0));

    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_compressed = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, total * sizeof(float)));

    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_compressed, comp_size));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSin2DKernel<<<blocks, threads>>>(d_input, NX, NY, 40.0f, 40.0f);
    HIPCHECK(hipDeviceSynchronize());

    long compressed_length = 0;
    float cr = 0;
    HIPCHECK(compressWithAutoRMS2D(scale, d_input, d_compressed, &compressed_length, &cr, plan));
    printf("  compress: CR=%.2f, %ld bytes\n", cr, compressed_length);

    HIPCHECK(hipDecompress(d_compressed, d_output, plan, 0));

    std::vector<float> h_in(total), h_out(total);
    HIPCHECK(hipMemcpy(h_in.data(), d_input, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_out.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), total);
    float max_err = maxAbsError(h_in.data(), h_out.data(), total);
    printf("  decompress: max_err=%.6e, rms=%.6e, rel_max_err=%.6e\n", max_err, rms, max_err / rms);

    bool pass = (max_err < rms) && (cr > 1.0f);
    printf("  round-trip: %s\n", pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_output); hipFree(d_compressed);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_copy_to_from_2d()
{
    printf("Test 4: 2D copy-to/copy-from wavelet layout\n");
    const int EX = 100, EY = 80;
    const int WNX = ((EX + 31) / 32) * 32;   // 128
    const int WNY = ((EY + 31) / 32) * 32;   // 96
    const int src_total = EX * EY;
    const int wav_total = WNX * WNY;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, WNX, WNY, 1, 0));

    float* d_src = nullptr;
    float* d_wav = nullptr;
    float* d_dst = nullptr;
    double* d_rms = nullptr;
    HIPCHECK(hipMalloc(&d_src, EX * EY * sizeof(float)));
    HIPCHECK(hipMalloc(&d_wav, wav_total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_dst, EX * EY * sizeof(float)));
    HIPCHECK(hipMalloc(&d_rms, sizeof(double)));

    int threads = 256, blocks = (src_total + threads - 1) / threads;
    initSin2DKernel<<<blocks, threads>>>(d_src, EX, EY, 10.0f, 10.0f);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipCopyToWaveletLayout(
        d_src, EX, EX,
        0, 0, 0, EX, EY, 1,
        d_wav, d_rms, plan, 0));

    HIPCHECK(hipCopyFromWaveletLayout(
        d_wav, d_dst, EX, EX,
        0, 0, 0, EX, EY, 1,
        plan, 0));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_src(src_total), h_dst(src_total);
    HIPCHECK(hipMemcpy(h_src.data(), d_src, src_total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_dst.data(), d_dst, src_total * sizeof(float), hipMemcpyDeviceToHost));

    float max_err = maxAbsError(h_src.data(), h_dst.data(), src_total);
    printf("  copy round-trip max_err=%.6e\n", max_err);

    double h_rms;
    HIPCHECK(hipMemcpy(&h_rms, d_rms, sizeof(double), hipMemcpyDeviceToHost));
    float cpu_rms = hostRMS(h_src.data(), src_total);
    float rms_err = fabsf((float)h_rms - cpu_rms);
    printf("  GPU RMS=%.6e, CPU RMS=%.6e, diff=%.6e\n", (float)h_rms, cpu_rms, rms_err);

    bool pass = (max_err < 1e-6f) && (rms_err < 1e-4f);
    printf("  copy-to/from 2D: %s\n", pass ? "PASS" : "FAIL");

    hipFree(d_src); hipFree(d_wav); hipFree(d_dst); hipFree(d_rms);
    hipCompressDestroyPlan(plan);
    return pass;
}

static bool test_round_trip_with_copy_2d()
{
    printf("Test 5: 2D full pipeline (copy-to → compress → decompress → copy-from)\n");
    const int EX = 100, EY = 80;
    const int WNX = ((EX + 31) / 32) * 32;
    const int WNY = ((EY + 31) / 32) * 32;
    const float scale = 5e-2f;

    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, WNX, WNY, 1, 0));

    float* d_src = nullptr;
    float* d_wav = nullptr;
    float* d_dec = nullptr;
    float* d_dst = nullptr;
    unsigned char* d_comp = nullptr;
    HIPCHECK(hipMalloc(&d_src, EX * EY * sizeof(float)));
    HIPCHECK(hipMalloc(&d_wav, WNX * WNY * sizeof(float)));
    HIPCHECK(hipMalloc(&d_dec, WNX * WNY * sizeof(float)));
    HIPCHECK(hipMalloc(&d_dst, EX * EY * sizeof(float)));
    size_t comp_size = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_size));
    HIPCHECK(hipMalloc(&d_comp, comp_size));

    int threads = 256, blocks = (EX * EY + threads - 1) / threads;
    initSin2DKernel<<<blocks, threads>>>(d_src, EX, EY, 15.0f, 15.0f);
    HIPCHECK(hipDeviceSynchronize());

    // Copy to wavelet layout + RMS
    HIPCHECK(hipCopyToWaveletLayout(
        d_src, EX, EX, 0, 0, 0, EX, EY, 1,
        d_wav, plan->d_rms, plan, 0));

    // Compress
    long compressed_length = 0;
    float cr = 0;
    hipError_t err = hipCompress(scale, plan->d_rms, d_wav, d_comp, plan, 0);
    HIPCHECK(err);
    HIPCHECK(hipCompressSynchronize(plan, &compressed_length, &cr));
    printf("  compress: CR=%.2f, %ld bytes\n", cr, compressed_length);

    // Decompress
    HIPCHECK(hipDecompress(d_comp, d_dec, plan, 0));

    // Copy from wavelet layout
    HIPCHECK(hipCopyFromWaveletLayout(
        d_dec, d_dst, EX, EX, 0, 0, 0, EX, EY, 1, plan, 0));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_src(EX * EY), h_dst(EX * EY);
    HIPCHECK(hipMemcpy(h_src.data(), d_src, EX * EY * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_dst.data(), d_dst, EX * EY * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_src.data(), EX * EY);
    float max_err = maxAbsError(h_src.data(), h_dst.data(), EX * EY);
    printf("  max_err=%.6e, rms=%.6e, rel=%.6e\n", max_err, rms, max_err / rms);

    bool pass = (max_err < rms) && (cr > 1.0f);
    printf("  full pipeline 2D: %s\n", pass ? "PASS" : "FAIL");

    hipFree(d_src); hipFree(d_wav); hipFree(d_dec); hipFree(d_dst); hipFree(d_comp);
    hipCompressDestroyPlan(plan);
    return pass;
}

// 2D quadtree significance coder: round-trip correctness + bit-exact match to
// the RLE 2D path (both use the same transform/quantizer, so at equal scale the
// reconstructed field must be identical) + reported CR for each codec.
static bool test_quadtree_2d_round_trip()
{
    printf("Test 6: 2D quadtree codec round-trip + RLE equivalence\n");
    const int NX = 256, NY = 256, total = NX * NY;
    const float scale = 5e-2f;

    hipCompressPlan* p_rle = nullptr;
    hipCompressPlan* p_qt  = nullptr;
    HIPCHECK(hipCompressCreatePlan(&p_rle, NX, NY, 1, 0, HIP_COMPRESS_KERNEL_ZLINE));
    hipError_t err = hipCompressCreatePlan(&p_qt, NX, NY, 1, 0, HIP_COMPRESS_KERNEL_QUADTREE);
    if (err != hipSuccess || !p_qt) {
        printf("  FAIL: create quadtree plan: %s\n", hipGetErrorString(err));
        return false;
    }

    // Quadtree must be rejected for 3D dims.
    hipCompressPlan* p_bad = nullptr;
    if (hipCompressCreatePlan(&p_bad, 64, 64, 64, 0, HIP_COMPRESS_KERNEL_QUADTREE) == hipSuccess) {
        printf("  FAIL: quadtree accepted for 3D\n");
        hipCompressDestroyPlan(p_bad);
        return false;
    }
    printf("  reject quadtree for 3D: PASS\n");

    float* d_input = nullptr;
    float* d_out_rle = nullptr;
    float* d_out_qt = nullptr;
    unsigned char* d_comp_rle = nullptr;
    unsigned char* d_comp_qt = nullptr;
    HIPCHECK(hipMalloc(&d_input, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out_rle, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out_qt, total * sizeof(float)));
    size_t comp_rle = 0, comp_qt = 0;
    HIPCHECK(hipCompressMaxOutputSize(p_rle, &comp_rle));
    HIPCHECK(hipCompressMaxOutputSize(p_qt, &comp_qt));
    HIPCHECK(hipMalloc(&d_comp_rle, comp_rle));
    HIPCHECK(hipMalloc(&d_comp_qt, comp_qt));

    int threads = 256, blocks = (total + threads - 1) / threads;
    initSin2DKernel<<<blocks, threads>>>(d_input, NX, NY, 24.0f, 24.0f);
    HIPCHECK(hipDeviceSynchronize());

    long len_rle = 0, len_qt = 0;
    float cr_rle = 0, cr_qt = 0;
    HIPCHECK(compressWithAutoRMS2D(scale, d_input, d_comp_rle, &len_rle, &cr_rle, p_rle));
    HIPCHECK(compressWithAutoRMS2D(scale, d_input, d_comp_qt,  &len_qt,  &cr_qt,  p_qt));
    printf("  RLE      : CR=%.2f, %ld bytes\n", cr_rle, len_rle);
    printf("  quadtree : CR=%.2f, %ld bytes\n", cr_qt, len_qt);

    HIPCHECK(hipDecompress(d_comp_rle, d_out_rle, p_rle, 0));
    HIPCHECK(hipDecompress(d_comp_qt,  d_out_qt,  p_qt,  0));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> h_in(total), h_rle(total), h_qt(total);
    HIPCHECK(hipMemcpy(h_in.data(),  d_input,   total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_rle.data(), d_out_rle, total * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_qt.data(),  d_out_qt,  total * sizeof(float), hipMemcpyDeviceToHost));

    float rms = hostRMS(h_in.data(), total);
    float qt_err = maxAbsError(h_in.data(), h_qt.data(), total);
    float vs_rle = maxAbsError(h_rle.data(), h_qt.data(), total);
    printf("  quadtree vs input: max_err=%.6e (rms=%.6e, rel=%.6e)\n", qt_err, rms, qt_err / rms);
    printf("  quadtree vs RLE  : max_err=%.6e (float-epsilon of RLE path)\n", vs_rle);

    // The quadtree path shares the RLE transform/quantizer, so it must match the
    // RLE reconstruction to within float rounding (the two inverse kernels are the
    // same algorithm but the compiler contracts FP independently) -- i.e. far
    // below the quantization error.  Require the codecs agree to <1e-4 abs.
    bool pass = (qt_err < rms) && (cr_qt > 1.0f) && (vs_rle < 1e-4f);
    printf("  quadtree 2D: %s\n", pass ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_out_rle); hipFree(d_out_qt);
    hipFree(d_comp_rle); hipFree(d_comp_qt);
    hipCompressDestroyPlan(p_rle);
    hipCompressDestroyPlan(p_qt);
    return pass;
}

int main()
{
    printf("=== 2D Compression Tests ===\n\n");
    fflush(stdout);

    int passed = 0, total = 0;

    total++; if (test_plan_lifecycle_2d()) passed++;
    printf("\n");
    total++; if (test_round_trip_2d()) passed++;
    printf("\n");
    total++; if (test_round_trip_2d_large()) passed++;
    printf("\n");
    total++; if (test_copy_to_from_2d()) passed++;
    printf("\n");
    total++; if (test_round_trip_with_copy_2d()) passed++;
    printf("\n");
    total++; if (test_quadtree_2d_round_trip()) passed++;
    printf("\n");

    printf("=== Results: %d/%d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
