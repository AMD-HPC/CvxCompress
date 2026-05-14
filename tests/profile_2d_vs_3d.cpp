// Isolated decompress-only benchmark for profiling 2D vs 3D inverse kernels.
// Compresses once, then runs decompress N times for profiler capture.

#include <cstdio>
#include <cstdlib>
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

__global__ void initSin2D(float* data, int nx, int ny, float kx, float ky)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;
    int iy = idx / nx, ix = idx % nx;
    data[idx] = sinf(kx * (float)ix / nx) * sinf(ky * (float)iy / ny);
}

__global__ void initSin3D(float* data, int nx, int ny, int nz, float kx, float ky, float kz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;
    int iz = idx / (nx * ny);
    int rem = idx - iz * nx * ny;
    int iy = rem / nx, ix = rem % nx;
    data[idx] = sinf(kx * (float)ix / nx) * sinf(ky * (float)iy / ny) * sinf(kz * (float)iz / nz);
}

int main(int argc, char** argv)
{
    int iters = 20;
    if (argc > 1) iters = atoi(argv[1]);

    const float scale = 5e-2f;

    // ---- 3D: 256^3 ----
    {
        const int N = 256;
        long total = (long)N * N * N;
        hipCompressPlan* plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&plan, N, N, N, 0));

        float *d_in, *d_out;
        unsigned char* d_comp;
        HIPCHECK(hipMalloc(&d_in, total * sizeof(float)));
        HIPCHECK(hipMalloc(&d_out, total * sizeof(float)));
        size_t comp_sz; HIPCHECK(hipCompressMaxOutputSize(plan, &comp_sz));
        HIPCHECK(hipMalloc(&d_comp, comp_sz));

        initSin3D<<<(total+255)/256, 256>>>(d_in, N, N, N, 20.f, 20.f, 20.f);
        HIPCHECK(hipDeviceSynchronize());

        long len; float cr;
        HIPCHECK(hipCompress(scale, nullptr, d_in, d_comp, plan, 0));
        HIPCHECK(hipCompressSynchronize(plan, &len, &cr));
        printf("3D 256^3: CR=%.1f, %ld bytes\n", cr, len);

        // warmup
        for (int i = 0; i < 5; i++) {
            HIPCHECK(hipDecompress(d_comp, d_out, plan, 0));
            HIPCHECK(hipDeviceSynchronize());
        }

        // timed region for profiler
        printf("3D decompress: %d iterations\n", iters);
        for (int i = 0; i < iters; i++) {
            HIPCHECK(hipDecompress(d_comp, d_out, plan, 0));
            HIPCHECK(hipDeviceSynchronize());
        }

        hipFree(d_in); hipFree(d_out); hipFree(d_comp);
        hipCompressDestroyPlan(plan);
    }

    // ---- 2D: 4064x4064 (similar total floats) ----
    {
        const int NX = 4064, NY = 4064;
        long total = (long)NX * NY;
        hipCompressPlan* plan = nullptr;
        HIPCHECK(hipCompressCreatePlan(&plan, NX, NY, 1, 0));

        float *d_in, *d_out;
        unsigned char* d_comp;
        HIPCHECK(hipMalloc(&d_in, total * sizeof(float)));
        HIPCHECK(hipMalloc(&d_out, total * sizeof(float)));
        size_t comp_sz; HIPCHECK(hipCompressMaxOutputSize(plan, &comp_sz));
        HIPCHECK(hipMalloc(&d_comp, comp_sz));

        initSin2D<<<(total+255)/256, 256>>>(d_in, NX, NY, 20.f, 20.f);
        HIPCHECK(hipDeviceSynchronize());

        long len; float cr;
        HIPCHECK(hipCompress(scale, nullptr, d_in, d_comp, plan, 0));
        HIPCHECK(hipCompressSynchronize(plan, &len, &cr));
        printf("2D %dx%d: CR=%.1f, %ld bytes\n", NX, NY, cr, len);

        for (int i = 0; i < 5; i++) {
            HIPCHECK(hipDecompress(d_comp, d_out, plan, 0));
            HIPCHECK(hipDeviceSynchronize());
        }

        printf("2D decompress: %d iterations\n", iters);
        for (int i = 0; i < iters; i++) {
            HIPCHECK(hipDecompress(d_comp, d_out, plan, 0));
            HIPCHECK(hipDeviceSynchronize());
        }

        hipFree(d_in); hipFree(d_out); hipFree(d_comp);
        hipCompressDestroyPlan(plan);
    }

    printf("Done.\n");
    return 0;
}
