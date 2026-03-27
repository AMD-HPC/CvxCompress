// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// Performance benchmark: quantize+RLE encode kernel variants.
// Tests Option A (direct-to-global), B (scratch buf), C (two-pass compacted).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocprim/block/block_scan.hpp>
#include "Run_Length_Escape_Codes.hxx"

#define HIPCHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

using qrle_float4_vec = __attribute__((__vector_size__(4 * sizeof(float)))) float;

static constexpr int QRLE_BUF = 160;

// ===========================================================================
// Unified branchless encode template.
// WRITE=true: stores bytes to dst.  WRITE=false: count-only (dst unused).
// Single template ensures identical codegen for ival computation.
// ===========================================================================

template<bool WRITE, bool SAFE = false>
__device__ __forceinline__
int qrle_zline(const qrle_float4_vec* planes, int x_off, float scale,
               unsigned char* dst)
{
    int bp = 0, rle = 0;
    #pragma unroll
    for (int z = 0; z < 32; ++z) {
        float fval = scale * planes[z][x_off];
        int ival = (int)fval;
        if (ival == 0) { ++rle; } else {
            if constexpr (!SAFE) {
                bool r1 = (rle == 1), rs = (rle >= 2) & (rle < 256);
                int rb = (r1 ? 1 : (rs ? 2 : 4)) & -(rle > 0);
                if constexpr (WRITE) {
                    dst[bp]   = r1 ? (unsigned char)0 : rs ? (unsigned char)(RLESC1&0xFF) : (unsigned char)(RLESC3&0xFF);
                    dst[bp+1] = rs ? (unsigned char)rle : (unsigned char)(rle&0xFF);
                    dst[bp+2] = (unsigned char)((rle>>8)&0xFF);
                    dst[bp+3] = (unsigned char)((rle>>16)&0xFF);
                }
                bp += rb;
            } else {
                if (rle == 1) {
                    dst[bp] = 0;
                    bp += 1;
                } else if (rle >= 2 && rle < 256) {
                    dst[bp] = (unsigned char)(RLESC1 & 0xFF);
                    dst[bp+1] = (unsigned char)rle;
                    bp += 2;
                } else if (rle >= 256) {
                    dst[bp] = (unsigned char)(RLESC3 & 0xFF);
                    dst[bp+1] = (unsigned char)(rle & 0xFF);
                    dst[bp+2] = (unsigned char)((rle >> 8) & 0xFF);
                    dst[bp+3] = (unsigned char)((rle >> 16) & 0xFF);
                    bp += 4;
                }
            }
            rle = 0;

            bool ib = (ival>VLESC2)&(ival<RLESC3), i16 = (ival>=-32768)&(ival<=32767);
            bool i24 = (ival>=-8388608)&(ival<=8388607), f32 = !ib&!i16&!i24;
            if constexpr (!SAFE) {
                if constexpr (WRITE) {
                    unsigned u; __builtin_memcpy(&u, &fval, 4);
                    dst[bp] = ib ? (unsigned char)(signed char)ival : i16 ? (unsigned char)(signed char)VLESC2
                            : i24 ? (unsigned char)(signed char)VLESC3 : (unsigned char)(signed char)VLESC4;
                    unsigned pay = f32 ? u : (unsigned)ival;
                    dst[bp+1]=(unsigned char)(pay&0xFF); dst[bp+2]=(unsigned char)((pay>>8)&0xFF);
                    dst[bp+3]=(unsigned char)((pay>>16)&0xFF); dst[bp+4]=(unsigned char)((pay>>24)&0xFF);
                }
            } else {
                unsigned u; __builtin_memcpy(&u, &fval, 4);
                unsigned pay = f32 ? u : (unsigned)ival;
                if (ib) {
                    dst[bp] = (unsigned char)(signed char)ival;
                } else if (i16) {
                    dst[bp] = (unsigned char)(signed char)VLESC2;
                    dst[bp+1] = (unsigned char)(pay & 0xFF);
                    dst[bp+2] = (unsigned char)((pay >> 8) & 0xFF);
                } else if (i24) {
                    dst[bp] = (unsigned char)(signed char)VLESC3;
                    dst[bp+1] = (unsigned char)(pay & 0xFF);
                    dst[bp+2] = (unsigned char)((pay >> 8) & 0xFF);
                    dst[bp+3] = (unsigned char)((pay >> 16) & 0xFF);
                } else {
                    dst[bp] = (unsigned char)(signed char)VLESC4;
                    dst[bp+1] = (unsigned char)(pay & 0xFF);
                    dst[bp+2] = (unsigned char)((pay >> 8) & 0xFF);
                    dst[bp+3] = (unsigned char)((pay >> 16) & 0xFF);
                    dst[bp+4] = (unsigned char)((pay >> 24) & 0xFF);
                }
            }
            bp += ib ? 1 : i16 ? 3 : i24 ? 4 : 5;
        }
    }
    if constexpr (!SAFE) {
        { bool r1=(rle==1),rs=(rle>=2)&(rle<256); int rb=(r1?1:(rs?2:4))&-(rle>0);
          if constexpr (WRITE) {
              dst[bp]=r1?(unsigned char)0:rs?(unsigned char)(RLESC1&0xFF):(unsigned char)(RLESC3&0xFF);
              dst[bp+1]=rs?(unsigned char)rle:(unsigned char)(rle&0xFF);
              dst[bp+2]=(unsigned char)((rle>>8)&0xFF); dst[bp+3]=(unsigned char)((rle>>16)&0xFF);
          }
          bp+=rb; }
    } else {
        if (rle == 1) {
            dst[bp] = 0;
            bp += 1;
        } else if (rle >= 2 && rle < 256) {
            dst[bp] = (unsigned char)(RLESC1 & 0xFF);
            dst[bp+1] = (unsigned char)rle;
            bp += 2;
        } else if (rle >= 256) {
            dst[bp] = (unsigned char)(RLESC3 & 0xFF);
            dst[bp+1] = (unsigned char)(rle & 0xFF);
            dst[bp+2] = (unsigned char)((rle >> 8) & 0xFF);
            dst[bp+3] = (unsigned char)((rle >> 16) & 0xFF);
            bp += 4;
        }
    }
    return bp;
}

// ===========================================================================
// Option A: direct-to-global (no local buf, zero scratch)
// ===========================================================================
__launch_bounds__(256, 2)
__global__ void quantizeRLE_A(
    float* __restrict__ input,
    unsigned char* __restrict__ output,
    int* __restrict__ byte_counts,
    float scale,
    int ldimx, int ldimxy)
{
    constexpr int SLC = 2;
    constexpr int ZPB = 32 * 32;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;

    float* bb = input + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(bb, 0, -1, 0x00027000);
    int boff = (gx + gy * ldimx) * (int)sizeof(float);
    int bstr = ldimxy * (int)sizeof(float);

    qrle_float4_vec planes[32];
    #pragma unroll
    for (int p = 0; p < 32; p++)
        planes[p] = __builtin_bit_cast(qrle_float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(rsrc, boff, p * bstr, SLC));

    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    for (int x_off = 0; x_off < 4; ++x_off) {
        int zl = bid * ZPB + tid * 4 + x_off;
        unsigned char* dst = output + (long)zl * QRLE_BUF;
        int bp = qrle_zline<true>(planes, x_off, scale, dst);
        byte_counts[zl] = bp;
    }
}

// ===========================================================================
// Option B: scratch buf (buf[160] in scratch, write to global at end)
// ===========================================================================
__launch_bounds__(256, 2)
__global__ void quantizeRLE_B(
    float* __restrict__ input,
    unsigned char* __restrict__ output,
    int* __restrict__ byte_counts,
    float scale,
    int ldimx, int ldimxy)
{
    constexpr int SLC = 2;
    constexpr int ZPB = 32 * 32;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;

    float* bb = input + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(bb, 0, -1, 0x00027000);
    int boff = (gx + gy * ldimx) * (int)sizeof(float);
    int bstr = ldimxy * (int)sizeof(float);

    qrle_float4_vec planes[32];
    #pragma unroll
    for (int p = 0; p < 32; p++)
        planes[p] = __builtin_bit_cast(qrle_float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(rsrc, boff, p * bstr, SLC));

    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    unsigned char buf[QRLE_BUF];

    for (int x_off = 0; x_off < 4; ++x_off) {
        int zl = bid * ZPB + tid * 4 + x_off;
        int bp = qrle_zline<true>(planes, x_off, scale, buf);
        unsigned char* dst = output + (long)zl * QRLE_BUF;
        for (int i = 0; i < bp; ++i) dst[i] = buf[i];
        byte_counts[zl] = bp;
    }
}

// ===========================================================================
// Option C: two-pass (dry ALU + direct-to-compacted-LDS), zero scratch.
//
// 32 KB LDS reused: prefix scan, then compacted encode region.
// Occupancy 2 on gfx942.
// ===========================================================================

static constexpr int QRLE_LDS_BYTES = 32768;

__launch_bounds__(256, 2)
__global__ void quantizeRLE_C(
    float* __restrict__ input,
    unsigned char* __restrict__ output,
    int* __restrict__ block_offsets,
    int* __restrict__ block_sizes,
    float scale,
    int ldimx, int ldimxy)
{
    constexpr int SLC = 2;
    constexpr int NTHREADS = 256;
    using BlockScan = rocprim::block_scan<int, NTHREADS>;

    __shared__ union {
        typename BlockScan::storage_type scan;
        unsigned char compact[QRLE_LDS_BYTES];
    } lds;

    int tid = threadIdx.x;
    int gx  = blockIdx.x * 32 + (tid % 8) * 4;
    int gy  = blockIdx.y * 32 + tid / 8;

    float* bb = input + (size_t)blockIdx.z * 32 * ldimxy;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(bb, 0, -1, 0x00027000);
    int boff = (gx + gy * ldimx) * (int)sizeof(float);
    int bstr = ldimxy * (int)sizeof(float);

    qrle_float4_vec planes[32];
    #pragma unroll
    for (int p = 0; p < 32; p++)
        planes[p] = __builtin_bit_cast(qrle_float4_vec,
            __builtin_amdgcn_raw_buffer_load_b128(rsrc, boff, p * bstr, SLC));

    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    unsigned char* block_out = output + (long)bid * 4 * QRLE_LDS_BYTES;
    int block_total = 0;

    for (int x_off = 0; x_off < 4; ++x_off) {

        // 1. Dry encode: ALU only, get byte count
        int my_count = qrle_zline<false>(planes, x_off, scale, nullptr);

        // 2. Exclusive prefix sum (DPP-based via rocprim)
        int my_offset, pass_total;
        BlockScan().exclusive_scan(my_count, my_offset, 0, pass_total, lds.scan);
        __syncthreads();

        // 3. Wet encode: write directly to compacted LDS (safe, no overflow)
        qrle_zline<true, true>(planes, x_off, scale, lds.compact + my_offset);
        __syncthreads();

        // 4. Coalesced dword writes from LDS to global
        int aligned_total = (pass_total + 3) & ~3;
        for (int off = tid * 4; off < aligned_total; off += NTHREADS * 4) {
            unsigned val = 0;
            if (off < pass_total) {
                val  = (unsigned)lds.compact[off];
                if (off+1 < pass_total) val |= (unsigned)lds.compact[off+1] << 8;
                if (off+2 < pass_total) val |= (unsigned)lds.compact[off+2] << 16;
                if (off+3 < pass_total) val |= (unsigned)lds.compact[off+3] << 24;
            }
            if (off < aligned_total) {
                unsigned* dst32 = (unsigned*)(block_out + block_total + off);
                *dst32 = val;
            }
        }
        __syncthreads();

        block_total += pass_total;
    }

    if (tid == 0)
        block_sizes[bid] = block_total;
}

// ===========================================================================
// Benchmark harness
// ===========================================================================

struct KernelStats {
    float time_ms;
    float input_bw_GBs;
    long total_encoded_bytes;
};

template<typename KernelFunc>
KernelStats bench_AB(const char* name, KernelFunc kernel,
                     float* d_input, unsigned char* d_output, int* d_counts,
                     float scale, int nx, int ny, int nz, int ldimx, int ldimxy,
                     int warmup, int runs)
{
    dim3 grid((nx+31)/32, (ny+31)/32, (nz+31)/32);
    dim3 block(256);
    long nblocks = (long)grid.x * grid.y * grid.z;
    long total_zlines = nblocks * 1024L;
    long input_bytes = (long)nx * ny * nz * sizeof(float);

    for (int i = 0; i < warmup; ++i)
        kernel<<<grid, block>>>(d_input, d_output, d_counts, scale, ldimx, ldimxy);
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < runs; ++i)
        kernel<<<grid, block>>>(d_input, d_output, d_counts, scale, ldimx, ldimxy);
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    ms /= runs;

    int* h_counts = (int*)malloc(total_zlines * sizeof(int));
    HIPCHECK(hipMemcpy(h_counts, d_counts, total_zlines * sizeof(int), hipMemcpyDeviceToHost));
    long total_enc = 0;
    for (long i = 0; i < total_zlines; ++i) total_enc += h_counts[i];
    free(h_counts);

    float in_bw = (float)input_bytes / (ms * 1e6);
    float cr = (float)input_bytes / (float)total_enc;
    float wr_bw = (float)(total_zlines * QRLE_BUF) / (ms * 1e6);

    printf("  %-22s %7.3f ms | read %7.1f GB/s | write %7.1f GB/s | enc %ld B | CR %.1f:1\n",
           name, ms, in_bw, wr_bw, total_enc, cr);

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    return {ms, in_bw, total_enc};
}

KernelStats bench_C(const char* name,
                    float* d_input, unsigned char* d_output,
                    int* d_block_offsets, int* d_block_sizes,
                    float scale, int nx, int ny, int nz, int ldimx, int ldimxy,
                    int warmup, int runs)
{
    dim3 grid((nx+31)/32, (ny+31)/32, (nz+31)/32);
    dim3 block(256);
    long nblocks = (long)grid.x * grid.y * grid.z;
    long input_bytes = (long)nx * ny * nz * sizeof(float);

    for (int i = 0; i < warmup; ++i)
        quantizeRLE_C<<<grid, block>>>(d_input, d_output, d_block_offsets, d_block_sizes,
                                        scale, ldimx, ldimxy);
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIPCHECK(hipEventCreate(&t0));
    HIPCHECK(hipEventCreate(&t1));
    HIPCHECK(hipEventRecord(t0));
    for (int i = 0; i < runs; ++i)
        quantizeRLE_C<<<grid, block>>>(d_input, d_output, d_block_offsets, d_block_sizes,
                                        scale, ldimx, ldimxy);
    HIPCHECK(hipEventRecord(t1));
    HIPCHECK(hipEventSynchronize(t1));

    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    ms /= runs;

    int* h_sizes = (int*)malloc(nblocks * sizeof(int));
    HIPCHECK(hipMemcpy(h_sizes, d_block_sizes, nblocks * sizeof(int), hipMemcpyDeviceToHost));
    long total_enc = 0;
    for (long i = 0; i < nblocks; ++i) total_enc += h_sizes[i];
    free(h_sizes);

    float in_bw = (float)input_bytes / (ms * 1e6);
    float cr = (float)input_bytes / (float)total_enc;
    float wr_bw = (float)total_enc / (ms * 1e6);

    printf("  %-22s %7.3f ms | read %7.1f GB/s | write %7.1f GB/s | enc %ld B | CR %.1f:1\n",
           name, ms, in_bw, wr_bw, total_enc, cr);

    HIPCHECK(hipEventDestroy(t0));
    HIPCHECK(hipEventDestroy(t1));
    return {ms, in_bw, total_enc};
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

int main()
{
    int NX = 512, NY = 512, NZ = 512;
    int ldimx = NX, ldimxy = NX * NY;
    long N = (long)NX * NY * NZ;
    long nblocks = (long)(NX/32) * (NY/32) * (NZ/32);
    long total_zlines = nblocks * 1024;

    printf("=== Quantize+RLE Encode Benchmark ===\n");
    printf("Grid: %d^3, Blocks: %ld, Z-lines: %ld\n", NX, nblocks, total_zlines);
    printf("LDS compact cap: %d bytes (occ 2)\n\n", QRLE_LDS_BYTES);

    float* d_input;
    unsigned char* d_output_AB;
    unsigned char* d_output_C;
    int* d_counts;
    int* d_block_offsets;
    int* d_block_sizes;

    HIPCHECK(hipMalloc(&d_input, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output_AB, total_zlines * QRLE_BUF));
    HIPCHECK(hipMalloc(&d_output_C, nblocks * 4 * QRLE_LDS_BYTES));
    HIPCHECK(hipMalloc(&d_counts, total_zlines * sizeof(int)));
    HIPCHECK(hipMalloc(&d_block_offsets, nblocks * sizeof(int)));
    HIPCHECK(hipMalloc(&d_block_sizes, nblocks * sizeof(int)));

    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        initRandom<<<blocks, threads>>>(d_input, N, 42);
        HIPCHECK(hipDeviceSynchronize());
    }

    float scales[] = {0.01f, 0.1f, 1.0f, 10.0f, 100.0f};
    int nscales = sizeof(scales) / sizeof(scales[0]);

    for (int s = 0; s < nscales; ++s) {
        float sc = scales[s];
        printf("scale = %e\n", sc);
        bench_AB("A (direct-global)", quantizeRLE_A, d_input, d_output_AB, d_counts,
                 sc, NX, NY, NZ, ldimx, ldimxy, 2, 10);
        bench_AB("B (scratch buf)", quantizeRLE_B, d_input, d_output_AB, d_counts,
                 sc, NX, NY, NZ, ldimx, ldimxy, 2, 10);
        bench_C("C (2pass compact)", d_input, d_output_C, d_block_offsets, d_block_sizes,
                sc, NX, NY, NZ, ldimx, ldimxy, 2, 10);
        printf("\n");
    }

    HIPCHECK(hipFree(d_input));
    HIPCHECK(hipFree(d_output_AB));
    HIPCHECK(hipFree(d_output_C));
    HIPCHECK(hipFree(d_counts));
    HIPCHECK(hipFree(d_block_offsets));
    HIPCHECK(hipFree(d_block_sizes));
    return 0;
}
