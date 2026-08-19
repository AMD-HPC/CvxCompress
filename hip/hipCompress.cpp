// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#include "hipCompress.h"

#define DS79_INCLUDE_REG32
#include "hipWaveletRLE.h"
#include "hipWaveletRLEInverse.h"
#include "hipWaveletRLE2D.h"
#include "hipWaveletBitmap.h"
#include "hipWaveletOctree.h"
#include "hipWaveletQuadtree2D.h"
#include "hipBlockCopy.h"

#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Set plan error and return the corresponding hipError_t.
#define PLAN_ERROR(p, code, hip_err) do { (p)->last_error = (code); return (hip_err); } while(0)

#define HIPCHECK_PLAN(p, cmd) do { \
    hipError_t _e = (cmd); \
    if (_e != hipSuccess) { \
        (p)->last_error = HIP_COMPRESS_ERROR_HIP_RUNTIME; \
        return _e; \
    } \
} while(0)

hipCompressError_t hipCompressGetLastError(const hipCompressPlan* plan)
{
    if (!plan) return HIP_COMPRESS_ERROR_NULL_PLAN;
    return plan->last_error;
}

const char* hipCompressErrorString(hipCompressError_t err)
{
    switch (err) {
    case HIP_COMPRESS_SUCCESS:                return "success";
    case HIP_COMPRESS_ERROR_NULL_PLAN:        return "plan pointer is null";
    case HIP_COMPRESS_ERROR_NULL_INPUT:       return "input pointer is null";
    case HIP_COMPRESS_ERROR_NULL_OUTPUT:      return "output pointer is null";
    case HIP_COMPRESS_ERROR_INVALID_DIMENSIONS: return "invalid dimensions (must be > 0)";
    case HIP_COMPRESS_ERROR_NOT_MULTIPLE_OF_32: return "dimensions must be multiples of 32";
    case HIP_COMPRESS_ERROR_WINDOW_TOO_SMALL: return "window dimensions must be >= 32";
    case HIP_COMPRESS_ERROR_BOTH_OUTPUTS_NULL: return "both d_dst and d_rms_out are null";
    case HIP_COMPRESS_ERROR_COMPRESS_PENDING: return "previous compress not yet synchronized";
    case HIP_COMPRESS_ERROR_NO_COMPRESS_PENDING: return "no compress pending to synchronize";
    case HIP_COMPRESS_ERROR_MEMORY_ALLOCATION: return "memory allocation failed";
    case HIP_COMPRESS_ERROR_INVALID_SCALE:    return "scale must be > 0 and finite";
    case HIP_COMPRESS_ERROR_EXTRACTION_DIMS_MISMATCH:     return "extraction wavelet dims must equal plan dims";
    case HIP_COMPRESS_ERROR_PLANE_TOO_LARGE:  return "single plane exceeds 4 GB (nx * ny * 4 > 2^32)";
    case HIP_COMPRESS_ERROR_HIP_RUNTIME:      return "internal HIP runtime error";
    default:                                  return "unknown error";
    }
}

hipError_t hipCompressCreatePlan(hipCompressPlan** plan, int nx, int ny, int nz,
                                 hipStream_t aux_stream, hipCompressKernel kernel)
{
    if (!plan)
        return hipErrorInvalidValue;

    hipCompressPlan* p = (hipCompressPlan*)calloc(1, sizeof(hipCompressPlan));
    if (!p) return hipErrorMemoryAllocation;
    *plan = p;

    if (nx <= 0 || ny <= 0 || nz <= 0)
        PLAN_ERROR(p, HIP_COMPRESS_ERROR_INVALID_DIMENSIONS, hipErrorInvalidValue);

    bool is_2d = (nz == 1);

    if (is_2d) {
        if (nx % 32 != 0 || ny % 32 != 0)
            PLAN_ERROR(p, HIP_COMPRESS_ERROR_NOT_MULTIPLE_OF_32, hipErrorInvalidValue);
    } else {
        if (nx % 32 != 0 || ny % 32 != 0 || nz % 32 != 0)
            PLAN_ERROR(p, HIP_COMPRESS_ERROR_NOT_MULTIPLE_OF_32, hipErrorInvalidValue);
    }

    if ((long)nx * (long)ny * (long)sizeof(float) > (1L << 32))
        PLAN_ERROR(p, HIP_COMPRESS_ERROR_PLANE_TOO_LARGE, hipErrorInvalidValue);

    // Resolve the dimensionality-selected default to a concrete codec: quadtree
    // for 2D, octree for 3D.  Dimensions are already validated as 32-multiples
    // above (shared by every codec), so the structured coders always apply for
    // valid dims; ZLINE remains the fallback for configurations they cannot
    // handle, guarding future constraints so the default flip never breaks
    // callers.  Downstream logic only ever sees the resolved concrete kernel.
    if (kernel == HIP_COMPRESS_KERNEL_AUTO) {
        kernel = is_2d ? HIP_COMPRESS_KERNEL_QUADTREE
                       : HIP_COMPRESS_KERNEL_OCTREE;
    }

    p->kernel = kernel;
    p->nx = nx; p->ny = ny; p->nz = nz;
    p->is_2d = is_2d;
    p->aux_stream = aux_stream;
    p->compress_pending = false;
    p->last_error = HIP_COMPRESS_ERROR_HIP_RUNTIME;

    if (is_2d) {
        // Octree and two-level significance coders are 3D only.
        if (kernel == HIP_COMPRESS_KERNEL_OCTREE ||
            kernel == HIP_COMPRESS_KERNEL_TWOLEVEL)
            PLAN_ERROR(p, HIP_COMPRESS_ERROR_INVALID_DIMENSIONS, hipErrorInvalidValue);
        p->num_blocks = (nx / 32) * (ny / 32);
        // Quadtree: d_scratch holds the per-block int32 32x32 grid (encode: k1
        // output; decode: stage-A output).  RLE: fixed-stride RLE slots.
        p->scratch_slot_stride = (kernel == HIP_COMPRESS_KERNEL_QUADTREE)
                               ? (size_t)WQT2D_GRID_BYTES : (size_t)WRLE2D_SLOT_BYTES;
    } else {
        // Quadtree significance coder is 2D only.
        if (kernel == HIP_COMPRESS_KERNEL_QUADTREE)
            PLAN_ERROR(p, HIP_COMPRESS_ERROR_INVALID_DIMENSIONS, hipErrorInvalidValue);
        p->num_blocks = (nx / 32) * (ny / 32) * (nz / 32);
        // Octree / two-level: d_scratch holds the kernel-1 bitmap+values layout
        // (encode: k1 output; decode: stage-A output).  RLE: fixed-stride slots.
        bool bitmap_split = (kernel == HIP_COMPRESS_KERNEL_OCTREE ||
                             kernel == HIP_COMPRESS_KERNEL_TWOLEVEL);
        p->scratch_slot_stride = bitmap_split
                               ? (size_t)WBMP_SLOT_BYTES : 4L * WRLE_LDS_BYTES;
    }

    int nb = p->num_blocks;
    long scratch_size = (long)nb * (long)p->scratch_slot_stride;

    HIPCHECK_PLAN(p, hipMalloc(&p->d_scratch, scratch_size));
    HIPCHECK_PLAN(p, hipMalloc(&p->d_mulfac, sizeof(float)));
    HIPCHECK_PLAN(p, hipMalloc(&p->d_block_sizes, nb * sizeof(size_t)));
    HIPCHECK_PLAN(p, hipMalloc(&p->d_block_offsets, nb * sizeof(size_t)));

    if (kernel == HIP_COMPRESS_KERNEL_OCTREE) {
        HIPCHECK_PLAN(p, hipMalloc(&p->d_octree_coded,
                                   (long)nb * WOCT_CODE_SLOT_BYTES));
        HIPCHECK_PLAN(p, hipMalloc(&p->d_octree_sig_sizes, nb * sizeof(size_t)));
        HIPCHECK_PLAN(p, hipMalloc(&p->d_inv_scale, sizeof(float)));
    } else if (kernel == HIP_COMPRESS_KERNEL_QUADTREE) {
        HIPCHECK_PLAN(p, hipMalloc(&p->d_octree_coded,
                                   (long)nb * WQT2D_CODE_SLOT_BYTES));
        HIPCHECK_PLAN(p, hipMalloc(&p->d_octree_sig_sizes, nb * sizeof(size_t)));
        HIPCHECK_PLAN(p, hipMalloc(&p->d_inv_scale, sizeof(float)));
    } else if (kernel == HIP_COMPRESS_KERNEL_TWOLEVEL) {
        HIPCHECK_PLAN(p, hipMalloc(&p->d_octree_coded,
                                   (long)nb * WBMP_TL_SLOT_BYTES));
        HIPCHECK_PLAN(p, hipMalloc(&p->d_inv_scale, sizeof(float)));
        // Select the encoder by device arch: the LDS-staged opt kernel needs
        // gfx950's 160 KB LDS; every other arch uses the portable encoder.
        int dev = 0;
        hipDeviceProp_t prop;
        if (hipGetDevice(&dev) == hipSuccess &&
            hipGetDeviceProperties(&prop, dev) == hipSuccess)
            p->tl_use_opt = (strstr(prop.gcnArchName, "gfx950") != nullptr);
    }

    p->scan_temp_bytes = 0;
    hipError_t err = hipWaveletRLECompactScanTempSize(nb, &p->scan_temp_bytes);
    if (err != hipSuccess) { p->last_error = HIP_COMPRESS_ERROR_HIP_RUNTIME; return err; }

    HIPCHECK_PLAN(p, hipMalloc(&p->d_scan_temp, p->scan_temp_bytes));

    HIPCHECK_PLAN(p, hipMalloc(&p->d_rms, sizeof(double)));

    int nz_for_copy = is_2d ? 1 : nz;
    p->max_copy_blocks = (nx / 32) * (ny / 32)
                       * ((nz_for_copy + BCOPY_ZPB - 1) / BCOPY_ZPB);
    HIPCHECK_PLAN(p, hipMalloc(&p->d_partial_sums, p->max_copy_blocks * sizeof(double)));

    HIPCHECK_PLAN(p, hipEventCreateWithFlags(&p->ready_event, hipEventDisableTiming));
    HIPCHECK_PLAN(p, hipHostMalloc(&p->h_staging, 2 * sizeof(size_t)));

    // JIT warmup: exclusive_scan on aux_stream
    err = rocprim::exclusive_scan(
        p->d_scan_temp, p->scan_temp_bytes,
        p->d_block_sizes, p->d_block_offsets, (size_t)0, (size_t)1,
        rocprim::plus<size_t>(), aux_stream);
    if (err != hipSuccess) { p->last_error = HIP_COMPRESS_ERROR_HIP_RUNTIME; return err; }
    HIPCHECK_PLAN(p, hipStreamSynchronize(aux_stream));

    p->last_error = HIP_COMPRESS_SUCCESS;
    return hipSuccess;
}

hipError_t hipCompressDestroyPlan(hipCompressPlan* plan)
{
    if (!plan) return hipErrorInvalidValue;
    (void)hipFree(plan->d_scratch);
    (void)hipFree(plan->d_mulfac);
    (void)hipFree(plan->d_block_sizes);
    (void)hipFree(plan->d_block_offsets);
    (void)hipFree(plan->d_scan_temp);
    (void)hipFree(plan->d_partial_sums);
    (void)hipFree(plan->d_rms);
    (void)hipFree(plan->d_octree_coded);
    (void)hipFree(plan->d_octree_sig_sizes);
    (void)hipFree(plan->d_inv_scale);
    if (plan->h_staging) (void)hipHostFree(plan->h_staging);
    if (plan->ready_event) (void)hipEventDestroy(plan->ready_event);
    free(plan);
    return hipSuccess;
}

hipError_t hipCompress(
    float scale,
    const double* d_rms,
    const float* d_input,
    unsigned char* d_output,
    hipCompressPlan* plan,
    hipStream_t user_stream)
{
    if (!plan) return hipErrorInvalidValue;
    plan->last_error = HIP_COMPRESS_SUCCESS;
    if (!d_input)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_NULL_INPUT, hipErrorInvalidValue);
    if (!d_output)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_NULL_OUTPUT, hipErrorInvalidValue);
    if (plan->compress_pending)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_COMPRESS_PENDING, hipErrorNotReady);
    if (scale <= 0.0f || !std::isfinite(scale))
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_INVALID_SCALE, hipErrorInvalidValue);

    const int nx = plan->nx, ny = plan->ny, nz = plan->nz;
    const int ldimx = nx;
    const int nb = plan->num_blocks;
    const int num_mulfacs = 1;
    // OCTREE and QUADTREE use the self-contained header with a per-block
    // significance-size table; the RLE / two-level paths use the compact header.
    const bool octree_hdr = (plan->kernel == HIP_COMPRESS_KERNEL_OCTREE ||
                             plan->kernel == HIP_COMPRESS_KERNEL_QUADTREE);
    const int hdr_size = octree_hdr
                       ? hipOctreeHeaderSize(nb, num_mulfacs)
                       : hipCompressHeaderSize(nb, num_mulfacs);
    hipStream_t s = user_stream;
    hipStream_t aux = plan->aux_stream;

    // 1. Fused wavelet + quantize + encode → scratch (user_stream)
    if (plan->is_2d) {
        int nbx = nx / 32, nby = ny / 32;
        dim3 grid((nbx + WRLE2D_TILES_PER_WG - 1) / WRLE2D_TILES_PER_WG, nby);
        if (plan->kernel == HIP_COMPRESS_KERNEL_QUADTREE) {
            // k1: 2D wavelet + quantize → int32 32x32 grid in d_scratch.
            waveletQuadtree2DForwardKernel<<<grid, dim3(256), 0, s>>>(
                d_input, reinterpret_cast<int*>(plan->d_scratch),
                scale, ldimx, nbx, d_rms, plan->d_mulfac);
            // k2: quadtree significance + width table + packed values → coded
            // slots.  d_block_sizes := coded length; d_octree_sig_sizes :=
            // significance length (both per block, for scan + header).
            waveletQuadtree2DCodeKernel<<<nb, dim3(WQT2D_CODE_THREADS), 0, s>>>(
                reinterpret_cast<const int*>(plan->d_scratch), plan->d_octree_coded,
                plan->d_block_sizes, plan->d_octree_sig_sizes);
        } else {
            waveletRLE2DFusedKernel<<<grid, dim3(256), 0, s>>>(
                d_input, plan->d_scratch, plan->d_block_sizes,
                scale, ldimx, nbx,
                d_rms, plan->d_mulfac);
        }
    } else {
        const int ldimxy = nx * ny;
        dim3 grid((nx + 31) / 32, (ny + 31) / 32, (nz + 31) / 32);
        if (plan->kernel == HIP_COMPRESS_KERNEL_OCTREE) {
            // k1: wavelet ZYX + quantize → [bitmap][packed int32] in d_scratch.
            waveletBitmapFusedKernel<<<grid, dim3(256), 0, s>>>(
                d_input, plan->d_scratch, plan->d_block_sizes,
                scale, ldimx, ldimxy,
                d_rms, plan->d_mulfac);
            // k2: octree significance + width table + packed values → coded
            // slots.  d_block_sizes := coded length; d_octree_sig_sizes :=
            // significance length (both per block, for scan + header).
            waveletOctreeCodeParKernel<<<nb, dim3(WOCT_PAR_THREADS), 0, s>>>(
                plan->d_scratch, plan->d_octree_coded,
                plan->d_block_sizes, plan->d_octree_sig_sizes);
            // 4-align each coded length so packed blocks keep uint32 reads
            // (width table / flat masks) aligned in the compacted stream.
            waveletOctreeCodeParAlignSizes(plan->d_block_sizes, nb, s);
        } else if (plan->kernel == HIP_COMPRESS_KERNEL_TWOLEVEL) {
            // k1: wavelet ZYX + quantize → [bitmap][packed int32] in d_scratch.
            waveletBitmapFusedKernel<<<grid, dim3(256), 0, s>>>(
                d_input, plan->d_scratch, plan->d_block_sizes,
                scale, ldimx, ldimxy,
                d_rms, plan->d_mulfac);
            // k2: two-level occupancy + width table + packed values → coded
            // slots.  Arch-selected encoder; both emit the identical stream.
            if (plan->tl_use_opt)
                waveletBitmapCodeTwoLevelOptKernel<<<nb, dim3(wbmp_opt::WBMP_OPT_THREADS), 0, s>>>(
                    plan->d_scratch, nullptr, plan->d_octree_coded, plan->d_block_sizes);
            else
                waveletBitmapCodeTwoLevelKernel<<<nb, dim3(256), 0, s>>>(
                    plan->d_scratch, nullptr, plan->d_octree_coded, plan->d_block_sizes);
            // 4-align each coded length (uint32 occupancy/mask/width reads).
            waveletOctreeCodeParAlignSizes(plan->d_block_sizes, nb, s);
        } else if (plan->kernel == HIP_COMPRESS_KERNEL_SEGRLE) {
            waveletSegRLEFusedKernel<<<grid, dim3(256), 0, s>>>(
                d_input, plan->d_scratch, plan->d_block_sizes,
                scale, ldimx, ldimxy,
                d_rms, plan->d_mulfac);
        } else {
            waveletRLEFusedKernel<<<grid, dim3(256), 0, s>>>(
                d_input, plan->d_scratch, plan->d_block_sizes,
                scale, ldimx, ldimxy,
                d_rms, plan->d_mulfac);
        }
    }

    // 2. Exclusive scan for compaction offsets (user_stream)
    HIPCHECK_PLAN(plan, rocprim::exclusive_scan(
        plan->d_scan_temp, plan->scan_temp_bytes,
        plan->d_block_sizes, plan->d_block_offsets, (size_t)0, (size_t)nb,
        rocprim::plus<size_t>(), s));

    // 3. Bridge: signal user_stream done, aux_stream waits
    HIPCHECK_PLAN(plan, hipEventRecord(plan->ready_event, s));
    HIPCHECK_PLAN(plan, hipStreamWaitEvent(aux, plan->ready_event, 0));

    // 4. Compact + write header on aux_stream
    if (plan->is_2d && plan->kernel == HIP_COMPRESS_KERNEL_QUADTREE) {
        wqt2dCompactKernel<<<nb, 256, 0, aux>>>(
            plan->d_octree_coded, d_output + hdr_size,
            plan->d_block_sizes, plan->d_block_offsets, plan->d_octree_sig_sizes,
            d_output, nb, num_mulfacs, plan->d_mulfac);
    } else if (plan->is_2d) {
        wrle2DCompactKernel<<<nb, 256, 0, aux>>>(
            plan->d_scratch, d_output + hdr_size,
            plan->d_block_sizes, plan->d_block_offsets,
            d_output, nb, num_mulfacs, plan->d_mulfac,
            plan->scratch_slot_stride);
    } else if (plan->kernel == HIP_COMPRESS_KERNEL_OCTREE) {
        woctCompactKernel<<<nb, 256, 0, aux>>>(
            plan->d_octree_coded, d_output + hdr_size,
            plan->d_block_sizes, plan->d_block_offsets, plan->d_octree_sig_sizes,
            d_output, nb, num_mulfacs, plan->d_mulfac);
    } else if (plan->kernel == HIP_COMPRESS_KERNEL_TWOLEVEL) {
        wtlCompactKernel<<<nb, 256, 0, aux>>>(
            plan->d_octree_coded, d_output + hdr_size,
            plan->d_block_sizes, plan->d_block_offsets,
            d_output, nb, num_mulfacs, plan->d_mulfac);
    } else {
        wrleCompactKernel<<<nb, 256, 0, aux>>>(
            plan->d_scratch, d_output + hdr_size,
            plan->d_block_sizes, plan->d_block_offsets,
            d_output, nb, num_mulfacs, plan->d_mulfac);
    }

    // 5. Async readback on aux_stream
    HIPCHECK_PLAN(plan, hipMemcpyAsync(&plan->h_staging[0], plan->d_block_offsets + (nb - 1),
                            sizeof(size_t), hipMemcpyDeviceToHost, aux));
    HIPCHECK_PLAN(plan, hipMemcpyAsync(&plan->h_staging[1], plan->d_block_sizes + (nb - 1),
                            sizeof(size_t), hipMemcpyDeviceToHost, aux));

    plan->compress_pending = true;
    return hipSuccess;
}

hipError_t hipCompressSynchronize(
    hipCompressPlan* plan,
    long* compressed_length,
    float* compression_ratio)
{
    if (!plan) return hipErrorInvalidValue;
    plan->last_error = HIP_COMPRESS_SUCCESS;
    if (!plan->compress_pending)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_NO_COMPRESS_PENDING, hipErrorNotReady);

    HIPCHECK_PLAN(plan, hipStreamSynchronize(plan->aux_stream));

    const int nx = plan->nx, ny = plan->ny, nz = plan->nz;
    const int nb = plan->num_blocks;
    const bool octree_hdr = (plan->kernel == HIP_COMPRESS_KERNEL_OCTREE ||
                             plan->kernel == HIP_COMPRESS_KERNEL_QUADTREE);
    const int hdr_size = octree_hdr
                       ? hipOctreeHeaderSize(nb, 1)
                       : hipCompressHeaderSize(nb, 1);
    size_t total_payload = plan->h_staging[0] + plan->h_staging[1];
    long total_bytes = (long)hdr_size + (long)total_payload;

    // Round the reported length up to 8 bytes. A caller packing streams
    // back-to-back (base += compressed_length) then keeps each stream base
    // >=8B aligned, which the in-stream size_t block-offset table requires.
    // Decode is header/offset-driven and never reads the trailing pad bytes.
    // CR carries this overhead since compressed_length is the reported size.
    total_bytes = (total_bytes + 7) & ~7L;

    if (compressed_length)
        *compressed_length = total_bytes;
    if (compression_ratio) {
        long total = (long)nx * ny * nz;
        *compression_ratio = (float)((long)total * sizeof(float)) / (float)total_bytes;
    }

    plan->compress_pending = false;
    return hipSuccess;
}

hipError_t hipCopyToWaveletLayout(
    const float* d_src,
    int ldimx, int ldimxy,
    int x0, int y0, int z0,
    int ex, int ey, int ez,
    float* d_dst,
    double* d_rms_out,
    hipCompressPlan* plan,
    hipStream_t user_stream)
{
    if (!plan) return hipErrorInvalidValue;
    plan->last_error = HIP_COMPRESS_SUCCESS;
    if (!d_src)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_NULL_INPUT, hipErrorInvalidValue);
    if (!d_dst && !d_rms_out)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_BOTH_OUTPUTS_NULL, hipErrorInvalidValue);
    if (plan->is_2d) {
        if (ex < 32 || ey < 32 || ez != 1)
            PLAN_ERROR(plan, HIP_COMPRESS_ERROR_WINDOW_TOO_SMALL, hipErrorInvalidValue);
    } else {
        if (ex < 32 || ey < 32 || ez < 32)
            PLAN_ERROR(plan, HIP_COMPRESS_ERROR_WINDOW_TOO_SMALL, hipErrorInvalidValue);
    }
    if ((long)ldimxy * (long)sizeof(float) > (1L << 32))
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_PLANE_TOO_LARGE, hipErrorInvalidValue);

    int wnx = hipCompressWaveletDim(ex);
    int wny = hipCompressWaveletDim(ey);
    int wnz = plan->is_2d ? 1 : hipCompressWaveletDim(ez);

    if (wnx != plan->nx || wny != plan->ny || wnz != plan->nz)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_EXTRACTION_DIMS_MISMATCH, hipErrorInvalidValue);

    hipStream_t s = user_stream;
    bool do_copy = (d_dst != nullptr);
    bool do_rms  = (d_rms_out != nullptr);
    long total_samples = (long)ex * ey * ez;

    dim3 grid(wnx / 32, wny / 32, (wnz + BCOPY_ZPB - 1) / BCOPY_ZPB);
    int total_blocks = grid.x * grid.y * grid.z;

    if (do_copy && do_rms) {
        copyToWaveletKernelOpt<true, true><<<grid, 256, 0, s>>>(
            d_src, ldimx, ldimxy,
            x0, y0, z0, ex, ey, ez,
            d_dst, wnx, wny, wnz,
            plan->d_partial_sums);
    } else if (do_copy) {
        copyToWaveletKernelOpt<true, false><<<grid, 256, 0, s>>>(
            d_src, ldimx, ldimxy,
            x0, y0, z0, ex, ey, ez,
            d_dst, wnx, wny, wnz,
            nullptr);
    } else {
        copyToWaveletKernelOpt<false, true><<<grid, 256, 0, s>>>(
            d_src, ldimx, ldimxy,
            x0, y0, z0, ex, ey, ez,
            nullptr, wnx, wny, wnz,
            plan->d_partial_sums);
    }

    if (do_rms) {
        reducePartialSumsToRMS<<<1, 256, 0, s>>>(
            plan->d_partial_sums, d_rms_out, total_blocks, total_samples);
    }

    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess)
        plan->last_error = HIP_COMPRESS_ERROR_HIP_RUNTIME;
    return launch_err;
}

hipError_t hipComputeRMS(
    const float* d_src,
    int ldimx, int ldimxy,
    int x0, int y0, int z0,
    int ex, int ey, int ez,
    double* d_rms_out,
    hipCompressPlan* plan,
    hipStream_t user_stream)
{
    if (!plan) return hipErrorInvalidValue;
    plan->last_error = HIP_COMPRESS_SUCCESS;
    if (!d_rms_out)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_NULL_OUTPUT, hipErrorInvalidValue);
    return hipCopyToWaveletLayout(
        d_src, ldimx, ldimxy,
        x0, y0, z0, ex, ey, ez,
        /*d_dst=*/nullptr, d_rms_out,
        plan, user_stream);
}

hipError_t hipCopyFromWaveletLayout(
    const float* d_src,
    float* d_dst,
    int ldimx, int ldimxy,
    int x0, int y0, int z0,
    int ex, int ey, int ez,
    hipCompressPlan* plan,
    hipStream_t user_stream)
{
    if (!plan) return hipErrorInvalidValue;
    plan->last_error = HIP_COMPRESS_SUCCESS;
    if (!d_src)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_NULL_INPUT, hipErrorInvalidValue);
    if (!d_dst)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_NULL_OUTPUT, hipErrorInvalidValue);
    if (plan->is_2d) {
        if (ex < 32 || ey < 32 || ez != 1)
            PLAN_ERROR(plan, HIP_COMPRESS_ERROR_WINDOW_TOO_SMALL, hipErrorInvalidValue);
    } else {
        if (ex < 32 || ey < 32 || ez < 32)
            PLAN_ERROR(plan, HIP_COMPRESS_ERROR_WINDOW_TOO_SMALL, hipErrorInvalidValue);
    }
    if ((long)ldimxy * (long)sizeof(float) > (1L << 32))
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_PLANE_TOO_LARGE, hipErrorInvalidValue);

    int wnx = hipCompressWaveletDim(ex);
    int wny = hipCompressWaveletDim(ey);
    int wnz = plan->is_2d ? 1 : hipCompressWaveletDim(ez);

    if (wnx != plan->nx || wny != plan->ny || wnz != plan->nz)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_EXTRACTION_DIMS_MISMATCH, hipErrorInvalidValue);

    dim3 grid(wnx / 32, wny / 32, (wnz + BCOPY_ZPB - 1) / BCOPY_ZPB);
    copyFromWaveletKernelOpt<<<grid, 256, 0, user_stream>>>(
        d_src, wnx, wny, wnz,
        d_dst, ldimx, ldimxy, x0, y0, z0, ex, ey, ez);

    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess)
        plan->last_error = HIP_COMPRESS_ERROR_HIP_RUNTIME;
    return launch_err;
}

hipError_t hipDecompress(
    const unsigned char* d_input,
    float* d_output,
    hipCompressPlan* plan,
    hipStream_t user_stream)
{
    if (!plan) return hipErrorInvalidValue;
    plan->last_error = HIP_COMPRESS_SUCCESS;
    if (!d_input)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_NULL_INPUT, hipErrorInvalidValue);
    if (!d_output)
        PLAN_ERROR(plan, HIP_COMPRESS_ERROR_NULL_OUTPUT, hipErrorInvalidValue);

    const int nx = plan->nx, ny = plan->ny, nz = plan->nz;
    const int ldimx = nx;

    if (plan->is_2d) {
        int nbx = nx / 32, nby = ny / 32;
        dim3 grid((nbx + WRLE2D_TILES_PER_WG - 1) / WRLE2D_TILES_PER_WG, nby);
        if (plan->kernel == HIP_COMPRESS_KERNEL_QUADTREE) {
            const int nb = plan->num_blocks;
            // stage A: compacted quadtree stream → int32 32x32 grid scratch;
            // also publishes inv_scale = 1/mulfac (device) for stage B.
            waveletQuadtree2DDecodeHdrKernel<<<nb, dim3(WQT2D_CODE_THREADS), 0, user_stream>>>(
                d_input, reinterpret_cast<int*>(plan->d_scratch), plan->d_inv_scale);
            // stage B: dequantize + inverse 2D wavelet → wavefield.
            waveletQuadtree2DInverseKernel<<<grid, dim3(256), 0, user_stream>>>(
                reinterpret_cast<const int*>(plan->d_scratch), d_output,
                plan->d_inv_scale, ldimx, nbx);
        } else {
            waveletRLE2DInverseFusedKernel<<<grid, dim3(256), 0, user_stream>>>(
                d_input, nullptr, nullptr,
                d_output, 0.0f, ldimx, nbx, 1);
        }
    } else {
        const int ldimxy = nx * ny;
        dim3 grid((nx + 31) / 32, (ny + 31) / 32, (nz + 31) / 32);
        if (plan->kernel == HIP_COMPRESS_KERNEL_OCTREE) {
            const int nb = plan->num_blocks;
            // stage A: compacted coded stream → kernel-1 scratch layout; also
            // publishes inv_scale = 1/mulfac (device) for stage B.
            waveletOctreeDecodeToBitmapHdrKernel<<<nb, dim3(WOCT_PAR_THREADS), 0, user_stream>>>(
                d_input, plan->d_scratch, plan->d_inv_scale);
            // stage B: dequantize + inverse wavelet ZYX → wavefield.
            waveletBitmapInverseFusedDevKernel<<<grid, dim3(256), 0, user_stream>>>(
                plan->d_scratch, d_output, plan->d_inv_scale, ldimx, ldimxy);
        } else if (plan->kernel == HIP_COMPRESS_KERNEL_TWOLEVEL) {
            const int nb = plan->num_blocks;
            // stage A: compacted two-level stream → kernel-1 scratch layout;
            // also publishes inv_scale = 1/mulfac (device) for stage B.
            waveletBitmapTwoLevelDecodeHdrKernel<<<nb, dim3(wbmp_opt::WBMP_OPT_THREADS), 0, user_stream>>>(
                d_input, plan->d_scratch, plan->d_inv_scale);
            // stage B: dequantize + inverse wavelet ZYX → wavefield.
            waveletBitmapInverseFusedDevKernel<<<grid, dim3(256), 0, user_stream>>>(
                plan->d_scratch, d_output, plan->d_inv_scale, ldimx, ldimxy);
        } else if (plan->kernel == HIP_COMPRESS_KERNEL_SEGRLE) {
            waveletSegRLEInverseFusedKernel<<<grid, dim3(256), 0, user_stream>>>(
                d_input, nullptr, nullptr,
                d_output, 0.0f, ldimx, ldimxy, 1);
        } else {
            waveletRLEInverseFusedKernel<<<grid, dim3(256), 0, user_stream>>>(
                d_input, nullptr, nullptr,
                d_output, 0.0f, ldimx, ldimxy, 1);
        }
    }

    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess)
        plan->last_error = HIP_COMPRESS_ERROR_HIP_RUNTIME;
    return launch_err;
}

hipError_t hipCompressBufferSize(const hipCompressPlan* plan, size_t* size)
{
    if (!plan) return hipErrorInvalidValue;
    if (!size) {
        plan->last_error = HIP_COMPRESS_ERROR_NULL_OUTPUT;
        return hipErrorInvalidValue;
    }
    plan->last_error = HIP_COMPRESS_SUCCESS;
    *size = (size_t)plan->nx * plan->ny * plan->nz * sizeof(float);
    return hipSuccess;
}

hipError_t hipCompressMaxOutputSize(const hipCompressPlan* plan, size_t* size)
{
    if (!plan) return hipErrorInvalidValue;
    if (!size) {
        plan->last_error = HIP_COMPRESS_ERROR_NULL_OUTPUT;
        return hipErrorInvalidValue;
    }
    plan->last_error = HIP_COMPRESS_SUCCESS;
    size_t raw;
    if (plan->kernel == HIP_COMPRESS_KERNEL_OCTREE) {
        // Worst case: octree header + every block at its coded-slot upper bound.
        int hdr_size = hipOctreeHeaderSize(plan->num_blocks, 1);
        raw = (size_t)hdr_size + (size_t)plan->num_blocks * WOCT_CODE_SLOT_BYTES;
    } else if (plan->kernel == HIP_COMPRESS_KERNEL_QUADTREE) {
        // Worst case: octree-style header + every block at the quadtree slot bound.
        int hdr_size = hipOctreeHeaderSize(plan->num_blocks, 1);
        raw = (size_t)hdr_size + (size_t)plan->num_blocks * WQT2D_CODE_SLOT_BYTES;
    } else if (plan->kernel == HIP_COMPRESS_KERNEL_TWOLEVEL) {
        // Worst case: RLE-style header + every block at the two-level slot bound
        // (larger than the WBMP_SLOT_BYTES scratch stride by the occupancy mask).
        int hdr_size = hipCompressHeaderSize(plan->num_blocks, 1);
        raw = (size_t)hdr_size + (size_t)plan->num_blocks * WBMP_TL_SLOT_BYTES;
    } else {
        int hdr_size = hipCompressHeaderSize(plan->num_blocks, 1);
        raw = (size_t)hdr_size + (size_t)plan->num_blocks * plan->scratch_slot_stride;
    }
    // Include the 8B round-up slack applied to compressed_length so a buffer
    // sized from this value can always hold the padded stream.
    *size = (raw + 7) & ~(size_t)7;
    return hipSuccess;
}
