// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIP_CVX_COMPRESS_H
#define HIP_CVX_COMPRESS_H

#include <hip/hip_runtime.h>

enum hipCompressError_t {
    HIP_COMPRESS_SUCCESS = 0,
    HIP_COMPRESS_ERROR_NULL_PLAN,
    HIP_COMPRESS_ERROR_NULL_INPUT,
    HIP_COMPRESS_ERROR_NULL_OUTPUT,
    HIP_COMPRESS_ERROR_INVALID_DIMENSIONS,
    HIP_COMPRESS_ERROR_NOT_MULTIPLE_OF_32,
    HIP_COMPRESS_ERROR_WINDOW_TOO_SMALL,
    HIP_COMPRESS_ERROR_BOTH_OUTPUTS_NULL,
    HIP_COMPRESS_ERROR_COMPRESS_PENDING,
    HIP_COMPRESS_ERROR_NO_COMPRESS_PENDING,
    HIP_COMPRESS_ERROR_MEMORY_ALLOCATION,
    HIP_COMPRESS_ERROR_INVALID_SCALE,
    HIP_COMPRESS_ERROR_EXTRACTION_DIMS_MISMATCH,
    HIP_COMPRESS_ERROR_PLANE_TOO_LARGE,
    HIP_COMPRESS_ERROR_HIP_RUNTIME,
};

struct hipCompressPlan;
hipCompressError_t hipCompressGetLastError(const hipCompressPlan* plan);
const char* hipCompressErrorString(hipCompressError_t err);

enum hipCompressKernel {
    HIP_COMPRESS_KERNEL_ZLINE  = 0,  // parallel z-line RLE (per-block metadata)
    HIP_COMPRESS_KERNEL_SEGRLE = 1,  // segment-aligned RLE (no metadata overhead)
};

struct hipCompressPlan {
    hipCompressKernel kernel;
    int nx, ny, nz;
    int num_blocks;

    unsigned char* d_scratch;
    float* d_mulfac;
    size_t* d_block_sizes;
    size_t* d_block_offsets;
    void* d_scan_temp;
    size_t scan_temp_bytes;

    double* d_partial_sums;
    int     max_copy_blocks;
    double* d_rms;

    hipStream_t aux_stream;
    hipEvent_t  ready_event;
    size_t* h_staging;  // pinned host, 2 values: [offsets[nb-1], sizes[nb-1]]

    bool compress_pending;  // true between hipCompress and hipCompressSynchronize
    mutable hipCompressError_t last_error;
};

// Round up to the next multiple of 32.
inline int hipCompressWaveletDim(int n) {
    return (n + 31) & ~31;
}

// Compute 32-divisible wavelet dimensions from a window.
inline void hipCompressWaveletDims(int wx, int wy, int wz,
                                   int* wnx, int* wny, int* wnz) {
    *wnx = hipCompressWaveletDim(wx);
    *wny = hipCompressWaveletDim(wy);
    *wnz = hipCompressWaveletDim(wz);
}


// Plan owns internal buffers and an event for stream bridging.
// aux_stream: compact, entropy coding, D2H readback. User-owned, shareable
// across plans. One plan must not be used concurrently from multiple host
// threads. Compress is exclusive (writes plan buffers).
hipError_t hipCompressCreatePlan(
    hipCompressPlan** plan,
    int nx, int ny, int nz,
    hipStream_t aux_stream,
    hipCompressKernel kernel = HIP_COMPRESS_KERNEL_ZLINE);

hipError_t hipCompressDestroyPlan(hipCompressPlan* plan);

// Copy from a strided source volume to a 32-divisible wavelet-layout buffer.
// Copies ex*ey*ez samples starting at (x0,y0,z0) in the source grid.
// Zero-fills the padding band (ex..wnx-1, etc.).
// RMS is computed over the ex*ey*ez extraction window only.
//
// ex,ey,ez >= 32. Wavelet dims are derived internally: wnx = round32(ex), etc.
// Must equal plan dimensions exactly.
//
// d_dst may be NULL (RMS-only mode). d_rms_out may be NULL (copy-only mode).
// Both NULL is an error.
// All GPU work is enqueued on user_stream.
hipError_t hipCopyToWaveletLayout(
    const float* d_src,
    int ldimx, int ldimxy,
    int x0, int y0, int z0,
    int ex, int ey, int ez,
    float* d_dst,
    double* d_rms_out,
    hipCompressPlan* plan,
    hipStream_t user_stream);

// Copy from a wavelet-layout buffer back to a strided destination volume.
// Only the ex*ey*ez extraction samples are written; padding is skipped.
// Wavelet dims are derived from the extraction window.
// All GPU work is enqueued on user_stream.
hipError_t hipCopyFromWaveletLayout(
    const float* d_src,
    float* d_dst,
    int ldimx, int ldimxy,
    int x0, int y0, int z0,
    int ex, int ey, int ez,
    hipCompressPlan* plan,
    hipStream_t user_stream);

// Wavelet transform + quantize + RLE encode → self-contained compressed stream.
// Fully async — returns immediately after queuing all GPU work.
// d_input must be a wavelet-layout buffer (32-divisible dims matching plan).
//
// scale + d_rms control the quantization multiplier (mulfac):
//   d_rms != NULL: mulfac = 1 / (rms * scale).  scale is the error tolerance;
//                  smaller scale → finer quantization → lower error, lower CR.
//   d_rms == NULL: mulfac = scale.  Caller supplies mulfac directly.
//
// Fused wavelet+RLE and scan run on user_stream.
// Compact and D2H readback run on aux_stream (internal event bridge).
// Rejects with hipErrorNotReady if a previous compress has not been
// synchronized via hipCompressSynchronize.
// Call hipCompressSynchronize to retrieve compressed_length and CR.
hipError_t hipCompress(
    float scale,
    const double* d_rms,
    const float* d_input,
    unsigned char* d_output,
    hipCompressPlan* plan,
    hipStream_t user_stream);

// Block on aux_stream and retrieve the result of a previous hipCompress.
// If compress_pending: syncs aux_stream, writes compressed_length and
//   compression_ratio (either may be NULL), clears pending. Returns hipSuccess.
// If !compress_pending: no-op, leaves outputs untouched. Returns hipErrorNotReady.
hipError_t hipCompressSynchronize(
    hipCompressPlan* plan,
    long* compressed_length,
    float* compression_ratio);

// RLE decode + inverse wavelet → wavelet-layout buffer.
// Reads the self-contained header (block offsets, mulfac) from d_input.
// Single kernel launch on user_stream. No sync.
hipError_t hipDecompress(
    const unsigned char* d_input,
    float* d_output,
    hipCompressPlan* plan,
    hipStream_t user_stream);

// Upper bound on compressed output size in bytes (for allocation).
hipError_t hipCompressMaxOutputSize(const hipCompressPlan* plan, size_t* size);

#endif
