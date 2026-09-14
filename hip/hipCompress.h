// Copyright (C) 2026 Advanced Micro Devices, Inc.
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
    HIP_COMPRESS_ERROR_INVALID_CODEC,
    HIP_COMPRESS_ERROR_INVALID_AUX_STAGE,
    HIP_COMPRESS_ERROR_HIP_RUNTIME,
};

struct hipCompressPlan;
hipCompressError_t hipCompressGetLastError(const hipCompressPlan* plan);
const char* hipCompressErrorString(hipCompressError_t err);

enum hipCompressKernel {
    HIP_COMPRESS_KERNEL_OCTREE   = 0,  // octree significance coder (3D only)
    HIP_COMPRESS_KERNEL_QUADTREE = 1,  // quadtree significance coder (2D only) --
                                       // the 2D counterpart of OCTREE.
    HIP_COMPRESS_KERNEL_AUTO     = 2,  // dimensionality-selected default: QUADTREE
                                       // for 2D and OCTREE for 3D.
};

// Which stage of the encode chain aux_stream picks up from. The chain is
//
//   transform -> encode -> scan -> compact -> D2H readback
//
// where "transform" is the fused wavelet + quantize + significance pass and
// "encode" is the entropy coder (octree/quadtree significance + PFOR values).
//
// Transform never moves: it is bandwidth-bound and reads the caller's live
// input buffer, so overlapping it only steals HBM from the caller.
//
// This is a placement knob, not a correctness one: all three settings produce
// byte-identical output. It is also not a free win. Measured against a wave
// propagation kernel that already saturates the GPU, FROM_ENCODE is 2-4% SLOWER
// than NONE, because the entropy coder is a throughput kernel with no idle
// slack to reclaim and co-residency costs more than the hiding saves. See
// rtm_storage/hipcvx_overlap_findings.md, sections 3f and 3h. Profile your own
// caller before moving off the default.
typedef enum {
    // aux_stream unused; the whole chain runs on user_stream.
    HIP_COMPRESS_AUX_NONE = 0,
    // Scan, compaction and the D2H readback run on aux_stream. Default.
    HIP_COMPRESS_AUX_FROM_COMPACT = 1,
    // Entropy coding onward runs on aux_stream (adds the encode stage and its
    // size-alignment pass to the above).
    HIP_COMPRESS_AUX_FROM_ENCODE = 2,
} hipCompressAuxStage;

struct hipCompressPlan {
    hipCompressKernel kernel;
    int nx, ny, nz;
    int num_blocks;
    bool is_2d;             // true when nz == 1
    size_t scratch_slot_stride;  // per-block scratch slot size

    unsigned char* d_scratch;
    float* d_mulfac;
    size_t* d_block_sizes;
    size_t* d_block_offsets;
    void* d_scan_temp;
    size_t scan_temp_bytes;

    // Octree / quadtree kernels:
    // d_octree_coded is the fixed-stride intermediate coded buffer.
    // d_octree_sig_sizes is the per-block significance-size table.
    // d_inv_scale is the device inv_scale published for stage-B decode.
    unsigned char* d_octree_coded;
    size_t* d_octree_sig_sizes;
    float* d_inv_scale;

    double* d_partial_sums;
    int     max_copy_blocks;
    double* d_rms;

    hipStream_t aux_stream;
    // Read fresh on every hipCompress call, so it may be changed between calls
    // (but not while compress_pending is true) to move the split per snapshot.
    hipCompressAuxStage aux_from;
    hipEvent_t  ready_event;
    size_t* h_staging;  // pinned host, 2 values: [offsets[nb-1], sizes[nb-1]]

    // Stream the D2H readback of the pending compress was enqueued on, i.e. the
    // one hipCompressSynchronize must block on. Equals aux_stream except at
    // AUX_NONE, where the tail stays on the caller's user_stream and syncing
    // aux_stream would return without waiting for anything.
    hipStream_t pending_stream;
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
// aux_stream runs the stages selected by aux_from. User-owned, shareable across
// plans. One plan must not be used concurrently from multiple host threads.
// Compress is exclusive (writes plan buffers).
hipError_t hipCompressCreatePlan(
    hipCompressPlan** plan,
    int nx, int ny, int nz,
    hipStream_t aux_stream,
    hipCompressKernel kernel = HIP_COMPRESS_KERNEL_AUTO,
    hipCompressAuxStage aux_from = HIP_COMPRESS_AUX_FROM_COMPACT);

hipError_t hipCompressDestroyPlan(hipCompressPlan* plan);

// Copy from a strided source volume to a 32-divisible wavelet-layout buffer.
// Copies ex*ey*ez samples starting at (x0,y0,z0) in the source grid.
// Zero-fills the padding band (ex..wnx-1, etc.).
// RMS is computed over the ex*ey*ez extraction window only.
//
// 3D: ex,ey,ez >= 32. Wavelet dims: wnx = round32(ex), etc.
// 2D (nz=1): ex,ey >= 32, ez must be 1, wnz = 1.
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

// Convenience wrapper: RMS-only mode of hipCopyToWaveletLayout.
// Computes RMS over the ex*ey*ez extraction window; writes no wavelet buffer.
// Equivalent to hipCopyToWaveletLayout(..., d_dst=NULL, d_rms_out, ...).
// Returns HIP_COMPRESS_ERROR_NULL_OUTPUT if d_rms_out is NULL.
hipError_t hipComputeRMS(
    const float* d_src,
    int ldimx, int ldimxy,
    int x0, int y0, int z0,
    int ex, int ey, int ez,
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

// Wavelet transform + quantize + entropy encode → self-contained stream.
// Fully async — returns immediately after queuing all GPU work.
// d_input must be a wavelet-layout buffer (32-divisible dims matching plan).
//
// scale + d_rms control the quantization multiplier (mulfac):
//   d_rms != NULL: mulfac = 1 / (rms * scale).  scale is the error tolerance;
//                  smaller scale → finer quantization → lower error, lower CR.
//   d_rms == NULL: mulfac = scale.  Caller supplies mulfac directly.
//
// The fused wavelet + quantize + significance pass always runs on user_stream.
// Which of the later stages run on aux_stream is set by plan->aux_from (see
// hipCompressAuxStage); they are bridged across with an internal event. Passing
// the same stream as both aux_stream and user_stream makes the split a no-op
// regardless of aux_from.
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

// Entropy decode + inverse wavelet → wavelet-layout buffer.
// Reads the self-contained header (block offsets, mulfac) from d_input.
// Single kernel launch on user_stream. No sync.
hipError_t hipDecompress(
    const unsigned char* d_input,
    float* d_output,
    hipCompressPlan* plan,
    hipStream_t user_stream);

// Uncompressed wavelet-layout buffer size in bytes (for allocation).
hipError_t hipCompressBufferSize(const hipCompressPlan* plan, size_t* size);

// Upper bound on compressed output size in bytes (for allocation).
hipError_t hipCompressMaxOutputSize(const hipCompressPlan* plan, size_t* size);

#endif
