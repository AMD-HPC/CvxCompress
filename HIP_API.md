# hipCVXCompress — HIP GPU Compression API

> **Experimental / Proof of Concept** — APIs are unstable and may change without
> notice. Optimizations are ongoing. Not tested in an integrated production
> setting. No backward compatibility guarantee for the compressed bitstream
> format. Compression ratios differ from the CPU reference due to different
> block tiling strategies.

GPU-accelerated lossy compression for 2D and 3D floating-point volumes on AMD
Instinct GPUs (MI200, MI300, MI355). Targets seismic imaging workloads where
wavefield snapshots must be stored and retrieved at GPU memory bandwidth.

The pipeline applies the DS 7/9 wavelet transform, quantizes the coefficients,
and codes their significance and values. The coder is selectable per plan; the default
resolves to octree for 3D and quadtree for 2D.
Error norms match the CPU reference (CvxCompress) to floating-point rounding.

The API can move the scan, compaction, and readback tail to a separate stream.
The best stream placement depends on the caller's GPU workload. Profile the
integrated pipeline instead of assuming overlap will hide compression.

## Requirements

- ROCm 7.x (`module load rocm/7.2.1`)
- AMD GPU: gfx90a (MI200), gfx942 (MI300X), or gfx950 (MI355X)
- C++17, `hipcc`, `rocprim`

## Building

```bash
module load rocm/7.2.1

# Build the CPU reference library (needed by tests)
make libcvxcompress.so

# Build the retained correctness suites and performance benchmark
make HIP_ARCH=gfx942 test_compress_api_hip
make HIP_ARCH=gfx942 test_compress_2d_hip
make HIP_ARCH=gfx942 test_async_overlap_3d
make HIP_ARCH=gfx942 test_inverse_wavelet_hip
make HIP_ARCH=gfx942 bench_encode_full
```

Set `HIP_ARCH` to match your target: `gfx90a` for MI200, `gfx942` for MI300X,
or `gfx950` for MI355X.

## API Overview

All functions are declared in [`hip/hipCompress.h`](hip/hipCompress.h).

### Plan Management

| Function | Description |
|----------|-------------|
| `hipCompressCreatePlan` | Allocate plan and internal buffers for given wavelet dimensions |
| `hipCompressDestroyPlan` | Free plan and all internal buffers |

### Data Layout

| Function | Description |
|----------|-------------|
| `hipCopyToWaveletLayout` | Copy from strided grid → contiguous wavelet buffer (zero-pad + optional RMS) |
| `hipComputeRMS` | Convenience wrapper: RMS over extraction window only (no copy) |
| `hipCopyFromWaveletLayout` | Copy from wavelet buffer → strided grid (extraction window only) |

### Compression / Decompression

| Function | Description |
|----------|-------------|
| `hipCompress` | Wavelet + quantize + encode (per-plan codec) → self-contained compressed stream (async) |
| `hipCompressSynchronize` | Block until compress completes, retrieve compressed length and CR |
| `hipDecompress` | Decode (per-plan codec) + inverse wavelet → wavelet buffer (async) |

### Utilities

| Function | Description |
|----------|-------------|
| `hipCompressMaxOutputSize` | Upper bound on compressed output size (for allocation) |
| `hipCompressWaveletDims` | Round dimensions up to multiples of 32 |
| `hipCompressGetLastError` | Library-specific error code from last call |
| `hipCompressErrorString` | Human-readable error message |

## Usage

### Minimal Round-Trip

```cpp
#include "hipCompress.h"

// User's 3D grid: 200 x 300 x 100 sub-domain in a larger volume
const int ldimx = 512, ldimy = 512;          // leading dims of full allocation
const int ldimxy = ldimx * ldimy;
const int ex = 200, ey = 300, ez = 100;      // extraction window size
const int ex0 = 10, ey0 = 20, ez0 = 5;      // extraction origin in source grid

// 1. Compute wavelet dimensions (round up to multiples of 32)
int wnx, wny, wnz;
hipCompressWaveletDims(ex, ey, ez, &wnx, &wny, &wnz);  // → 224, 320, 128

// 2. Create plan and streams
hipStream_t user_stream, aux_stream;
hipStreamCreate(&user_stream);
hipStreamCreateWithFlags(&aux_stream, hipStreamNonBlocking);

hipCompressPlan* plan = nullptr;
hipCompressCreatePlan(&plan, wnx, wny, wnz, aux_stream);

// 3. Allocate device buffers
float* d_grid;          // user's source/destination grid (already allocated)
float* d_wavelet;       // wavelet-layout buffer
unsigned char* d_comp;  // compressed output
size_t comp_cap;

hipMalloc(&d_wavelet, (size_t)wnx * wny * wnz * sizeof(float));
hipCompressMaxOutputSize(plan, &comp_cap);
hipMalloc(&d_comp, comp_cap);

// 4. Copy to wavelet layout (with RMS computation for adaptive quantization)
//    Padding band (ex..wnx-1, etc.) is zero-filled.
hipCopyToWaveletLayout(
    d_grid, ldimx, ldimxy,
    ex0, ey0, ez0, ex, ey, ez,
    d_wavelet, plan->d_rms, plan, user_stream);

// 5. Compress (async — user_stream is free after this returns)
float scale = 5e-2f;   // error tolerance relative to RMS
hipCompress(scale, plan->d_rms, d_wavelet, d_comp, plan, user_stream);

// 6. Synchronize and get result
long compressed_bytes;
float compression_ratio;
hipCompressSynchronize(plan, &compressed_bytes, &compression_ratio);

// ... store d_comp[0..compressed_bytes-1] to disk or transfer ...

// 7. Decompress (later, possibly different stream)
hipDecompress(d_comp, d_wavelet, plan, user_stream);

// 8. Copy back to user's grid
hipCopyFromWaveletLayout(
    d_wavelet, d_grid, ldimx, ldimxy,
    ex0, ey0, ez0, ex, ey, ez, plan, user_stream);

// 9. Clean up
hipCompressDestroyPlan(plan);
hipFree(d_wavelet);
hipFree(d_comp);
```

### Async Pipeline Overlap

See [`tests/test_async_overlap_3d.cpp`](tests/test_async_overlap_3d.cpp) for the
tested pipeline pattern. It overlaps simulation compute on `user_stream` with
compression on `aux_stream`, then decompresses and checks the result. The
two-stream design lets the simulation proceed immediately after `hipCompress`
returns. Compaction and host readback run concurrently on the auxiliary stream.

## Key Concepts

### Wavelet Dimensions

All plan dimensions must be multiples of 32. Use `hipCompressWaveletDims()` to
round up arbitrary window sizes. The wavelet buffer is a contiguous
`wnx × wny × wnz` float array with no stride gaps.

### Extraction Window

`hipCopyToWaveletLayout` copies `ex × ey × ez` samples starting at origin
`(x0, y0, z0)` in the source grid. The wavelet buffer is `wnx × wny × wnz`
(rounded up to multiples of 32). Positions beyond the extraction window are
zero-filled — zero padding yields better compression ratios than mirror padding
(20–55% smaller compressed output for non-multiple-of-32 volumes).

### Scale Parameter

The `scale` argument to `hipCompress` controls the quality/compression tradeoff:

- With RMS (`d_rms != NULL`): quantization multiplier = `1 / (rms × scale)`.
  `scale` is the relative error tolerance. Smaller → finer quantization → lower
  error, lower compression ratio.
- Without RMS (`d_rms == NULL`): `scale` is used directly as the quantization
  multiplier.

### Kernel Variants

| Variant | Enum | Dims | Description |
|---------|------|------|-------------|
| Auto      | `HIP_COMPRESS_KERNEL_AUTO`     | 2D/3D | **Default.** Resolves to quadtree for 2D (`nz == 1`) and octree for 3D |
| Octree    | `HIP_COMPRESS_KERNEL_OCTREE`   | 3D    | Octree significance coder with per-block PFOR value coding. Best compression ratio and fastest decode; recommended for storage/archival |
| Quadtree  | `HIP_COMPRESS_KERNEL_QUADTREE` | 2D    | Quadtree significance coder with per-block PFOR value coding -- the 2D counterpart of octree |

The codec is selected **at runtime, per plan** via the last argument of
`hipCompressCreatePlan` — it is an ordinary function parameter, so switching
between codecs requires **no recompilation** of the library or the application:

```cpp
// default (kernel arg omitted) → auto: octree for 3D, quadtree for 2D
hipCompressCreatePlan(&plan_def, nx, ny, nz, aux);

// storage-oriented volume → octree (best ratio) — explicit form of the 3D default
hipCompressCreatePlan(&plan_oct, nx, ny, nz, aux, HIP_COMPRESS_KERNEL_OCTREE);

```

The codec is bound to the plan (internal buffer sizes and stream header layout
differ per codec), so the switching granularity is "which plan you create"; an
existing plan's codec cannot be changed in place. Octree is 3D only and
quadtree is 2D only. Requesting one for the wrong dimensionality fails
plan creation with `HIP_COMPRESS_ERROR_INVALID_CODEC`. `AUTO` avoids this
by resolving to the dimensionality-appropriate coder at plan creation.

**Choosing a codec.** Use `AUTO` unless you need to pin the dimensionality-specific
codec explicitly.

### Two-Stream Model

The encode chain is

```
transform  ->  encode  ->  scan  ->  compact  ->  D2H readback
```

where *transform* is the fused wavelet + quantize + significance pass and
*encode* is the entropy coder.

- **`user_stream`**: passed to each API call. The transform always runs here; it
  is bandwidth-bound and reads the caller's live input buffer, so moving it off
  `user_stream` only steals HBM from the caller. The stream is free as soon as
  the stages that stayed on it have been enqueued.
- **`aux_stream`**: owned by the user, passed at plan creation. Shared across
  plans. Which stages it picks up is set by the `aux_from` argument to
  `hipCompressCreatePlan`:

| `hipCompressAuxStage` | Runs on `aux_stream` |
|---|---|
| `HIP_COMPRESS_AUX_NONE` | nothing — the whole chain is serial on `user_stream` |
| `HIP_COMPRESS_AUX_FROM_COMPACT` | scan, compaction, header write, D2H readback (**default**) |
| `HIP_COMPRESS_AUX_FROM_ENCODE` | the entropy coder and its size-alignment pass, plus all of the above |

An internal event bridges the two streams at whichever boundary `aux_from`
selects. This lets simulation kernels continue on `user_stream` while the tail
of compression finishes on `aux_stream`.

`aux_from` is a **placement** knob, not a correctness one: all three settings
produce byte-identical output. It is also not a free win. Measured against a
wave propagation kernel that already saturates the GPU (512³ TTI, snapshot every
7 steps, MI355X), `FROM_ENCODE` runs 3–4% *slower* than `NONE` — the entropy
coder is a throughput kernel with no idle slack to reclaim, so co-residency costs
more than the hiding saves. Profile your own caller before moving off the
default. Passing the same stream as both `aux_stream` and `user_stream` makes the
split a no-op regardless of `aux_from`.

## Error Handling

All API functions return `hipError_t`. For library-specific diagnostics:

```cpp
hipError_t err = hipCompress(...);
if (err != hipSuccess) {
    hipCompressError_t detail = hipCompressGetLastError(plan);
    fprintf(stderr, "compress failed: %s\n", hipCompressErrorString(detail));
}
```

## Limitations

- **Plane size**: `nx × ny` must not exceed `INT_MAX` elements. Source strides
  use `int`; the library does not know or validate the source allocation size.
- **Minimum dimensions**: a 3D extraction must be at least 32 in each axis. A
  2D extraction uses `ez = 1` and requires `ex, ey >= 32`.
- **Dimensions**: 3D plan dimensions must be multiples of 32. A 2D plan uses
  `nz = 1` and requires `nx` and `ny` to be multiples of 32.
- **Compressed buffers**: stream bases must be 8-byte aligned. Returned
  compressed lengths preserve this alignment when streams are packed.
- **Compressed input**: `hipDecompress` expects a complete trusted stream
  produced by `hipCompress`; the API does not accept a compressed length.
- **Wavefield values**: compression input must contain finite `float` values.
  NaN and infinity are outside the API contract.
- **Concurrency**: a plan must not be used from multiple host threads. One
  `hipCompress` must be synchronized before the next. Sequential calls may use
  different user streams; the plan orders copy, RMS, compress, and decompress
  work through an internal event.
- **Data type**: `float` only (single precision).

## File Structure

```
hip/
  hipCompress.h                  Public API header
  hipCompress.cpp                API implementation
  hipCompact.h                   Compact stream header and scan helpers
  hipBlockCopy.h                 CopyTo / CopyFrom kernels
  hipCodecCommon.h               Shared quantization and width helpers
  hipWaveletBitmap.h             Bitmap significance split and inverse transform
  hipWaveletOctree.h             Octree significance coder + shared inverse stage
  hipWaveletQuadtree2D.h         Quadtree coder and 2D transform
  ds79.h                         DS 7/9 wavelet filter coefficients and transforms
  ds79_reg32.inc                 Unrolled forward wavelet (32-point)
  us79_reg32.inc                 Unrolled inverse wavelet (32-point)
tests/
  hip_test_common.h              Shared HIP test error handling
  test_compress_api_hip.cpp      3D API regression suite
  test_compress_2d_hip.cpp       2D quadtree regression suite
  test_async_overlap_3d.cpp      Async stream correctness and overlap
  test_inverse_wavelet_hip.cpp   Wavelet primitive regression suite
  bench_encode_full.cpp          End-to-end encode benchmark
```

## License

Copyright (C) 2026 Advanced Micro Devices, Inc. Licensed under the
[MIT License](https://opensource.org/licenses/MIT).
