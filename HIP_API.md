# hipCVXCompress — HIP GPU Compression API

> **Experimental / Proof of Concept** — APIs are unstable and may change without
> notice. Optimizations are ongoing. Not tested in an integrated production
> setting. No backward compatibility guarantee for the compressed bitstream
> format. Compression ratios differ from the CPU reference due to different
> block tiling strategies.

GPU-accelerated lossy compression for 3D floating-point volumes on AMD Instinct
GPUs (MI200, MI300). Targets seismic imaging workloads where wavefield snapshots
must be stored and retrieved at GPU memory bandwidth.

Single fused kernel: wavelet transform (DS 7/9) → quantization → RLE encoding.
Error norms match the CPU reference (CvxCompress) to floating-point rounding.

### Performance (MI300X vs 128-core EPYC 9554, AVX, best thread count)

| Volume | GPU fwd (ms) | CPU best fwd (ms) | CPU threads | Speedup |
|--------|-------------|-------------------|-------------|---------|
| 256^3 (64 MB)   | 0.14 | 3.7  | 32  | 27x |
| 512^3 (512 MB)  | 0.72 | 14.6 | 128 | 20x |
| 1024^3 (4 GB)   | 5.4  | 76.2 | 64  | 14x |

The GPU advantage is larger in production because compression runs concurrently
with simulation on a separate stream, so the effective cost is near-zero overlap
time rather than end-to-end latency.

## Requirements

- ROCm 7.x (`module load rocm/7.2.0`)
- AMD GPU: gfx90a (MI200) or gfx942 (MI300X)
- C++17, `hipcc`, `rocprim`

## Building

```bash
module load rocm/7.2.0

# Build the CPU reference library (needed by tests)
make libcvxcompress.so

# Build the API test suite (37 tests + benchmarks)
make HIP_ARCH=gfx942 test_compress_api_hip

# Build the async pipeline example
make HIP_ARCH=gfx942 example_async_pipeline
```

Set `HIP_ARCH` to match your target: `gfx90a` for MI200, `gfx942` for MI300X.

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
| `hipCopyFromWaveletLayout` | Copy from wavelet buffer → strided grid (extraction window only) |

### Compression / Decompression

| Function | Description |
|----------|-------------|
| `hipCompress` | Wavelet + quantize + RLE encode → self-contained compressed stream (async) |
| `hipCompressSynchronize` | Block until compress completes, retrieve compressed length and CR |
| `hipDecompress` | RLE decode + inverse wavelet → wavelet buffer (single kernel, async) |

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

See [`tests/example_async_pipeline.cpp`](tests/example_async_pipeline.cpp) for a
complete example that overlaps simulation compute on `user_stream` with compression
on `aux_stream`. The two-stream design lets the simulation proceed immediately
after `hipCompress` returns — compaction and host readback run concurrently on the
auxiliary stream.

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

| Variant | Enum | Description |
|---------|------|-------------|
| Z-line  | `HIP_COMPRESS_KERNEL_ZLINE` | Parallel z-line RLE with per-block metadata (default) |
| Seg-RLE | `HIP_COMPRESS_KERNEL_SEGRLE` | Segment-aligned RLE, no metadata overhead (unoptimized) |

Select at plan creation: `hipCompressCreatePlan(&plan, nx, ny, nz, aux, HIP_COMPRESS_KERNEL_SEGRLE)`.

### Two-Stream Model

- **`user_stream`**: passed to each API call. Wavelet transform and RLE encoding
  run here. The stream is free immediately after `hipCompress` returns.
- **`aux_stream`**: owned by the user, passed at plan creation. Compaction, header
  writing, and D2H readback run here. Shared across plans.

An internal event bridges the two streams. This design lets simulation kernels
continue on `user_stream` while compression finishes on `aux_stream`.

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

- **Plane size**: `nx × ny × 4` must not exceed 4 GB (plan dimensions and source
  grid `ldimxy` are both validated).
- **Minimum dimensions**: extraction window must be ≥ 32 in each axis.
- **Alignment**: all plan dimensions must be multiples of 32.
- **Concurrency**: a plan must not be used from multiple host threads. One
  `hipCompress` must be synchronized before the next.
- **Data type**: `float` only (single precision).

## File Structure

```
hip/
  hipCompress.h                  Public API header
  hipCompress.cpp                API implementation
  hipBlockCopy.h                 CopyTo / CopyFrom kernels
  hipWaveletRLE.h                Fused forward wavelet + RLE kernels
  hipWaveletRLEInverse.h         Fused inverse RLE + wavelet kernels
  hipRLEDecode.h                 Z-line RLE decoder
  hipSegmentedRLE.h              Segment-aligned RLE encode/decode
  ds79.h                         DS 7/9 wavelet filter coefficients and transforms
  ds79_reg32.inc                 Unrolled forward wavelet (32-point)
  us79_reg32.inc                 Unrolled inverse wavelet (32-point)
tests/
  test_compress_api_hip.cpp      API test suite (37 tests + benchmarks)
  example_async_pipeline.cpp     Async overlap example
```

## License

Copyright (C) 2025 Advanced Micro Devices, Inc. Licensed under the
[MIT License](https://opensource.org/licenses/MIT).
