// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIPCVXCOMPRESS_HIP_WAVELET_TRANSFORM_BLOCKED_H
#define HIPCVXCOMPRESS_HIP_WAVELET_TRANSFORM_BLOCKED_H

#include <hip/hip_runtime.h>

// 2D thread block (8x32) batched kernel with inline asm.
// Out-of-place: reads from input, writes to output.
template<int Nplanes, int Batch>
hipError_t hipWaveletTransform3DBlockedForward2DImpl(
    const float* input,
    float* output,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// 3D wavelet forward — Y-direction only (first step toward full 3D).
hipError_t hipWaveletTransform3DForwardY(
    const float* input,
    float* output,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// All-register kernel: no shared memory, all planes in VGPRs.
template<int T, int TargetOcc>
hipError_t hipWaveletTransform3DBlockedAllReg(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

#endif
