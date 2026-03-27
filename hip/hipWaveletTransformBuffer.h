// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIPCVXCOMPRESS_HIP_WAVELET_TRANSFORM_BUFFER_H
#define HIPCVXCOMPRESS_HIP_WAVELET_TRANSFORM_BUFFER_H

#include <hip/hip_runtime.h>

// All-register kernel: 32 planes in VGPRs, buffer instructions for ld/st.
template<int T, int TargetOcc>
hipError_t hipWaveletTransformBuffer(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// Buffer kernel with Z-direction wavelet transform (ds79 on float4_vec).
template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZ(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// Buffer kernel with Z-direction wavelet (scalar tmp, one component at a time).
template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZScalarTmp(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// saddr workaround: float4 nt with uint8_t* + uint32_t offset trick.
template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddr(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// Baseline: standard float4 pointer loads/stores, 64-bit addressing.
template<int T, int TargetOcc>
hipError_t hipWaveletTransformBaseline(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// Scalar baseline: individual float loads/stores, 64-bit addressing.
template<int T, int TargetOcc>
hipError_t hipWaveletTransformScalarNTBaseline(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// saddr workaround with Z-direction wavelet (float4_vec tmp).
template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZ(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// Buffer Z+Y wavelet (Z in registers, Y via LDS transpose).
template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZY(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// Buffer Z+Y wavelet with XOR bank-conflict avoidance.
template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZYXor(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// saddr Z+Y wavelet with XOR bank-conflict avoidance.
template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZYXor(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// saddr workaround with Z+Y wavelet (Z in registers, Y via LDS transpose).
template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZY(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// saddr workaround with Z-direction wavelet (scalar tmp, one component at a time).
template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZScalarTmp(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// Scalar baseline (no nontemporal): individual float loads/stores, 64-bit addressing.
template<int T, int TargetOcc>
hipError_t hipWaveletTransformScalarBaseline(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// Buffer Z+Y+X wavelet (Z in registers, Y and X via LDS).
template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZYX(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// saddr Z+Y+X wavelet (Z in registers, Y and X via LDS).
template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZYX(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// Buffer Z+Y wavelet with grouped XOR (b128-friendly, 4-way conflicts).
template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferZYXor4(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// saddr Z+Y wavelet with grouped XOR (b128-friendly, 4-way conflicts).
template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrZYXor4(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// Pipelined XYZ (buffer): overlaps global loads with LDS XY transforms, then Z in regs.
template<int T, int TargetOcc>
hipError_t hipWaveletTransformBufferPipeXYZ(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

// Pipelined XYZ (saddr): overlaps global loads with LDS XY transforms, then Z in regs.
template<int T, int TargetOcc>
hipError_t hipWaveletTransformSaddrPipeXYZ(
    float* data,
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int ldimx, int ldimxy);

#endif
