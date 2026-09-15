// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIP_CODEC_COMMON_H
#define HIP_CODEC_COMMON_H

#include <hip/hip_runtime.h>

__host__ __device__ __forceinline__ int hipcvx_quantize_i32(float value)
{
#if defined(__HIP_DEVICE_COMPILE__)
    // The upper endpoint is the largest float below 2^31.
    float clamped = __builtin_amdgcn_fmed3f(
        value, -2147483648.0f, 2147483520.0f);
    return (int)clamped;
#else
    if (!(value == value)) return 0;
    if (value >= 2147483520.0f) return 2147483520;
    if (value <= -2147483648.0f) return (-2147483647 - 1);
    return (int)value;
#endif
}

__host__ __device__ __forceinline__ unsigned hipcvx_abs_i32(int value)
{
    return value < 0 ? 0u - (unsigned)value : (unsigned)value;
}

// Minimum signed byte width that holds [-maxabs, maxabs].
__host__ __device__ __forceinline__ int hipcvx_width_bytes(unsigned maxabs)
{
    if (maxabs <= 0x7fu)     return 1;
    if (maxabs <= 0x7fffu)   return 2;
    if (maxabs <= 0x7fffffu) return 3;
    return 4;
}

#endif // HIP_CODEC_COMMON_H
