// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIP_TEST_COMMON_H
#define HIP_TEST_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define HIPCHECK(cmd) do { \
    hipError_t hipcheck_error = (cmd); \
    if (hipcheck_error != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", \
                hipGetErrorString(hipcheck_error), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// Busy-waits on the device for `ticks` wall-clock ticks (100 MHz on AMD GPUs,
// so 1e6 ticks is about 10 ms). Queue it on a stream to hold that stream's later
// work back. A test that needs cross-stream ordering then fails reliably when the
// ordering is missing, instead of passing because the kernels were too short to
// overlap.
__global__ void spinKernel(long long ticks)
{
    const long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {}
}

#endif // HIP_TEST_COMMON_H
