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

#endif // HIP_TEST_COMMON_H
