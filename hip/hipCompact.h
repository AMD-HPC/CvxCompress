// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIPCVXCOMPRESS_HIP_COMPACT_H
#define HIPCVXCOMPRESS_HIP_COMPACT_H

#include <hip/hip_runtime.h>
#include <rocprim/device/device_scan.hpp>

// Header used by compacted streams without a separate significance-size table.
// Layout: [num_blocks][num_mulfacs][block offsets][quantization multipliers].
inline int hipCompressHeaderSize(int num_blocks, int num_mulfacs)
{
    return 8 + 8 * num_blocks + 4 * num_mulfacs;
}

inline hipError_t hipCompressCompactScanTempSize(
    int num_blocks, size_t* scan_temp_bytes)
{
    return rocprim::exclusive_scan(
        nullptr, *scan_temp_bytes,
        (size_t*)nullptr, (size_t*)nullptr, (size_t)0, (size_t)num_blocks,
        rocprim::plus<size_t>());
}

#endif
