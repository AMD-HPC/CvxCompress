// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// Single global-memory plane-addressing path for the wavefield/wavelet buffers.
//
// The codec kernels used to read/write planes with buffer instructions
// (make_buffer_rsrc + raw_buffer_load/store_*), whose intra-plane byte offset
// is 32-bit -- capping a single plane at 4 GB (nx*ny*4 <= 2^32).  A/B
// benchmarking (buffer vs global "saddr" load, uint32 vs size_t offset) showed
// the global path matches or beats buffer on both gfx942 and gfx950 with equal
// or lower register pressure, and that widening the offset to 64 bits is free.
// So every plane access now uses a 64-bit base pointer plus a size_t byte
// offset, lifting the 4 GB per-plane limit.  Loads/stores are nontemporal to
// preserve the previous SLC (streaming) cache hint.
//
// Note: buffer instructions also provided free out-of-bounds clamping via
// num_records.  Global loads/stores do not, so any kernel that could touch a
// ragged (non-32-multiple) edge must bound-check explicitly.  The fused codec
// kernels operate only on 32-aligned full tiles (the public API mandates
// 32-multiple dims), so they never go out of bounds; the block-copy kernels,
// which handle arbitrary extraction windows, carry explicit per-lane guards.

#pragma once

#include <cstddef>
#include <cstdint>

// Vector load of V (e.g. float4 / int4) from base + byte_off, nontemporal.
template <typename V>
__device__ __forceinline__
V hipPlaneLoadNT(const float* base, size_t byte_off)
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(base) + byte_off;
    return __builtin_nontemporal_load(reinterpret_cast<const V*>(p));
}

// Vector store of V to base + byte_off, nontemporal.
template <typename V>
__device__ __forceinline__
void hipPlaneStoreNT(float* base, size_t byte_off, V v)
{
    uint8_t* p = reinterpret_cast<uint8_t*>(base) + byte_off;
    __builtin_nontemporal_store(v, reinterpret_cast<V*>(p));
}

// Alignment-tolerant float4 load/store.  Block-copy touches the caller's array
// with arbitrary strides/origins (ldimx, x0), so an intra-plane byte offset is
// only guaranteed 4-byte aligned (everything is elem*sizeof(float)), not 16.
// A 128-bit global load needs just 4-byte element alignment, so declaring the
// vector aligned(4) avoids the natural-16B assumption of hipPlaneLoadNT<float4>
// (which faults on, e.g., x0=10 -> offset 8 mod 16).  The fused codec kernels
// keep the 16-aligned path (nx is a 32-multiple, so offsets are 16-aligned).
using hip_float4_u = float __attribute__((ext_vector_type(4), __aligned__(4)));

__device__ __forceinline__
hip_float4_u hipPlaneLoadF4u(const float* base, size_t byte_off)
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(base) + byte_off;
    return __builtin_nontemporal_load(reinterpret_cast<const hip_float4_u*>(p));
}

__device__ __forceinline__
void hipPlaneStoreF4u(float* base, size_t byte_off, hip_float4_u v)
{
    uint8_t* p = reinterpret_cast<uint8_t*>(base) + byte_off;
    __builtin_nontemporal_store(v, reinterpret_cast<hip_float4_u*>(p));
}

// Scalar float load/store for ragged-edge lanes.
__device__ __forceinline__
float hipPlaneLoadScalarNT(const float* base, size_t byte_off)
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(base) + byte_off;
    return __builtin_nontemporal_load(reinterpret_cast<const float*>(p));
}

__device__ __forceinline__
void hipPlaneStoreScalarNT(float* base, size_t byte_off, float v)
{
    uint8_t* p = reinterpret_cast<uint8_t*>(base) + byte_off;
    __builtin_nontemporal_store(v, reinterpret_cast<float*>(p));
}
