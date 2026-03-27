// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIP_SEGMENTED_RLE_H
#define HIP_SEGMENTED_RLE_H

// Segment-aligned RLE encoding/decoding.
//
// Format: continuous byte stream with null padding at segment boundaries.
// Every SEG_SIZE-byte boundary is a valid RLE decoder entry point.
//
// Changes vs legacy z-line RLE:
//   - 0x00 is null/padding (was: inline zero value)
//   - VLESC2_8x (0x82, -126) encodes a single zero (was: unused)
//   - No per-z-line metadata; data encoded in raster order (x, y, plane)
//   - Zero runs can span across what were z-line boundaries

#include <hip/hip_runtime.h>
#include "Run_Length_Escape_Codes.hxx"

static constexpr int SEG_RLE_SEG_SIZE = 128;

// ---- Encoder helpers (thread 0 only) ----

// Pad with null bytes to the next segment boundary.
__device__ __forceinline__
void seg_rle_pad_to_boundary(unsigned char* out, int* bp, int seg_size)
{
    int rem = *bp % seg_size;
    if (rem != 0) {
        int pad = seg_size - rem;
        for (int i = 0; i < pad; ++i)
            out[(*bp)++] = 0x00;
    }
}

// Ensure code_size bytes fit before the next segment boundary.
// If not, pad with nulls to the boundary.
__device__ __forceinline__
void seg_rle_ensure_fit(unsigned char* out, int* bp, int code_size, int seg_size)
{
    int rem = *bp % seg_size;
    int remaining = (rem == 0) ? seg_size : seg_size - rem;
    if (code_size > remaining) {
        for (int i = 0; i < remaining; ++i)
            out[(*bp)++] = 0x00;
    }
}

// Flush a pending zero run, respecting segment boundaries.
__device__ __forceinline__
void seg_rle_flush_zeros(unsigned char* out, int* bp, int* rle, int seg_size)
{
    while (*rle > 0) {
        if (*rle == 1) {
            seg_rle_ensure_fit(out, bp, 1, seg_size);
            out[(*bp)++] = (unsigned char)(VLESC2_8x & 0xFF);
            *rle = 0;
        } else if (*rle < 256) {
            seg_rle_ensure_fit(out, bp, 2, seg_size);
            out[*bp]     = (unsigned char)(RLESC1 & 0xFF);
            out[*bp + 1] = (unsigned char)*rle;
            *bp += 2;
            *rle = 0;
        } else {
            int run = *rle;
            if (run > 0xFFFFFF) run = 0xFFFFFF;
            seg_rle_ensure_fit(out, bp, 4, seg_size);
            out[*bp]     = (unsigned char)(RLESC3 & 0xFF);
            out[*bp + 1] = (unsigned char)(run & 0xFF);
            out[*bp + 2] = (unsigned char)((run >> 8) & 0xFF);
            out[*bp + 3] = (unsigned char)((run >> 16) & 0xFF);
            *bp += 4;
            *rle -= run;
        }
    }
}

// Emit a non-zero quantized value, respecting segment boundaries.
__device__ __forceinline__
void seg_rle_emit_value(unsigned char* out, int* bp,
                        int ival, float fval, int seg_size)
{
    bool ib = (ival > VLESC2) && (ival < RLESC3) && (ival != 0);
    bool i16 = (ival >= -32768) && (ival <= 32767);
    bool i24 = (ival >= -8388608) && (ival <= 8388607);

    if (ib) {
        seg_rle_ensure_fit(out, bp, 1, seg_size);
        out[(*bp)++] = (unsigned char)(signed char)ival;
    } else if (i16) {
        seg_rle_ensure_fit(out, bp, 3, seg_size);
        out[*bp]     = (unsigned char)(signed char)VLESC2;
        out[*bp + 1] = (unsigned char)((unsigned)ival & 0xFF);
        out[*bp + 2] = (unsigned char)(((unsigned)ival >> 8) & 0xFF);
        *bp += 3;
    } else if (i24) {
        seg_rle_ensure_fit(out, bp, 4, seg_size);
        out[*bp]     = (unsigned char)(signed char)VLESC3;
        out[*bp + 1] = (unsigned char)((unsigned)ival & 0xFF);
        out[*bp + 2] = (unsigned char)(((unsigned)ival >> 8) & 0xFF);
        out[*bp + 3] = (unsigned char)(((unsigned)ival >> 16) & 0xFF);
        *bp += 4;
    } else {
        unsigned u;
        __builtin_memcpy(&u, &fval, 4);
        seg_rle_ensure_fit(out, bp, 5, seg_size);
        out[*bp]     = (unsigned char)(signed char)VLESC4;
        out[*bp + 1] = (unsigned char)(u & 0xFF);
        out[*bp + 2] = (unsigned char)((u >> 8) & 0xFF);
        out[*bp + 3] = (unsigned char)((u >> 16) & 0xFF);
        out[*bp + 4] = (unsigned char)((u >> 24) & 0xFF);
        *bp += 5;
    }
}

// Encode one batch of planes from LDS wavelet buffer.
// LDS layout: lds_wav[plane * 1024 + x * 32 + (y ^ x)] (XOR bank-free).
// Raster order: x fastest, then y, then plane.
// Thread 0 only. rle and bp carry across batches.
__device__ inline
void seg_rle_encode_batch(const float* lds_wav, float mulfac,
                          unsigned char* out,
                          int batch_planes, int seg_size,
                          int* bp, int* rle)
{
    for (int p = 0; p < batch_planes; ++p) {
        for (int y = 0; y < 32; ++y) {
            for (int x = 0; x < 32; ++x) {
                float fval = mulfac * lds_wav[p * 1024 + x * 32 + (y ^ x)];
                int ival = (int)fval;
                if (ival == 0) {
                    ++(*rle);
                } else {
                    seg_rle_flush_zeros(out, bp, rle, seg_size);
                    seg_rle_emit_value(out, bp, ival, fval, seg_size);
                }
            }
        }
    }
}

// Finalize: flush remaining zeros and pad to segment boundary.
__device__ inline
void seg_rle_finalize(unsigned char* out, int* bp, int* rle, int seg_size)
{
    seg_rle_flush_zeros(out, bp, rle, seg_size);
    seg_rle_pad_to_boundary(out, bp, seg_size);
}

// ---- Decoder (thread 0 only) ----

// Decode values from a segment-aligned RLE stream and write to LDS
// in the same raster order used by the encoder.
// Decodes exactly batch_planes*1024 values per call.
// stream_pos and pending_zeros carry across batches.
__device__ inline
void seg_rle_decode_batch(const unsigned char* input,
                          float inv_scale,
                          float* lds_wav,
                          int batch_planes, int seg_size,
                          int* stream_pos, int* pending_zeros)
{
    int pos = *stream_pos;
    int pz  = *pending_zeros;
    int count = 0;
    int target = batch_planes * 1024;

    auto write_zero = [&]() __device__ {
        int p = count / 1024, rem = count % 1024;
        int y = rem / 32, x = rem % 32;
        lds_wav[p * 1024 + x * 32 + (y ^ x)] = 0.0f;
        count++;
    };

    // Drain any pending zeros from a previous batch's partial run
    while (pz > 0 && count < target) { write_zero(); pz--; }

    while (count < target) {
        unsigned char raw = input[pos];
        if (raw == 0x00) {
            pos = ((pos / seg_size) + 1) * seg_size;
            continue;
        }
        signed char byte = (signed char)raw;

        if (byte == (signed char)VLESC2_8x) {
            write_zero();
            pos++;
        } else if (byte > (signed char)VLESC2 && byte < (signed char)RLESC3) {
            float val = (float)(int)byte * inv_scale;
            int p = count / 1024, rem = count % 1024;
            int y = rem / 32, x = rem % 32;
            lds_wav[p * 1024 + x * 32 + (y ^ x)] = val;
            count++;
            pos++;
        } else if (byte == (signed char)RLESC1) {
            int run = (int)(unsigned char)input[pos + 1];
            pos += 2;
            while (run > 0 && count < target) { write_zero(); run--; }
            pz = run;
        } else if (byte == (signed char)RLESC3) {
            int run = (int)(unsigned char)input[pos+1]
                    | ((int)(unsigned char)input[pos+2] << 8)
                    | ((int)(unsigned char)input[pos+3] << 16);
            pos += 4;
            while (run > 0 && count < target) { write_zero(); run--; }
            pz = run;
        } else if (byte == (signed char)VLESC2) {
            short quant;
            __builtin_memcpy(&quant, input + pos + 1, sizeof(short));
            float val = (float)(int)quant * inv_scale;
            int p = count / 1024, rem = count % 1024;
            int y = rem / 32, x = rem % 32;
            lds_wav[p * 1024 + x * 32 + (y ^ x)] = val;
            count++;
            pos += 3;
        } else if (byte == (signed char)VLESC3) {
            int quant = (int)(unsigned char)input[pos+1]
                      | ((int)(unsigned char)input[pos+2] << 8)
                      | ((int)(signed char)input[pos+3] << 16);
            float val = (float)quant * inv_scale;
            int p = count / 1024, rem = count % 1024;
            int y = rem / 32, x = rem % 32;
            lds_wav[p * 1024 + x * 32 + (y ^ x)] = val;
            count++;
            pos += 4;
        } else if (byte == (signed char)VLESC4) {
            float fval;
            __builtin_memcpy(&fval, input + pos + 1, sizeof(float));
            float val = fval * inv_scale;
            int p = count / 1024, rem = count % 1024;
            int y = rem / 32, x = rem % 32;
            lds_wav[p * 1024 + x * 32 + (y ^ x)] = val;
            count++;
            pos += 5;
        } else {
            pos++;
        }
    }
    *stream_pos = pos;
    *pending_zeros = pz;
}

#endif // HIP_SEGMENTED_RLE_H
