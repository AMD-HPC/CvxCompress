// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef HIP_RLE_DECODE_H
#define HIP_RLE_DECODE_H

// GPU z-line RLE decoder. Reverses the encoding in wrle_zline (hipWaveletRLE.h).
// Reads a variable-length byte stream and produces 32 dequantized float values.

#include <hip/hip_runtime.h>
#include "Run_Length_Escape_Codes.hxx"

// Decode a single z-line from a byte stream.
// src: encoded bytes, encoded_bytes: length of encoded data
// inv_scale: dequantization factor (= 1/scale = RMS * user_scale)
// out: 32 decoded float values
__device__ __forceinline__
void wrle_zline_decode(const unsigned char* src, int encoded_bytes,
                       float inv_scale, float* out)
{
    int num = 0;
    int pos = 0;
    while (num < 32 && pos < encoded_bytes) {
        signed char byte = (signed char)src[pos];
        if (byte > (signed char)VLESC2 && byte < (signed char)RLESC3) {
            out[num++] = (float)(int)byte * inv_scale;
            pos += 1;
        } else if (byte == (signed char)RLESC1) {
            int run = (unsigned char)src[pos + 1];
            for (int j = 0; j < run && num < 32; ++j)
                out[num++] = 0.0f;
            pos += 2;
        } else if (byte == (signed char)RLESC3) {
            int run = (unsigned char)src[pos+1]
                    | ((unsigned char)src[pos+2] << 8)
                    | ((unsigned char)src[pos+3] << 16);
            for (int j = 0; j < run && num < 32; ++j)
                out[num++] = 0.0f;
            pos += 4;
        } else if (byte == (signed char)VLESC2) {
            short quant;
            __builtin_memcpy(&quant, src + pos + 1, sizeof(short));
            out[num++] = (float)(int)quant * inv_scale;
            pos += 3;
        } else if (byte == (signed char)VLESC3) {
            int quant = (unsigned char)src[pos+1]
                      | ((unsigned char)src[pos+2] << 8)
                      | ((signed char)src[pos+3] << 16);
            out[num++] = (float)quant * inv_scale;
            pos += 4;
        } else if (byte == (signed char)VLESC4) {
            float fval;
            __builtin_memcpy(&fval, src + pos + 1, sizeof(float));
            out[num++] = fval * inv_scale;
            pos += 5;
        }
    }
    while (num < 32) out[num++] = 0.0f;
}

// Variant that decodes to integers (no dequantization). For testing.
__device__ __forceinline__
int wrle_zline_decode_int(const unsigned char* src, int encoded_bytes, int* out)
{
    int num = 0;
    int pos = 0;
    while (num < 32 && pos < encoded_bytes) {
        signed char byte = (signed char)src[pos];
        if (byte > (signed char)VLESC2 && byte < (signed char)RLESC3) {
            out[num++] = (int)byte;
            pos += 1;
        } else if (byte == (signed char)RLESC1) {
            int run = (unsigned char)src[pos + 1];
            for (int j = 0; j < run && num < 32; ++j)
                out[num++] = 0;
            pos += 2;
        } else if (byte == (signed char)RLESC3) {
            int run = (unsigned char)src[pos+1]
                    | ((unsigned char)src[pos+2] << 8)
                    | ((unsigned char)src[pos+3] << 16);
            for (int j = 0; j < run && num < 32; ++j)
                out[num++] = 0;
            pos += 4;
        } else if (byte == (signed char)VLESC2) {
            short quant;
            __builtin_memcpy(&quant, src + pos + 1, sizeof(short));
            out[num++] = (int)quant;
            pos += 3;
        } else if (byte == (signed char)VLESC3) {
            int quant = (unsigned char)src[pos+1]
                      | ((unsigned char)src[pos+2] << 8)
                      | ((signed char)src[pos+3] << 16);
            out[num++] = quant;
            pos += 4;
        } else if (byte == (signed char)VLESC4) {
            float fval;
            __builtin_memcpy(&fval, src + pos + 1, sizeof(float));
            out[num++] = (int)fval;
            pos += 5;
        }
    }
    while (num < 32) out[num++] = 0;
    return num;
}

#endif // HIP_RLE_DECODE_H
