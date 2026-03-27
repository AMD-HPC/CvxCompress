// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// Scalar CPU reference for z-line quantize + RLE encode/decode.
// Uses the same escape code format as Run_Length_Encode_Slow.cpp.
// Does NOT produce 8x packed codes (VLESC2_8x, VLESC3_8x) — those are
// AVX batch optimizations in the CPU encoder that we skip here.
//
// Two interfaces:
//   int-only:   encode_zline / decode_zline   (for unit testing escape codes)
//   float+scale: quantize_encode_zline / decode_dequantize_zline (full pipeline)

#ifndef QUANTIZE_RLE_REF_H
#define QUANTIZE_RLE_REF_H

#include "Run_Length_Escape_Codes.hxx"
#include <cstring>
#include <cassert>

// -----------------------------------------------------------------------
// RLE flush: write a pending zero-run to the byte stream.
// Matches EncodeRLE_Slow() in Run_Length_Encode_Slow.cpp.
// -----------------------------------------------------------------------
inline int rle_flush(int rle, unsigned char* dst, int bytepos)
{
    if (rle <= 0) return bytepos;
    if (rle == 1) {
        dst[bytepos++] = 0;
    } else if (rle < 256) {
        dst[bytepos++] = (unsigned char)(RLESC1 & 0xFF);
        dst[bytepos++] = (unsigned char)rle;
    } else {
        dst[bytepos++] = (unsigned char)(RLESC3 & 0xFF);
        dst[bytepos++] = rle & 0xFF;
        dst[bytepos++] = (rle >> 8) & 0xFF;
        dst[bytepos++] = (rle >> 16) & 0xFF;
    }
    return bytepos;
}

// -----------------------------------------------------------------------
// Encode pre-quantized integers → byte stream.
// For values beyond int24 range, stores float((float)ival) via VLESC4.
// Returns number of encoded bytes.
// out[] must have space for at least 5*n bytes (worst case).
// -----------------------------------------------------------------------
inline int encode_zline(const int* quantized, int n, unsigned char* out)
{
    int rle = 0;
    int bytepos = 0;

    for (int i = 0; i < n; ++i) {
        int ival = quantized[i];
        if (ival == 0) {
            ++rle;
        } else {
            bytepos = rle_flush(rle, out, bytepos);
            rle = 0;
            if (ival > VLESC2 && ival < RLESC3) {
                out[bytepos++] = (unsigned char)(signed char)ival;
            } else if (ival >= -32768 && ival <= 32767) {
                out[bytepos++] = (unsigned char)(signed char)VLESC2;
                out[bytepos++] = ival & 0xFF;
                out[bytepos++] = (ival >> 8) & 0xFF;
            } else if (ival >= -8388608 && ival <= 8388607) {
                out[bytepos++] = (unsigned char)(signed char)VLESC3;
                out[bytepos++] = ival & 0xFF;
                out[bytepos++] = (ival >> 8) & 0xFF;
                out[bytepos++] = (ival >> 16) & 0xFF;
            } else {
                float fval = (float)ival;
                out[bytepos++] = (unsigned char)(signed char)VLESC4;
                memcpy(out + bytepos, &fval, sizeof(float));
                bytepos += 4;
            }
        }
    }
    bytepos = rle_flush(rle, out, bytepos);
    return bytepos;
}

// -----------------------------------------------------------------------
// Decode byte stream → pre-quantized integers.
// Returns number of values decoded.
// -----------------------------------------------------------------------
inline int decode_zline(const unsigned char* in, int encoded_bytes, int* quantized, int n)
{
    int num = 0;
    int pos = 0;

    while (num < n && pos < encoded_bytes) {
        signed char byte = (signed char)in[pos];

        if (byte > (signed char)VLESC2 && byte < (signed char)RLESC3) {
            quantized[num++] = (int)byte;
            pos += 1;
        } else if (byte == (signed char)RLESC1) {
            int run = (unsigned char)in[pos + 1];
            for (int j = 0; j < run && num < n; ++j)
                quantized[num++] = 0;
            pos += 2;
        } else if (byte == (signed char)RLESC3) {
            int run = (unsigned char)in[pos+1]
                    | ((unsigned char)in[pos+2] << 8)
                    | ((unsigned char)in[pos+3] << 16);
            for (int j = 0; j < run && num < n; ++j)
                quantized[num++] = 0;
            pos += 4;
        } else if (byte == (signed char)VLESC2) {
            short quant;
            memcpy(&quant, in + pos + 1, sizeof(short));
            quantized[num++] = (int)quant;
            pos += 3;
        } else if (byte == (signed char)VLESC3) {
            int quant = (unsigned char)in[pos+1]
                      | ((unsigned char)in[pos+2] << 8)
                      | ((signed char)in[pos+3] << 16);
            quantized[num++] = quant;
            pos += 4;
        } else if (byte == (signed char)VLESC4) {
            float fval;
            memcpy(&fval, in + pos + 1, sizeof(float));
            quantized[num++] = (int)fval;
            pos += 5;
        } else {
            assert(false && "Unknown escape code in decode_zline");
            break;
        }
    }
    return num;
}

// -----------------------------------------------------------------------
// Quantize float coefficients and encode → byte stream.
// scale = 1/(RMS * user_scale).
// Matches Run_Length_Encode_Slow() scalar path.
// VLESC4 stores the scaled float (not the truncated int) for full precision.
// Returns number of encoded bytes.
// -----------------------------------------------------------------------
inline int quantize_encode_zline(const float* coeffs, float scale, int n,
                                 unsigned char* out)
{
    int rle = 0;
    int bytepos = 0;

    for (int i = 0; i < n; ++i) {
        float fval = scale * coeffs[i];
        int ival = (int)fval;
        if (ival == 0) {
            ++rle;
        } else {
            bytepos = rle_flush(rle, out, bytepos);
            rle = 0;
            if (ival > VLESC2 && ival < RLESC3) {
                out[bytepos++] = (unsigned char)(signed char)ival;
            } else if (ival >= -32768 && ival <= 32767) {
                out[bytepos++] = (unsigned char)(signed char)VLESC2;
                out[bytepos++] = ival & 0xFF;
                out[bytepos++] = (ival >> 8) & 0xFF;
            } else if (ival >= -8388608 && ival <= 8388607) {
                out[bytepos++] = (unsigned char)(signed char)VLESC3;
                out[bytepos++] = ival & 0xFF;
                out[bytepos++] = (ival >> 8) & 0xFF;
                out[bytepos++] = (ival >> 16) & 0xFF;
            } else {
                out[bytepos++] = (unsigned char)(signed char)VLESC4;
                memcpy(out + bytepos, &fval, sizeof(float));
                bytepos += 4;
            }
        }
    }
    bytepos = rle_flush(rle, out, bytepos);
    return bytepos;
}

// -----------------------------------------------------------------------
// Decode byte stream → dequantized float coefficients.
// inv_scale = 1/scale = RMS * user_scale.
// Returns number of values decoded.
// -----------------------------------------------------------------------
inline int decode_dequantize_zline(const unsigned char* in, int encoded_bytes,
                                   float inv_scale, float* coeffs, int n)
{
    int num = 0;
    int pos = 0;

    while (num < n && pos < encoded_bytes) {
        signed char byte = (signed char)in[pos];

        if (byte > (signed char)VLESC2 && byte < (signed char)RLESC3) {
            coeffs[num++] = (float)(int)byte * inv_scale;
            pos += 1;
        } else if (byte == (signed char)RLESC1) {
            int run = (unsigned char)in[pos + 1];
            for (int j = 0; j < run && num < n; ++j)
                coeffs[num++] = 0.0f;
            pos += 2;
        } else if (byte == (signed char)RLESC3) {
            int run = (unsigned char)in[pos+1]
                    | ((unsigned char)in[pos+2] << 8)
                    | ((unsigned char)in[pos+3] << 16);
            for (int j = 0; j < run && num < n; ++j)
                coeffs[num++] = 0.0f;
            pos += 4;
        } else if (byte == (signed char)VLESC2) {
            short quant;
            memcpy(&quant, in + pos + 1, sizeof(short));
            coeffs[num++] = (float)quant * inv_scale;
            pos += 3;
        } else if (byte == (signed char)VLESC3) {
            int quant = (unsigned char)in[pos+1]
                      | ((unsigned char)in[pos+2] << 8)
                      | ((signed char)in[pos+3] << 16);
            coeffs[num++] = (float)quant * inv_scale;
            pos += 4;
        } else if (byte == (signed char)VLESC4) {
            float fval;
            memcpy(&fval, in + pos + 1, sizeof(float));
            coeffs[num++] = fval * inv_scale;
            pos += 5;
        } else {
            assert(false && "Unknown escape code in decode_dequantize_zline");
            break;
        }
    }
    return num;
}

#endif // QUANTIZE_RLE_REF_H
