// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// Unit tests for scalar z-line quantize + RLE encode/decode.
// CPU-only, no HIP dependency. Compile with: g++ -O2 -std=c++11 -I. ...

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include "quantize_rle_ref.h"

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, fmt, ...) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; } \
    else { printf("  FAIL: " fmt "\n", ##__VA_ARGS__); } \
} while(0)

// -----------------------------------------------------------------------
// Test 1: all zeros
// -----------------------------------------------------------------------
static void test_all_zeros()
{
    printf("Test 1: all zeros (n=32) ...");
    int start_passed = tests_passed;
    int start_run = tests_run;
    int input[32] = {};
    unsigned char buf[256];
    int decoded[32];

    int enc_bytes = encode_zline(input, 32, buf);
    CHECK(enc_bytes == 2, "expected 2 bytes (RLESC1+32), got %d", enc_bytes);
    CHECK(buf[0] == (unsigned char)(RLESC1 & 0xFF), "byte 0 should be RLESC1");
    CHECK(buf[1] == 32, "byte 1 should be 32, got %d", (int)buf[1]);

    int dec_n = decode_zline(buf, enc_bytes, decoded, 32);
    CHECK(dec_n == 32, "decoded count should be 32, got %d", dec_n);
    bool match = true;
    for (int i = 0; i < 32; ++i)
        if (decoded[i] != 0) { match = false; break; }
    CHECK(match, "decoded values should all be 0");

    int local_passed = (tests_passed - start_passed) == (tests_run - start_run);
    printf(" %s\n", local_passed ? "PASSED" : "FAILED");
}

// -----------------------------------------------------------------------
// Test 2: all byte-range nonzeros
// -----------------------------------------------------------------------
static void test_all_byte_literals()
{
    printf("Test 2: all byte literals (n=32) ...");
    int start_passed = tests_passed;
    int start_run = tests_run;
    int input[32];
    for (int i = 0; i < 32; ++i) input[i] = (i % 2 == 0) ? (i/2 + 1) : -(i/2 + 1);
    unsigned char buf[256];
    int decoded[32];

    int enc_bytes = encode_zline(input, 32, buf);
    CHECK(enc_bytes == 32, "expected 32 bytes, got %d", enc_bytes);

    int dec_n = decode_zline(buf, enc_bytes, decoded, 32);
    CHECK(dec_n == 32, "decoded count should be 32, got %d", dec_n);
    bool match = true;
    for (int i = 0; i < 32; ++i)
        if (decoded[i] != input[i]) { match = false; break; }
    CHECK(match, "round-trip mismatch");

    int local_passed = (tests_passed - start_passed) == (tests_run - start_run);
    printf(" %s\n", local_passed ? "PASSED" : "FAILED");
}

// -----------------------------------------------------------------------
// Test 3: single nonzero at each position
// -----------------------------------------------------------------------
static void test_single_nonzero()
{
    printf("Test 3: single nonzero at each position ...");
    int start_passed = tests_passed;
    int start_run = tests_run;

    for (int pos = 0; pos < 32; ++pos) {
        int input[32] = {};
        input[pos] = 42;
        unsigned char buf[256];
        int decoded[32];

        int enc_bytes = encode_zline(input, 32, buf);
        int dec_n = decode_zline(buf, enc_bytes, decoded, 32);
        CHECK(dec_n == 32, "pos=%d: decoded %d values", pos, dec_n);
        bool match = true;
        for (int i = 0; i < 32; ++i)
            if (decoded[i] != input[i]) { match = false; break; }
        CHECK(match, "pos=%d: round-trip mismatch", pos);
    }
    int local_passed = (tests_passed - start_passed) == (tests_run - start_run);
    printf(" %s\n", local_passed ? "PASSED" : "FAILED");
}

// -----------------------------------------------------------------------
// Test 4: alternating zero / nonzero
// -----------------------------------------------------------------------
static void test_alternating()
{
    printf("Test 4: alternating 0, value ...");
    int start_passed = tests_passed;
    int start_run = tests_run;
    int input[32];
    for (int i = 0; i < 32; ++i) input[i] = (i % 2 == 0) ? 0 : (i/2 + 1);
    unsigned char buf[256];
    int decoded[32];

    int enc_bytes = encode_zline(input, 32, buf);
    int dec_n = decode_zline(buf, enc_bytes, decoded, 32);
    CHECK(dec_n == 32, "decoded count should be 32, got %d", dec_n);
    bool match = true;
    for (int i = 0; i < 32; ++i)
        if (decoded[i] != input[i]) { match = false; break; }
    CHECK(match, "round-trip mismatch");

    int local_passed = (tests_passed - start_passed) == (tests_run - start_run);
    printf(" %s\n", local_passed ? "PASSED" : "FAILED");
}

// -----------------------------------------------------------------------
// Test 5: magnitude boundaries
// -----------------------------------------------------------------------
static void test_magnitude_boundaries()
{
    printf("Test 5: magnitude boundaries ...");
    int start_passed = tests_passed;
    int start_run = tests_run;

    int test_values[] = {
        // byte range boundaries
        -124, 124, -1, 1,
        // just outside byte range → VLESC2
        -125, 125, -200, 200, -32768, 32767,
        // just outside int16 → VLESC3
        -32769, 32768, -100000, 100000, -8388608, 8388607,
        // just outside int24 → VLESC4
        -8388609, 8388608, -10000000, 10000000
    };
    int n_tests = sizeof(test_values) / sizeof(test_values[0]);

    for (int t = 0; t < n_tests; ++t) {
        int input[4] = {test_values[t], 0, 0, test_values[t]};
        unsigned char buf[64];
        int decoded[4];

        int enc_bytes = encode_zline(input, 4, buf);
        CHECK(enc_bytes > 0, "val=%d: encode returned 0 bytes", test_values[t]);
        int dec_n = decode_zline(buf, enc_bytes, decoded, 4);
        CHECK(dec_n == 4, "val=%d: decoded %d values", test_values[t], dec_n);

        bool match = (decoded[0] == input[0] && decoded[1] == 0 &&
                      decoded[2] == 0 && decoded[3] == input[3]);
        if (!match && test_values[t] > 8388607 || test_values[t] < -8388608) {
            // VLESC4: float round-trip may lose precision for large ints
            float fval = (float)test_values[t];
            int expected = (int)fval;
            match = (decoded[0] == expected && decoded[3] == expected);
        }
        CHECK(match, "val=%d: round-trip mismatch (got %d, %d)",
              test_values[t], decoded[0], decoded[3]);
    }
    int local_passed = (tests_passed - start_passed) == (tests_run - start_run);
    printf(" %s\n", local_passed ? "PASSED" : "FAILED");
}

// -----------------------------------------------------------------------
// Test 6: large zero run (> 255, triggers RLESC3)
// -----------------------------------------------------------------------
static void test_large_zero_run()
{
    printf("Test 6: large zero run (n=1024) ...");
    int start_passed = tests_passed;
    int start_run = tests_run;
    int n = 1024;
    int* input = (int*)calloc(n, sizeof(int));
    input[0] = 5;
    input[n-1] = -3;
    unsigned char* buf = (unsigned char*)malloc(n * 5);
    int* decoded = (int*)malloc(n * sizeof(int));

    int enc_bytes = encode_zline(input, n, buf);
    CHECK(enc_bytes > 0, "encode returned 0 bytes");

    int dec_n = decode_zline(buf, enc_bytes, decoded, n);
    CHECK(dec_n == n, "decoded %d values, expected %d", dec_n, n);
    bool match = true;
    for (int i = 0; i < n; ++i)
        if (decoded[i] != input[i]) { match = false; break; }
    CHECK(match, "round-trip mismatch");

    free(input); free(buf); free(decoded);
    int local_passed = (tests_passed - start_passed) == (tests_run - start_run);
    printf(" %s\n", local_passed ? "PASSED" : "FAILED");
}

// -----------------------------------------------------------------------
// Test 7: quantize + encode → decode + dequantize round-trip
// -----------------------------------------------------------------------
static void test_quantize_round_trip()
{
    printf("Test 7: quantize round-trip (n=32) ...");
    float coeffs[32];
    srand(12345);
    for (int i = 0; i < 32; ++i)
        coeffs[i] = ((float)rand() / RAND_MAX - 0.5f) * 200.0f;

    float scale = 10.0f;
    float inv_scale = 1.0f / scale;
    unsigned char buf[256];
    float decoded[32];

    int enc_bytes = quantize_encode_zline(coeffs, scale, 32, buf);
    CHECK(enc_bytes > 0, "encode returned 0 bytes");

    int dec_n = decode_dequantize_zline(buf, enc_bytes, inv_scale, decoded, 32);
    CHECK(dec_n == 32, "decoded %d values, expected 32", dec_n);

    float max_err = 0.0f;
    for (int i = 0; i < 32; ++i) {
        float err = fabsf(coeffs[i] - decoded[i]);
        if (err > max_err) max_err = err;
    }
    // Truncation toward zero: |fval - (int)fval| < 1.0, so error < 1.0/scale
    float bound = 1.0f / scale + 1e-6f;
    CHECK(max_err <= bound, "max error %.6e exceeds bound %.6e", max_err, bound);

    printf(" max_err=%.3e, bound=%.3e\n", max_err, bound);
}

// -----------------------------------------------------------------------
// Test 8: wavelet-like coefficient distribution (mostly zeros after quant)
// -----------------------------------------------------------------------
static void test_wavelet_like_distribution()
{
    printf("Test 8: wavelet-like distribution (n=32) ...");

    // Simulate: DC coeff large, detail coeffs decay exponentially
    float coeffs[32];
    coeffs[0] = 1000.0f;
    coeffs[1] = 50.0f;
    for (int i = 2; i < 32; ++i)
        coeffs[i] = 0.1f / (float)(i * i);

    float scale = 0.5f;
    float inv_scale = 1.0f / scale;
    unsigned char buf[256];
    float decoded[32];

    int enc_bytes = quantize_encode_zline(coeffs, scale, 32, buf);
    float raw_bytes = 32 * sizeof(float);
    float ratio = raw_bytes / (float)enc_bytes;

    int dec_n = decode_dequantize_zline(buf, enc_bytes, inv_scale, decoded, 32);
    CHECK(dec_n == 32, "decoded %d values, expected 32", dec_n);

    float max_err = 0.0f;
    for (int i = 0; i < 32; ++i) {
        float err = fabsf(coeffs[i] - decoded[i]);
        if (err > max_err) max_err = err;
    }
    float bound = 1.0f / scale + 1e-6f;
    CHECK(max_err <= bound, "max error %.6e exceeds bound %.6e", max_err, bound);

    printf(" enc=%d bytes, ratio=%.1f:1, max_err=%.3e\n", enc_bytes, ratio, max_err);
}

// -----------------------------------------------------------------------
// Test 9: edge case — single value
// -----------------------------------------------------------------------
static void test_single_value()
{
    printf("Test 9: single value ...");
    int start_passed = tests_passed;
    int start_run = tests_run;

    int cases[] = {0, 1, -1, 124, -124, 200, -200, 32767, -32768, 100000};
    int n_cases = sizeof(cases) / sizeof(cases[0]);

    for (int c = 0; c < n_cases; ++c) {
        int input[1] = {cases[c]};
        unsigned char buf[16];
        int decoded[1] = {-999};

        int enc_bytes = encode_zline(input, 1, buf);
        int dec_n = decode_zline(buf, enc_bytes, decoded, 1);
        CHECK(dec_n == 1, "val=%d: decoded %d values", cases[c], dec_n);

        bool match = (decoded[0] == cases[c]);
        if (!match && (cases[c] > 8388607 || cases[c] < -8388608)) {
            match = (decoded[0] == (int)(float)cases[c]);
        }
        CHECK(match, "val=%d: got %d", cases[c], decoded[0]);
    }
    int local_passed = (tests_passed - start_passed) == (tests_run - start_run);
    printf(" %s\n", local_passed ? "PASSED" : "FAILED");
}

// -----------------------------------------------------------------------
// Test 10: byte-level encoding format verification
// -----------------------------------------------------------------------
static void test_encoding_format()
{
    printf("Test 10: encoding format verification ...");
    int start_passed = tests_passed;
    int start_run = tests_run;

    // (a) Single zero → byte literal 0
    {
        int input[1] = {0};
        unsigned char buf[8];
        int enc_bytes = encode_zline(input, 1, buf);
        CHECK(enc_bytes == 1, "single zero: expected 1 byte, got %d", enc_bytes);
        CHECK(buf[0] == 0, "single zero: byte should be 0, got %d", (int)buf[0]);
    }

    // (b) Two zeros → RLESC1 + count=2
    {
        int input[2] = {0, 0};
        unsigned char buf[8];
        int enc_bytes = encode_zline(input, 2, buf);
        CHECK(enc_bytes == 2, "two zeros: expected 2 bytes, got %d", enc_bytes);
        CHECK(buf[0] == (unsigned char)(RLESC1 & 0xFF), "two zeros: byte 0 should be RLESC1");
        CHECK(buf[1] == 2, "two zeros: count should be 2, got %d", (int)buf[1]);
    }

    // (c) Value 50 → single byte literal
    {
        int input[1] = {50};
        unsigned char buf[8];
        int enc_bytes = encode_zline(input, 1, buf);
        CHECK(enc_bytes == 1, "literal 50: expected 1 byte, got %d", enc_bytes);
        CHECK((signed char)buf[0] == 50, "literal 50: got %d", (int)(signed char)buf[0]);
    }

    // (d) Value 200 → VLESC2 + 2 bytes
    {
        int input[1] = {200};
        unsigned char buf[8];
        int enc_bytes = encode_zline(input, 1, buf);
        CHECK(enc_bytes == 3, "VLESC2(200): expected 3 bytes, got %d", enc_bytes);
        CHECK((signed char)buf[0] == (signed char)VLESC2, "byte 0 should be VLESC2");
        short decoded_val;
        memcpy(&decoded_val, buf + 1, sizeof(short));
        CHECK(decoded_val == 200, "payload should be 200, got %d", (int)decoded_val);
    }

    // (e) Value 100000 → VLESC3 + 3 bytes
    {
        int input[1] = {100000};
        unsigned char buf[8];
        int enc_bytes = encode_zline(input, 1, buf);
        CHECK(enc_bytes == 4, "VLESC3(100000): expected 4 bytes, got %d", enc_bytes);
        CHECK((signed char)buf[0] == (signed char)VLESC3, "byte 0 should be VLESC3");
    }

    // (f) Value 10000000 → VLESC4 + 4 bytes
    {
        int input[1] = {10000000};
        unsigned char buf[8];
        int enc_bytes = encode_zline(input, 1, buf);
        CHECK(enc_bytes == 5, "VLESC4(10M): expected 5 bytes, got %d", enc_bytes);
        CHECK((signed char)buf[0] == (signed char)VLESC4, "byte 0 should be VLESC4");
    }

    int local_passed = (tests_passed - start_passed) == (tests_run - start_run);
    printf(" %s\n", local_passed ? "PASSED" : "FAILED");
}

// -----------------------------------------------------------------------
int main()
{
    printf("\n=== Quantize + RLE z-line unit tests ===\n\n");

    int initial_run = tests_run;
    int initial_passed = tests_passed;

    test_all_zeros();
    test_all_byte_literals();
    test_single_nonzero();
    test_alternating();
    test_magnitude_boundaries();
    test_large_zero_run();
    test_quantize_round_trip();
    test_wavelet_like_distribution();
    test_single_value();
    test_encoding_format();

    printf("\n--- Summary: %d/%d checks passed ---\n",
           tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
