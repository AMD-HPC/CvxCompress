// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#ifndef DS79_H
#define DS79_H

#ifdef __HIPCC__
#define DS79_HD __host__ __device__
#else
#define DS79_HD
#endif

#define DS79_AL0  8.526986790094000e-001f
#define DS79_AL1  3.774028556126500e-001f
#define DS79_AL2 -1.106244044184200e-001f
#define DS79_AL3 -2.384946501938001e-002f
#define DS79_AL4  3.782845550699501e-002f

#define DS79_AH0  7.884856164056601e-001f
#define DS79_AH1 -4.180922732222101e-001f
#define DS79_AH2 -4.068941760955800e-002f
#define DS79_AH3  6.453888262893799e-002f

DS79_HD
inline int ds79_mirr(int val, int dim) {
    val = val < 0 ? -val : val;
    val = (val >= dim) ? (2 * dim - 2 - val) : val;
    val = val < 0 ? -val : val;
    val = (val >= dim) ? (2 * dim - 2 - val) : val;
    return val;
}

// Antonini 7-9 synthesis (inverse) filter coefficients.
#define US79_SL0  7.884856164056601e-001f
#define US79_SL1  4.180922732222101e-001f
#define US79_SL2 -4.068941760955800e-002f
#define US79_SL3 -6.453888262893799e-002f

#define US79_SH0  8.526986790094000e-001f
#define US79_SH1 -3.774028556126500e-001f
#define US79_SH2 -1.106244044184200e-001f
#define US79_SH3  2.384946501938001e-002f
#define US79_SH4  3.782845550699501e-002f

// Mirror index within lowpass band [0, nl) for inverse transform.
// Half-sample symmetric boundary: reflects around nl - 0.5.
DS79_HD
inline int us79_mirr_sl(int val, int nl) {
    val = val < 0 ? -val : val;
    val = (val >= nl) ? (2 * nl - 1 - val) : val;
    val = val < 0 ? -val : val;
    val = (val >= nl) ? (2 * nl - 1 - val) : val;
    val = val < 0 ? -val : val;
    val = (val >= nl) ? (2 * nl - 1 - val) : val;
    return val;
}

// Mirror index within highpass band [nl, nl+nh) for inverse transform.
// Whole-sample antisymmetric at left, whole-sample symmetric at right.
DS79_HD
inline int us79_mirr_sh(int inp_val, int nl, int nh) {
    int val = inp_val - nl;
    val = val < 0 ? -val - 1 : val;
    val = (val >= nh) ? (2 * nh - 2 - val) : val;
    val = val < 0 ? -val - 1 : val;
    val = (val >= nh) ? (2 * nh - 2 - val) : val;
    val = val < 0 ? -val - 1 : val;
    val = (val >= nh) ? (2 * nh - 2 - val) : val;
    return nl + val;
}

// Antonini 7-9 tap forward wavelet transform.
// Operates in-place on a contiguous array of `dim` floats.
// Recursive decomposition: each level splits n into nl low + nh high coefficients.
DS79_HD
inline void ds79_forward(float* data, int dim) {
    float tmp[32];
    for (int n = dim; n >= 2; n = n - n / 2) {
        for (int i = 0; i < n; ++i) tmp[i] = data[i];
        int nh = n / 2;
        int nl = n - nh;
        for (int ix = 0; ix < nl; ++ix) {
            int i0 = 2 * ix;
            int im1 = ds79_mirr(i0 - 1, n), ip1 = ds79_mirr(i0 + 1, n);
            int im2 = ds79_mirr(i0 - 2, n), ip2 = ds79_mirr(i0 + 2, n);
            int im3 = ds79_mirr(i0 - 3, n), ip3 = ds79_mirr(i0 + 3, n);
            int im4 = ds79_mirr(i0 - 4, n), ip4 = ds79_mirr(i0 + 4, n);
            float acc = DS79_AL0 * tmp[i0];
            acc += DS79_AL1 * (tmp[im1] + tmp[ip1]);
            acc += DS79_AL2 * (tmp[im2] + tmp[ip2]);
            acc += DS79_AL3 * (tmp[im3] + tmp[ip3]);
            acc += DS79_AL4 * (tmp[im4] + tmp[ip4]);
            data[ix] = acc;
        }
        for (int ix = 0; ix < nh; ++ix) {
            int i0 = 2 * ix + 1;
            int im1 = ds79_mirr(i0 - 1, n), ip1 = ds79_mirr(i0 + 1, n);
            int im2 = ds79_mirr(i0 - 2, n), ip2 = ds79_mirr(i0 + 2, n);
            int im3 = ds79_mirr(i0 - 3, n), ip3 = ds79_mirr(i0 + 3, n);
            float acc = DS79_AH0 * tmp[i0];
            acc += DS79_AH1 * (tmp[im1] + tmp[ip1]);
            acc += DS79_AH2 * (tmp[im2] + tmp[ip2]);
            acc += DS79_AH3 * (tmp[im3] + tmp[ip3]);
            data[nl + ix] = acc;
        }
    }
}

// Antonini 7-9 tap inverse wavelet transform.
// Reverses ds79_forward. Levels are processed smallest-to-largest.
DS79_HD
inline void us79_inverse(float* data, int dim) {
    float tmp[32];
    int levels[16];
    int nlev = 0;
    for (int n = dim; n >= 2; n = n - n / 2) levels[nlev++] = n;
    for (int li = nlev - 1; li >= 0; --li) {
        int n = levels[li];
        for (int i = 0; i < n; ++i) tmp[i] = data[i];
        int nh = n / 2;
        int nl = n - nh;
        for (int k = 0; k < nl; ++k) {
            int sk = k;
            int skm1 = us79_mirr_sl(k - 1, nl), skp1 = us79_mirr_sl(k + 1, nl);
            int hkm1 = us79_mirr_sh(nl + k - 1, nl, nh), hk = us79_mirr_sh(nl + k, nl, nh);
            int hkm2 = us79_mirr_sh(nl + k - 2, nl, nh), hkp1 = us79_mirr_sh(nl + k + 1, nl, nh);
            data[2 * k] = US79_SL0 * tmp[sk]
                        + US79_SL2 * (tmp[skm1] + tmp[skp1])
                        + US79_SH1 * (tmp[hkm1] + tmp[hk])
                        + US79_SH3 * (tmp[hkm2] + tmp[hkp1]);
        }
        for (int k = 0; k < nh; ++k) {
            int sk = us79_mirr_sl(k, nl), skp1 = us79_mirr_sl(k + 1, nl);
            int skm1 = us79_mirr_sl(k - 1, nl), skp2 = us79_mirr_sl(k + 2, nl);
            int hk = nl + k;
            int hkm1 = us79_mirr_sh(nl + k - 1, nl, nh), hkp1 = us79_mirr_sh(nl + k + 1, nl, nh);
            int hkm2 = us79_mirr_sh(nl + k - 2, nl, nh), hkp2 = us79_mirr_sh(nl + k + 2, nl, nh);
            data[2 * k + 1] = US79_SL1 * (tmp[sk] + tmp[skp1])
                             + US79_SL3 * (tmp[skm1] + tmp[skp2])
                             + US79_SH0 * tmp[hk]
                             + US79_SH2 * (tmp[hkm1] + tmp[hkp1])
                             + US79_SH4 * (tmp[hkm2] + tmp[hkp2]);
        }
    }
}

#ifdef __HIPCC__
#include "ds79_reg32.inc"
#include "us79_reg32.inc"

using ds79_float4_vec = __attribute__((__vector_size__(4 * sizeof(float)))) float;

__device__ inline void ds79_forward_f4_scalar_tmp(ds79_float4_vec* data, int dim) {
    float tmp[32];
    for (int c = 0; c < 4; c++) {
        for (int n = dim; n >= 2; n = n - n / 2) {
            for (int i = 0; i < n; ++i) tmp[i] = data[i][c];
            int nh = n / 2, nl = n - nh;
            for (int ix = 0; ix < nl; ++ix) {
                int i0 = 2 * ix;
                int im1 = ds79_mirr(i0-1,n), ip1 = ds79_mirr(i0+1,n);
                int im2 = ds79_mirr(i0-2,n), ip2 = ds79_mirr(i0+2,n);
                int im3 = ds79_mirr(i0-3,n), ip3 = ds79_mirr(i0+3,n);
                int im4 = ds79_mirr(i0-4,n), ip4 = ds79_mirr(i0+4,n);
                float acc = DS79_AL0*tmp[i0];
                acc += DS79_AL1*(tmp[im1]+tmp[ip1]);
                acc += DS79_AL2*(tmp[im2]+tmp[ip2]);
                acc += DS79_AL3*(tmp[im3]+tmp[ip3]);
                acc += DS79_AL4*(tmp[im4]+tmp[ip4]);
                data[ix][c] = acc;
            }
            for (int ix = 0; ix < nh; ++ix) {
                int i0 = 2 * ix + 1;
                int im1 = ds79_mirr(i0-1,n), ip1 = ds79_mirr(i0+1,n);
                int im2 = ds79_mirr(i0-2,n), ip2 = ds79_mirr(i0+2,n);
                int im3 = ds79_mirr(i0-3,n), ip3 = ds79_mirr(i0+3,n);
                float acc = DS79_AH0*tmp[i0];
                acc += DS79_AH1*(tmp[im1]+tmp[ip1]);
                acc += DS79_AH2*(tmp[im2]+tmp[ip2]);
                acc += DS79_AH3*(tmp[im3]+tmp[ip3]);
                data[nl+ix][c] = acc;
            }
        }
    }
}

__device__ inline void us79_inverse_f4_scalar_tmp(ds79_float4_vec* data, int dim) {
    float tmp[32];
    int levels[16];
    int nlev = 0;
    for (int n = dim; n >= 2; n = n - n / 2) levels[nlev++] = n;
    for (int c = 0; c < 4; c++) {
        for (int li = nlev - 1; li >= 0; --li) {
            int n = levels[li];
            for (int i = 0; i < n; ++i) tmp[i] = data[i][c];
            int nh = n / 2, nl = n - nh;
            for (int k = 0; k < nl; ++k) {
                int sk = k;
                int skm1 = us79_mirr_sl(k-1,nl), skp1 = us79_mirr_sl(k+1,nl);
                int hkm1 = us79_mirr_sh(nl+k-1,nl,nh), hk = us79_mirr_sh(nl+k,nl,nh);
                int hkm2 = us79_mirr_sh(nl+k-2,nl,nh), hkp1 = us79_mirr_sh(nl+k+1,nl,nh);
                data[2*k][c] = US79_SL0*tmp[sk]
                    + US79_SL2*(tmp[skm1]+tmp[skp1])
                    + US79_SH1*(tmp[hkm1]+tmp[hk])
                    + US79_SH3*(tmp[hkm2]+tmp[hkp1]);
            }
            for (int k = 0; k < nh; ++k) {
                int sk = us79_mirr_sl(k,nl), skp1 = us79_mirr_sl(k+1,nl);
                int skm1 = us79_mirr_sl(k-1,nl), skp2 = us79_mirr_sl(k+2,nl);
                int hk = nl + k;
                int hkm1 = us79_mirr_sh(nl+k-1,nl,nh), hkp1 = us79_mirr_sh(nl+k+1,nl,nh);
                int hkm2 = us79_mirr_sh(nl+k-2,nl,nh), hkp2 = us79_mirr_sh(nl+k+2,nl,nh);
                data[2*k+1][c] = US79_SL1*(tmp[sk]+tmp[skp1])
                    + US79_SL3*(tmp[skm1]+tmp[skp2])
                    + US79_SH0*tmp[hk]
                    + US79_SH2*(tmp[hkm1]+tmp[hkp1])
                    + US79_SH4*(tmp[hkm2]+tmp[hkp2]);
            }
        }
    }
}
#endif

#endif
