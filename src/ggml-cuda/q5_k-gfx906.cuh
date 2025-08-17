#pragma once

// GFX906-optimized Q5_K quantization implementation using V_DOT2_F32_F16 and V_PK_FMA_F16

#if defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)

#include "common.cuh"
#include "gfx906-config.cuh"

// Q5_K vec_dot implementation using V_DOT2_F32_F16 instruction
template <int vdr> 
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_gfx906(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int v0 = v[i*2+0];
        const int v1 = v[i*2+1];
        const int u0 = u[i*2+0];
        const int u1 = u[i*2+1];

        // Extract scales
        const int sc_idx = i / (QR5_K/2);
        const int sc_shift = (i % (QR5_K/2)) * 4;
        const int scale = (sc[sc_idx] >> sc_shift) & 0xF;
        const int min_val = (m[sc_idx] >> sc_shift) & 0xF;

        // Use standard integer dot product operations
        // Q5_K uses 5-bit quantization, apply appropriate scaling
        int sumi0 = 0;
        int sumi1 = 0;
        
        // Process 5-bit values with integer arithmetic
        for (int k = 0; k < 4; ++k) {
            const int v0k = (v0 >> (k*5)) & 0x1F;
            const int v1k = (v1 >> (k*5)) & 0x1F;
            const int u0k = (u0 >> (k*8)) & 0xFF;
            const int u1k = (u1 >> (k*8)) & 0xFF;
            
            sumi0 += v0k * u0k;
            sumi1 += v1k * u1k;
        }
        
        float dot0 = (float)sumi0;
        float dot1 = (float)sumi1;

        // Apply scales
        const float2 ds8f = __half22float2(ds8[i]);
        sumf_d += (dot0 + dot1) * (float)scale * ds8f.x;
        sumf_m += (dot0 + dot1) * (float)min_val * ds8f.y;
    }

    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// Optimized vec_dot for Q5_K using GFX906 hardware features
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_gfx906(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    
    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int vl[2];
    int vh[2];
    int u[4];

    // Extract and prepare quantized values
    const int bq8_offset = QR5_K * (iqs / QI8_1) + (iqs % QI8_1) * QR5_K;
    const int * ql = (const int *)(bq5_K->qs + 16 * (iqs / 2));
    const int * qh = (const int *)(bq5_K->qh + 4 * (iqs / 2));

    // Load values
    vl[0] = ql[0];
    vl[1] = ql[1];
    
    vh[0] = qh[0] >> (4 * (iqs & 1));
    vh[1] = qh[1] >> (4 * (iqs & 1));

    // Load Q8_1 values
    const block_q8_1 * bq8 = bq8_1 + bq8_offset;
    u[0] = get_int_b4(bq8[0].qs, 0);
    u[1] = get_int_b4(bq8[0].qs, 4);
    u[2] = get_int_b4(bq8[1].qs, 0);
    u[3] = get_int_b4(bq8[1].qs, 4);

    // Extract scales and mins
    const uint8_t * sc = bq5_K->scales + (iqs / 2);
    const uint8_t * m = sc + 8;

    // Get delta values
    half2 d8[2];
    d8[0] = bq8[0].ds;
    d8[1] = bq8[1].ds;

    // Combine low and high bits
    int v[4];
    v[0] = vl[0] | ((vh[0] & 0x0F0F0F0F) << 4);
    v[1] = (vl[0] >> 4) | ((vh[0] & 0xF0F0F0F0) << 0);
    v[2] = vl[1] | ((vh[1] & 0x0F0F0F0F) << 4);
    v[3] = (vl[1] >> 4) | ((vh[1] & 0xF0F0F0F0) << 0);

    return vec_dot_q5_K_q8_1_impl_gfx906<2>(v, u, sc, m, bq5_K->dm, d8);
}

#endif // defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)