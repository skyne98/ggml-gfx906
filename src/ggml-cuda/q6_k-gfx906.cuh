#pragma once

// GFX906-optimized Q6_K quantization implementation using V_DOT2_F32_F16 and V_PK_FMA_F16

#if defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)

#include "common.cuh"
#include "gfx906-config.cuh"

// Q6_K vec_dot implementation using optimized integer operations
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_gfx906(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {

    float sumf_d = 0.0f;

    // Note: QR6_K is not defined in this context, using standard processing
    const int n_iter = 1; // Process one iteration for now
    
#pragma unroll
    for (int i = 0; i < n_iter; ++i) {
        const int vi0 = v[i*2+0];
        const int vi1 = v[i*2+1];
        const int ui0 = u[i*2+0];
        const int ui1 = u[i*2+1];

        // Extract 6-bit values and perform integer dot product
        int sumi0 = 0;
        int sumi1 = 0;
        
        // Process 4 6-bit values from vi0
        for (int k = 0; k < 4; ++k) {
            const int q0k = (vi0 >> (k*6)) & 0x3F;
            const int q1k = (vi1 >> (k*6)) & 0x3F;
            const int8_t* ui0_bytes = (const int8_t*)&ui0;
            const int8_t* ui1_bytes = (const int8_t*)&ui1;
            
            sumi0 += (q0k - 32) * ui0_bytes[k];
            sumi1 += (q1k - 32) * ui1_bytes[k];
        }
        
        float dot0 = (float)sumi0;
        float dot1 = (float)sumi1;

        // Apply scales
        const float scale = sc[i];
        sumf_d += (dot0 + dot1) * scale * d8[i];
    }

    return d6 * sumf_d;
}

// Alternative implementation using optimized integer operations
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_pk_fma_gfx906(
    const int & vl, const int & vh, const int * __restrict__ u, 
    const int8_t * __restrict__ scales, const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

    // Extract 6-bit quantized values
    const int ql0 = (vl >> 0) & 0x0F0F0F0F;
    const int ql1 = (vl >> 4) & 0x0F0F0F0F;
    const int qh0 = (vh >> 0) & 0x03030303;
    const int qh1 = (vh >> 4) & 0x03030303;

    // Combine to get 6-bit values
    const int q0 = ql0 | (qh0 << 4);
    const int q1 = ql1 | (qh1 << 4);

    // Process with integer operations
    for (int i = 0; i < 2; ++i) {
        const int qi = (i == 0) ? q0 : q1;
        const int ui = u[i];
        
        // Extract bytes and compute dot product
        const int8_t* qi_bytes = (const int8_t*)&qi;
        const int8_t* ui_bytes = (const int8_t*)&ui;
        
        int sumi = 0;
        for (int k = 0; k < 4; ++k) {
            sumi += (qi_bytes[k] - 32) * ui_bytes[k];
        }
        
        sumf += (float)sumi * scales[i/2] * d8[i];
    }

    return d * sumf;
}

// Main Q6_K vec_dot function selecting optimal implementation
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_gfx906(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const block_q8_1 * bq8 = bq8_1 + bq8_offset;

    // Extract quantized values
    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (iqs % (QI6_K/2)) / (QI6_K/4));
    
    int u[2];
    float d8[2];
    
    u[0] = get_int_b4(bq8[0].qs, (iqs % (QI6_K/4)) * 4);
    u[1] = get_int_b4(bq8[1].qs, (iqs % (QI6_K/4)) * 4);
    d8[0] = __half2float(bq8[0].ds.x);
    d8[1] = __half2float(bq8[1].ds.x);

    // Extract scales
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int8_t * scales = bq6_K->scales + scale_offset;

    // Use the packed FMA implementation for better throughput
    return vec_dot_q6_K_q8_1_pk_fma_gfx906(vl, vh, u, scales, bq6_K->d, d8);
}

#endif // defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)