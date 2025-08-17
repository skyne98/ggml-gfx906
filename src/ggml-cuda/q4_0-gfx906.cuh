#pragma once

// Optimized Q4_0 quantization kernels for GFX906 using V_DOT8_I32_I4
// Implements fused dequantize-and-GEMV operations with LDS optimization

#include "common.cuh"
#include "gfx906-config.cuh"

#if defined(GGML_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#ifdef __HIP_DEVICE_COMPILE__

// V_DOT8_I32_I4 intrinsic for GFX906
// Performs 8-way dot product on signed 4-bit integers
__device__ __forceinline__ int32_t gfx906_dot8_i4(uint32_t a, uint32_t b) {
    int32_t result;
    // Using the V_DOT8_I32_I4 instruction for signed 4-bit dot products
    // This instruction unpacks 8x 4-bit values from each 32-bit input
    // and performs: result = sum(a[i] * b[i]) for i=0..7
    asm volatile("v_dot8_i32_i4 %0, %1, %2, 0" : "=v"(result) : "v"(a), "v"(b));
    return result;
}

// Optimized Q4_0 to Q8_1 dot product using V_DOT8_I32_I4
template <int vdr>
__device__ __forceinline__ float vec_dot_q4_0_q8_1_gfx906(
    const void* __restrict__ vbq, 
    const block_q8_1* __restrict__ bq8_1, 
    const int& kbx, 
    const int& iqs) {
    
    const block_q4_0* bq4_0 = (const block_q4_0*) vbq;
    
    int32_t sumi = 0;
    
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int block_idx = kbx + i;
        
        // Load Q4_0 data: 32 4-bit values packed into 16 bytes
        // We process 8 values at a time using V_DOT8_I32_I4
        const uint32_t* q4_ptr = reinterpret_cast<const uint32_t*>(bq4_0[block_idx].qs);
        const int8_t* q8_ptr = bq8_1[block_idx].qs + iqs;
        
        // Process 32 values as 4 groups of 8
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            // Load 8 packed 4-bit values (one 32-bit word)
            uint32_t q4_vals = q4_ptr[j];
            
            // Load corresponding 8 int8 values and pack into 32-bit word
            uint32_t q8_vals = 0;
            for (int k = 0; k < 4; ++k) {
                q8_vals |= (static_cast<uint32_t>(q8_ptr[j*8 + k*2]) & 0xFF) << (k*8);
                q8_vals |= (static_cast<uint32_t>(q8_ptr[j*8 + k*2 + 1]) & 0xFF) << (k*8 + 4);
            }
            
            // Use V_DOT8_I32_I4 instruction for 8-way dot product
            sumi += gfx906_dot8_i4(q4_vals, q8_vals);
        }
    }
    
    // Apply scaling factors
    const float d4 = __half2float(bq4_0[kbx].d);
    const float2 ds8 = __half22float2(bq8_1[kbx].ds);
    
    // Account for Q4_0 offset (-8) in the scaling
    return d4 * (sumi * ds8.x - (8 * 32 * vdr) * ds8.y);
}

// Fused dequantize-and-GEMV kernel optimized for GFX906
template <int BLOCK_SIZE, int ROWS_PER_BLOCK>
__global__ void mul_mat_vec_q4_0_q8_1_gfx906(
    const void* __restrict__ vx,
    const void* __restrict__ vy, 
    float* __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int nrows_dst) {
    
    // Configure for 256 threads (4 wavefronts) per block
    static_assert(BLOCK_SIZE == 256, "Block size must be 256 for GFX906 optimization");
    
    // Shared memory for caching input vector and scale factors
    __shared__ float s_vec[256];  // Cache portion of input vector
    __shared__ half s_scales[16]; // Cache scale factors
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / 64;  // Wave ID within block
    const int lane_id = tid % 64;  // Lane ID within wave
    
    // Each block processes ROWS_PER_BLOCK rows of the matrix
    const int row_start = bid * ROWS_PER_BLOCK;
    const int row_end = min(row_start + ROWS_PER_BLOCK, nrows_x);
    
    const block_q4_0* x = (const block_q4_0*) vx;
    const block_q8_1* y = (const block_q8_1*) vy;
    
    // Number of Q4_0 blocks per row
    const int blocks_per_row = ncols_x / QK4_0;
    
    float result[ROWS_PER_BLOCK];
    for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
        result[i] = 0.0f;
    }
    
    // Process blocks in tiles to maximize LDS usage
    for (int block_offset = 0; block_offset < blocks_per_row; block_offset += 8) {
        const int blocks_to_process = min(8, blocks_per_row - block_offset);
        
        // Cooperatively load input vector tile into LDS
        if (tid < blocks_to_process * 32) {
            const int block_idx = block_offset + tid / 32;
            const int elem_idx = tid % 32;
            if (block_idx < blocks_per_row) {
                // Load and dequantize Q8_1 input vector element
                const int8_t q8_val = y[block_idx].qs[elem_idx];
                const half2 ds = y[block_idx].ds;
                const float d = __half2float(__low2half(ds));
                const float s = __half2float(__high2half(ds));
                s_vec[tid] = q8_val * d + s;
            }
        }
        
        // Load scale factors into LDS
        if (tid < blocks_to_process) {
            const int block_idx = block_offset + tid;
            if (block_idx < blocks_per_row) {
                s_scales[tid] = __low2half(y[block_idx].ds);
            }
        }
        
        __syncthreads();
        
        // Each thread processes one or more rows
        for (int row_idx = row_start + tid / 32; row_idx < row_end; row_idx += BLOCK_SIZE / 32) {
            const int local_row = row_idx - row_start;
            
            // Process current tile of blocks
            for (int b = 0; b < blocks_to_process; ++b) {
                const int global_block = block_offset + b;
                const block_q4_0* block_ptr = &x[row_idx * blocks_per_row + global_block];
                
                // Use V_DOT8_I32_I4 to compute dot product for this block
                int32_t block_sum = 0;
                
                // Process 32 values as 4 groups of 8
                const uint32_t* q4_ptr = reinterpret_cast<const uint32_t*>(block_ptr->qs);
                
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t q4_vals = q4_ptr[j];
                    
                    // Pack 8 dequantized input values from LDS
                    uint32_t q8_packed = 0;
                    for (int k = 0; k < 8; ++k) {
                        const float val = s_vec[b * 32 + j * 8 + k];
                        // Convert back to int8 range for dot product
                        const int8_t q8_val = static_cast<int8_t>(roundf(val * 127.0f / __half2float(s_scales[b])));
                        q8_packed |= (static_cast<uint32_t>(q8_val) & 0x0F) << (k * 4);
                    }
                    
                    block_sum += gfx906_dot8_i4(q4_vals, q8_packed);
                }
                
                // Apply scale factor and accumulate
                const float scale = __half2float(block_ptr->d) * __half2float(s_scales[b]);
                result[local_row] += scale * (block_sum - 8 * 32);  // Subtract offset
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
        const int row_idx = row_start + i;
        if (row_idx < nrows_dst && tid == i % BLOCK_SIZE) {
            dst[row_idx] = result[i];
        }
    }
}

// Optimized dequantization kernel for Q4_0 using V_DOT8_I32_I4
__global__ void dequantize_q4_0_gfx906(
    const block_q4_0* __restrict__ x,
    float* __restrict__ y,
    const int64_t nb32) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int total_blocks = gridDim.x * blockDim.x / 32;
    
    if (tid >= total_blocks) return;
    
    const int block_idx = tid;
    const block_q4_0* block_ptr = &x[block_idx];
    float* out_ptr = &y[block_idx * 32];
    
    const float scale = __half2float(block_ptr->d);
    const uint32_t* q4_ptr = reinterpret_cast<const uint32_t*>(block_ptr->qs);
    
    // Process 32 values as 4 groups of 8
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t packed = q4_ptr[i];
        
        // Extract and dequantize 8 values
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int q_val = (packed >> (j * 4)) & 0x0F;
            out_ptr[i * 8 + j] = scale * (q_val - 8);
        }
    }
}

#endif // __HIP_DEVICE_COMPILE__
#endif // GGML_USE_HIP || __HIP_PLATFORM_AMD__