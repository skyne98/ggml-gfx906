#pragma once

#ifdef __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_vector_types.h>

// Configuration for optimal GFX906 performance
#define GEMM_TILE_M 128
#define GEMM_TILE_N 128  
#define GEMM_TILE_K 32
#define GEMM_THREADS 256  // 4 wavefronts

// Helper function for packing half2
__device__ __forceinline__ uint32_t __pack_half2(const __half a, const __half b) {
    union {
        __half2 h2;
        uint32_t u32;
    } converter;
    converter.h2 = __half2{a, b};
    return converter.u32;
}

// Optimized FP16 GEMM kernel using inline assembly
__global__ void gemm_f16_gfx906_asm(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    // Shared memory for single-buffered tiles
    __shared__ __half lds_tile_a[GEMM_TILE_M][GEMM_TILE_K + 1];
    __shared__ __half lds_tile_b[GEMM_TILE_K][GEMM_TILE_N + 1];
    
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * GEMM_TILE_M;
    const int block_col = blockIdx.x * GEMM_TILE_N;
    
    // Each thread computes a 2x2 tile
    const int threads_per_row = GEMM_TILE_N / 2;
    const int thread_row = (tid / threads_per_row) * 2;
    const int thread_col = (tid % threads_per_row) * 2;
    
    // Accumulator registers
    float c_accum[2][2] = {{0.0f}};
    
    // Main loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += GEMM_TILE_K) {
        
        // Cooperative loading - each thread loads multiple elements
        const int elems_per_thread = (GEMM_TILE_M * GEMM_TILE_K) / GEMM_THREADS;
        
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            const int idx = tid * elems_per_thread + i;
            const int row = idx / GEMM_TILE_K;
            const int col = idx % GEMM_TILE_K;
            
            if (block_row + row < M && k_tile + col < K) {
                lds_tile_a[row][col] = A[(block_row + row) * K + k_tile + col];
            } else {
                lds_tile_a[row][col] = __float2half(0.0f);
            }
        }
        
        const int elems_per_thread_b = (GEMM_TILE_K * GEMM_TILE_N) / GEMM_THREADS;
        
        #pragma unroll
        for (int i = 0; i < elems_per_thread_b; i++) {
            const int idx = tid * elems_per_thread_b + i;
            const int row = idx / GEMM_TILE_N;
            const int col = idx % GEMM_TILE_N;
            
            if (k_tile + row < K && block_col + col < N) {
                lds_tile_b[row][col] = B[(k_tile + row) * N + block_col + col];
            } else {
                lds_tile_b[row][col] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // Core computation loop with inline assembly
        #pragma unroll
        for (int k = 0; k < GEMM_TILE_K; k += 2) {
            // Load packed data from LDS
            uint32_t a_packed[2];
            uint32_t b_packed[2];
            
            // Load A values (2 consecutive K values per row)
            a_packed[0] = *reinterpret_cast<const uint32_t*>(&lds_tile_a[thread_row][k]);
            a_packed[1] = *reinterpret_cast<const uint32_t*>(&lds_tile_a[thread_row + 1][k]);
            
            // Load and pack B values
            b_packed[0] = __pack_half2(lds_tile_b[k][thread_col], lds_tile_b[k + 1][thread_col]);
            b_packed[1] = __pack_half2(lds_tile_b[k][thread_col + 1], lds_tile_b[k + 1][thread_col + 1]);
            
            // Inline assembly for V_DOT2_F32_F16 instructions
            asm volatile(
                // Wait for LDS reads
                "s_waitcnt lgkmcnt(0)\n\t"
                
                // C[0][0] += A[0] * B[0]
                "v_dot2_f32_f16 %0, %4, %6, %0\n\t"
                
                // C[0][1] += A[0] * B[1]
                "v_dot2_f32_f16 %1, %4, %7, %1\n\t"
                
                // C[1][0] += A[1] * B[0]
                "v_dot2_f32_f16 %2, %5, %6, %2\n\t"
                
                // C[1][1] += A[1] * B[1]
                "v_dot2_f32_f16 %3, %5, %7, %3\n\t"
                
                : "+v"(c_accum[0][0]), "+v"(c_accum[0][1]),
                  "+v"(c_accum[1][0]), "+v"(c_accum[1][1])
                : "v"(a_packed[0]), "v"(a_packed[1]),
                  "v"(b_packed[0]), "v"(b_packed[1])
            );
        }
        
        __syncthreads();
    }
    
    // Store results
    const int global_row = block_row + thread_row;
    const int global_col = block_col + thread_col;
    
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            if (global_row + m < M && global_col + n < N) {
                const int idx = (global_row + m) * N + (global_col + n);
                float result = alpha * c_accum[m][n];
                if (beta != 0.0f) {
                    result += beta * __half2float(C[idx]);
                }
                C[idx] = __float2half(result);
            }
        }
    }
}

// Optimized FP32 GEMM kernel
__global__ void gemm_f32_gfx906_asm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    // Shared memory tiles
    __shared__ float lds_tile_a[GEMM_TILE_M][GEMM_TILE_K + 1];
    __shared__ float lds_tile_b[GEMM_TILE_K][GEMM_TILE_N + 1];
    
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * GEMM_TILE_M;
    const int block_col = blockIdx.x * GEMM_TILE_N;
    
    // Each thread computes a 4x4 tile for FP32
    const int threads_per_row = GEMM_TILE_N / 4;
    const int thread_row = (tid / threads_per_row) * 4;
    const int thread_col = (tid % threads_per_row) * 4;
    
    // Accumulator registers
    float c_accum[4][4] = {{0.0f}};
    
    // Main loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += GEMM_TILE_K) {
        
        // Load tiles cooperatively
        const int elems_per_thread = (GEMM_TILE_M * GEMM_TILE_K) / GEMM_THREADS;
        
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            const int idx = tid * elems_per_thread + i;
            const int row = idx / GEMM_TILE_K;
            const int col = idx % GEMM_TILE_K;
            
            if (block_row + row < M && k_tile + col < K) {
                lds_tile_a[row][col] = A[(block_row + row) * K + k_tile + col];
            } else {
                lds_tile_a[row][col] = 0.0f;
            }
        }
        
        const int elems_per_thread_b = (GEMM_TILE_K * GEMM_TILE_N) / GEMM_THREADS;
        
        #pragma unroll
        for (int i = 0; i < elems_per_thread_b; i++) {
            const int idx = tid * elems_per_thread_b + i;
            const int row = idx / GEMM_TILE_N;
            const int col = idx % GEMM_TILE_N;
            
            if (k_tile + row < K && block_col + col < N) {
                lds_tile_b[row][col] = B[(k_tile + row) * N + block_col + col];
            } else {
                lds_tile_b[row][col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Core computation with FMA
        #pragma unroll
        for (int k = 0; k < GEMM_TILE_K; k++) {
            float a_vals[4];
            float b_vals[4];
            
            // Load from LDS
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                a_vals[m] = lds_tile_a[thread_row + m][k];
            }
            
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                b_vals[n] = lds_tile_b[k][thread_col + n];
            }
            
            // FMA operations with inline assembly
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    asm volatile(
                        "v_fma_f32 %0, %1, %2, %0"
                        : "+v"(c_accum[m][n])
                        : "v"(a_vals[m]), "v"(b_vals[n])
                    );
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    const int global_row = block_row + thread_row;
    const int global_col = block_col + thread_col;
    
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            if (global_row + m < M && global_col + n < N) {
                const int idx = (global_row + m) * N + (global_col + n);
                C[idx] = alpha * c_accum[m][n] + beta * C[idx];
            }
        }
    }
}

#endif // __HIP_PLATFORM_AMD__