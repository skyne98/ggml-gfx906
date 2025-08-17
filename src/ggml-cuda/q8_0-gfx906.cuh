#pragma once

// Optimized Q8_0 quantization kernels for GFX906 using V_DOT4_I32_I8
// Implements fused dequantize-and-GEMV operations with LDS optimization

#include "common.cuh"
#include "gfx906-config.cuh"

#if defined(GGML_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#    ifdef __HIP_DEVICE_COMPILE__

// V_DOT4_I32_I8 intrinsic is defined in gfx906-config.cuh
// Use the existing gfx906_dot4_i8 function from there

// Optimized Q8_0 to Q8_1 dot product using V_DOT4_I32_I8
template <int vdr>
__device__ __forceinline__ float vec_dot_q8_0_q8_1_gfx906(const void * __restrict__ vbq,
                                                          const block_q8_1 * __restrict__ bq8_1,
                                                          const int & kbx,
                                                          const int & iqs) {
    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;

    int32_t sumi = 0;

#        pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int block_idx = kbx + i;

        // Load Q8_0 data: 32 8-bit values
        // We process 4 values at a time using V_DOT4_I32_I8
        const int8_t * q8_0_ptr = bq8_0[block_idx].qs + iqs;
        const int8_t * q8_1_ptr = bq8_1[block_idx].qs + iqs;

        // Process 8 groups of 4 values (32 total)
#        pragma unroll
        for (int j = 0; j < 8; ++j) {
            // Load 4 int8 values from Q8_0 and pack into 32-bit word
            int32_t q8_0_vals = *reinterpret_cast<const int32_t *>(q8_0_ptr + j * 4);

            // Load 4 int8 values from Q8_1 and pack into 32-bit word
            int32_t q8_1_vals = *reinterpret_cast<const int32_t *>(q8_1_ptr + j * 4);

            // Use V_DOT4_I32_I8 instruction for 4-way dot product
            sumi += gfx906_dot4_i8(q8_0_vals, q8_1_vals);
        }
    }

    // Apply scaling factors
    const float  d8_0  = __half2float(bq8_0[kbx].d);
    const float2 ds8_1 = __half22float2(bq8_1[kbx].ds);

    return d8_0 * ds8_1.x * sumi;
}

// Fused dequantize-and-GEMV kernel optimized for GFX906
template <int BLOCK_SIZE, int ROWS_PER_BLOCK>
__global__ void mul_mat_vec_q8_0_q8_1_gfx906(const void * __restrict__ vx,
                                             const void * __restrict__ vy,
                                             float * __restrict__ dst,
                                             const int ncols_x,
                                             const int nrows_x,
                                             const int nrows_dst) {
    // Configure for 256 threads (4 wavefronts) per block
    static_assert(BLOCK_SIZE == 256, "Block size must be 256 for GFX906 optimization");

    // Shared memory for caching input vector and results
    __shared__ float s_vec[256];                    // Cache portion of input vector
    __shared__ float s_partial[ROWS_PER_BLOCK][8];  // Partial results per row

    const int tid     = threadIdx.x;
    const int bid     = blockIdx.x;
    const int warp_id = tid / 64;  // Wave ID within block
    const int lane_id = tid % 64;  // Lane ID within wave

    // Each block processes ROWS_PER_BLOCK rows of the matrix
    const int row_start = bid * ROWS_PER_BLOCK;
    const int row_end   = min(row_start + ROWS_PER_BLOCK, nrows_x);

    const block_q8_0 * x = (const block_q8_0 *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    // Number of Q8_0 blocks per row
    const int blocks_per_row = ncols_x / QK8_0;

    // Initialize partial results
    if (tid < ROWS_PER_BLOCK * 8) {
        s_partial[tid / 8][tid % 8] = 0.0f;
    }
    __syncthreads();

    // Process blocks in tiles to maximize LDS usage
    for (int block_offset = 0; block_offset < blocks_per_row; block_offset += 8) {
        const int blocks_to_process = min(8, blocks_per_row - block_offset);

        // Cooperatively load and dequantize input vector tile into LDS
        if (tid < blocks_to_process * 32) {
            const int block_idx = block_offset + tid / 32;
            const int elem_idx  = tid % 32;
            if (block_idx < blocks_per_row) {
                // Load and dequantize Q8_1 input vector element
                const int8_t q8_val = y[block_idx].qs[elem_idx];
                const half2  ds     = y[block_idx].ds;
                const float  d      = __half2float(__low2half(ds));
                s_vec[tid]          = q8_val * d;
            }
        }
        __syncthreads();

        // Each thread processes one or more rows using V_DOT4_I32_I8
        for (int row_idx = row_start + tid / 32; row_idx < row_end; row_idx += BLOCK_SIZE / 32) {
            const int local_row = row_idx - row_start;
            float     row_sum   = 0.0f;

            // Process current tile of blocks
            for (int b = 0; b < blocks_to_process; ++b) {
                const int          global_block = block_offset + b;
                const block_q8_0 * block_ptr    = &x[row_idx * blocks_per_row + global_block];

                // Use V_DOT4_I32_I8 to compute dot product for this block
                int32_t block_sum = 0;

                // Process 32 values as 8 groups of 4
                const int32_t * q8_0_ptr = reinterpret_cast<const int32_t *>(block_ptr->qs);

#        pragma unroll
                for (int j = 0; j < 8; ++j) {
                    // Load 4 packed int8 values from Q8_0
                    int32_t q8_0_vals = q8_0_ptr[j];

                    // Pack 4 dequantized input values from LDS back to int8
                    int32_t q8_1_packed = 0;
                    for (int k = 0; k < 4; ++k) {
                        const float  val    = s_vec[b * 32 + j * 4 + k];
                        const half2  ds     = y[global_block].ds;
                        const float  d      = __half2float(__low2half(ds));
                        // Convert back to int8 range for dot product
                        const int8_t q8_val = static_cast<int8_t>(roundf(val / d));
                        q8_1_packed |= (static_cast<uint32_t>(q8_val) & 0xFF) << (k * 8);
                    }

                    // Use V_DOT4_I32_I8 instruction
                    block_sum += gfx906_dot4_i8(q8_0_vals, q8_1_packed);
                }

                // Apply scale factor and accumulate
                const float scale = __half2float(block_ptr->d) * __half2float(__low2half(y[global_block].ds));
                row_sum += scale * block_sum;
            }

            // Store partial result in shared memory
            atomicAdd(&s_partial[local_row][tid % 8], row_sum);
        }

        __syncthreads();
    }

    // Reduce partial results and write to global memory
    if (tid < ROWS_PER_BLOCK) {
        const int row_idx = row_start + tid;
        if (row_idx < nrows_dst) {
            float final_sum = 0.0f;
            for (int i = 0; i < 8; ++i) {
                final_sum += s_partial[tid][i];
            }
            dst[row_idx] = final_sum;
        }
    }
}

// Optimized dequantization kernel for Q8_0
__global__ void dequantize_q8_0_gfx906(const block_q8_0 * __restrict__ x, float * __restrict__ y, const int64_t nb32) {
    const int tid          = threadIdx.x + blockIdx.x * blockDim.x;
    const int total_blocks = gridDim.x * blockDim.x / 32;

    if (tid >= total_blocks) {
        return;
    }

    const int          block_idx = tid;
    const block_q8_0 * block_ptr = &x[block_idx];
    float *            out_ptr   = &y[block_idx * 32];

    const float    scale  = __half2float(block_ptr->d);
    const int8_t * q8_ptr = block_ptr->qs;

    // Process 32 values as 8 groups of 4 using vectorized loads
#        pragma unroll
    for (int i = 0; i < 8; ++i) {
        // Load 4 int8 values at once
        int32_t packed = *reinterpret_cast<const int32_t *>(q8_ptr + i * 4);

        // Extract and dequantize 4 values
#        pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int8_t q_val = (packed >> (j * 8)) & 0xFF;
            out_ptr[i * 4 + j] = scale * q_val;
        }
    }
}

// Optimized Q8_0 matrix multiplication kernel using V_DOT4_I32_I8
template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void gemm_q8_0_gfx906(const block_q8_0 * __restrict__ A,
                                 const block_q8_0 * __restrict__ B,
                                 float * __restrict__ C,
                                 const int M,
                                 const int N,
                                 const int K) {
    // Shared memory tiles for A and B matrices
    __shared__ int8_t tile_A[BLOCK_M][BLOCK_K];
    __shared__ int8_t tile_B[BLOCK_K][BLOCK_N];
    __shared__ float  scale_A[BLOCK_M];
    __shared__ float  scale_B[BLOCK_N];

    const int tid = threadIdx.x;
    const int tx  = tid % BLOCK_N;
    const int ty  = tid / BLOCK_N;

    const int bx = blockIdx.x * BLOCK_N;
    const int by = blockIdx.y * BLOCK_M;

    // Initialize accumulator
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // Loop over K dimension in tiles
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // Cooperatively load A tile
        if (ty < BLOCK_M && k_tile + tx < K) {
            const int a_idx    = (by + ty) * (K / 32) + (k_tile + tx) / 32;
            const int a_offset = (k_tile + tx) % 32;
            tile_A[ty][tx]     = A[a_idx].qs[a_offset];
            if (tx == 0) {
                scale_A[ty] = __half2float(A[a_idx].d);
            }
        }

        // Cooperatively load B tile
        if (ty < BLOCK_K && bx + tx < N) {
            const int b_idx    = (k_tile + ty) * (N / 32) + (bx + tx) / 32;
            const int b_offset = (bx + tx) % 32;
            tile_B[ty][tx]     = B[b_idx].qs[b_offset];
            if (ty == 0) {
                scale_B[tx] = __half2float(B[b_idx].d);
            }
        }

        __syncthreads();

        // Compute using V_DOT4_I32_I8
        if (ty < BLOCK_M && tx < BLOCK_N) {
            // Process 4 elements at a time using V_DOT4_I32_I8
            for (int k = 0; k < BLOCK_K; k += 4) {
                // Pack 4 int8 values from A
                int32_t a_packed = 0;
                for (int i = 0; i < 4; ++i) {
                    a_packed |= (static_cast<uint32_t>(tile_A[ty][k + i]) & 0xFF) << (i * 8);
                }

                // Pack 4 int8 values from B
                int32_t b_packed = 0;
                for (int i = 0; i < 4; ++i) {
                    b_packed |= (static_cast<uint32_t>(tile_B[k + i][tx]) & 0xFF) << (i * 8);
                }

                // Use V_DOT4_I32_I8 instruction
                acc[0] += gfx906_dot4_i8(a_packed, b_packed) * scale_A[ty] * scale_B[tx];
            }
        }

        __syncthreads();
    }

    // Write result to global memory
    if (by + ty < M && bx + tx < N) {
        C[(by + ty) * N + bx + tx] = acc[0];
    }
}

#    endif  // __HIP_DEVICE_COMPILE__
#endif      // GGML_USE_HIP || __HIP_PLATFORM_AMD__
