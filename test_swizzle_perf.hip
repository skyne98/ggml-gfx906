#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>

#define GGML_HIP_GFX906_OPTIMIZED
#include "src/ggml-cuda/gfx906-wave-primitives-swizzle.cuh"

using namespace gfx906;

// Test kernel for native swizzle
__global__ void test_swizzle(float* input, float* output, int n_iters) {
    float value = input[threadIdx.x];
    float result = 0;
    
    // Run multiple iterations for timing
    for (int i = 0; i < n_iters; i++) {
        result = wave_reduce_sum_swizzle(value);
    }
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = result;
    }
}

// Test kernel for shuffle 
__global__ void test_shuffle(float* input, float* output, int n_iters) {
    float value = input[threadIdx.x];
    float result = 0;
    
    for (int iter = 0; iter < n_iters; iter++) {
        float temp = value;
        #pragma unroll
        for (int offset = 32; offset >= 1; offset >>= 1) {
            temp += __shfl_xor(temp, offset, 64);
        }
        result = temp;
    }
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = result;
    }
}

int main() {
    const int n_blocks = 1000;
    const int n_threads = 64;
    const int n_iters = 1000;
    
    float *d_input, *d_output;
    float h_input[64];
    
    // Initialize input
    for (int i = 0; i < 64; i++) {
        h_input[i] = 1.0f;
    }
    
    hipMalloc(&d_input, 64 * sizeof(float));
    hipMalloc(&d_output, n_blocks * sizeof(float));
    hipMemcpy(d_input, h_input, 64 * sizeof(float), hipMemcpyHostToDevice);
    
    // Warmup
    test_swizzle<<<n_blocks, n_threads>>>(d_input, d_output, 10);
    test_shuffle<<<n_blocks, n_threads>>>(d_input, d_output, 10);
    hipDeviceSynchronize();
    
    // Test native swizzle
    auto start = std::chrono::high_resolution_clock::now();
    test_swizzle<<<n_blocks, n_threads>>>(d_input, d_output, n_iters);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto swizzle_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Test shuffle
    start = std::chrono::high_resolution_clock::now();
    test_shuffle<<<n_blocks, n_threads>>>(d_input, d_output, n_iters);
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto shuffle_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Verify correctness
    float h_output[n_blocks];
    hipMemcpy(h_output, d_output, sizeof(float), hipMemcpyDeviceToHost);
    printf("Result check: %.2f (expected 64.0)\n", h_output[0]);
    
    // Performance comparison
    printf("\nPerformance Comparison (%d blocks, %d iterations):\n", n_blocks, n_iters);
    printf("Native DS_SWIZZLE: %ld us\n", swizzle_time);
    printf("Shuffle XOR:       %ld us\n", shuffle_time);
    printf("Speedup:           %.2fx\n", (float)shuffle_time / swizzle_time);
    
    hipFree(d_input);
    hipFree(d_output);
    
    return 0;
}