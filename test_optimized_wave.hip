#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <chrono>

// Include optimized wave primitives
#ifndef GGML_HIP_GFX906_OPTIMIZED
#define GGML_HIP_GFX906_OPTIMIZED
#endif
#ifndef __gfx906__
#define __gfx906__
#endif

#include "src/ggml-cuda/gfx906-wave-primitives-optimized.cuh"

// Test kernel for reduction
__global__ void test_optimized_reduce(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float value = input[tid];
        float sum = wave_reduce_sum(value);
        
        if (gfx906::is_wave_leader()) {
            output[tid / gfx906::WAVE_SIZE] = sum;
        }
    }
}

// Benchmark kernel comparing DS_SWIZZLE vs __shfl_xor
__global__ void benchmark_reduce_swizzle(float* data, int iterations) {
    float value = threadIdx.x;
    
    for (int i = 0; i < iterations; i++) {
        value = gfx906::wave_reduce_sum_optimized(value);
    }
    
    if (threadIdx.x == 0) {
        data[blockIdx.x] = value;
    }
}

__global__ void benchmark_reduce_shfl(float* data, int iterations) {
    float value = threadIdx.x;
    
    for (int i = 0; i < iterations; i++) {
        // Manual __shfl_xor reduction
        value += __shfl_xor(value, 32, 64);
        value += __shfl_xor(value, 16, 64);
        value += __shfl_xor(value, 8, 64);
        value += __shfl_xor(value, 4, 64);
        value += __shfl_xor(value, 2, 64);
        value += __shfl_xor(value, 1, 64);
    }
    
    if (threadIdx.x == 0) {
        data[blockIdx.x] = value;
    }
}

int main() {
    const int n = 256;
    const int n_waves = n / gfx906::WAVE_SIZE;
    
    float *d_input, *d_output;
    float h_input[n], h_output[n_waves];
    
    // Initialize test data
    for (int i = 0; i < n; i++) {
        h_input[i] = i % 64;
    }
    
    hipMalloc(&d_input, n * sizeof(float));
    hipMalloc(&d_output, n_waves * sizeof(float));
    hipMemcpy(d_input, h_input, n * sizeof(float), hipMemcpyHostToDevice);
    
    // Test correctness
    test_optimized_reduce<<<1, n>>>(d_input, d_output, n);
    hipDeviceSynchronize();
    
    hipMemcpy(h_output, d_output, n_waves * sizeof(float), hipMemcpyDeviceToHost);
    
    printf("Testing Optimized Wave Primitives with DS_SWIZZLE:\n");
    printf("===================================================\n\n");
    
    // Verify reduction
    float expected = 0;
    for (int i = 0; i < 64; i++) expected += i;
    
    bool passed = true;
    for (int i = 0; i < n_waves; i++) {
        if (std::abs(h_output[i] - expected) > 1e-5) {
            printf("Wave %d: Expected %.2f, got %.2f\n", i, expected, h_output[i]);
            passed = false;
        }
    }
    printf("Reduction test: %s\n\n", passed ? "PASSED" : "FAILED");
    
    // Benchmark
    const int iterations = 100000;
    const int blocks = 1024;
    float *d_bench;
    hipMalloc(&d_bench, blocks * sizeof(float));
    
    // Warmup
    benchmark_reduce_swizzle<<<blocks, 64>>>(d_bench, 100);
    benchmark_reduce_shfl<<<blocks, 64>>>(d_bench, 100);
    hipDeviceSynchronize();
    
    // Benchmark DS_SWIZZLE
    auto start = std::chrono::high_resolution_clock::now();
    benchmark_reduce_swizzle<<<blocks, 64>>>(d_bench, iterations);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto swizzle_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Benchmark __shfl_xor
    start = std::chrono::high_resolution_clock::now();
    benchmark_reduce_shfl<<<blocks, 64>>>(d_bench, iterations);
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto shfl_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    printf("Performance Comparison:\n");
    printf("=======================\n");
    printf("DS_SWIZZLE time:  %ld us\n", swizzle_time);
    printf("__shfl_xor time:  %ld us\n", shfl_time);
    printf("Speedup: %.2fx\n", (float)shfl_time / swizzle_time);
    
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_bench);
    
    return 0;
}