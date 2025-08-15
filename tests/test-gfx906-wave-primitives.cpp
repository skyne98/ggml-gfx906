// Test suite for GFX906 wave-level primitives
// This test verifies the correctness of wave operations on AMD GFX906 hardware

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <cmath>

// Include the wave primitives header
#define GGML_USE_HIP
#define GGML_HIP_GFX906_OPTIMIZED
#include "../src/ggml-cuda/gfx906-wave-primitives.cuh"

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Test kernels
__global__ void test_wave_reduce_sum(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float value = input[tid];
        float sum = gfx906::wave_reduce_sum(value);
        
        // Only wave leader writes the result
        if (gfx906::is_wave_leader()) {
            output[tid / gfx906::WAVE_SIZE] = sum;
        }
    }
}

__global__ void test_wave_broadcast(float* input, float* output, int n, int src_lane) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float value = input[tid];
        float broadcasted = gfx906::wave_broadcast(value, src_lane);
        output[tid] = broadcasted;
    }
}

__global__ void test_wave_shuffle(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float value = input[tid];
        int src_lane = (gfx906::__lane_id() + 1) % gfx906::WAVE_SIZE;
        float shuffled = gfx906::wave_shuffle(value, src_lane);
        output[tid] = shuffled;
    }
}

__global__ void test_wave_prefix_sum(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float value = input[tid];
        float prefix = gfx906::wave_prefix_sum_inclusive(value);
        output[tid] = prefix;
    }
}

__global__ void test_wave_reduce_max(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float value = input[tid];
        float max_val = gfx906::wave_reduce_max(value);
        
        if (gfx906::is_wave_leader()) {
            output[tid / gfx906::WAVE_SIZE] = max_val;
        }
    }
}

__global__ void test_wave_dot4_i8(int32_t* a, int32_t* b, int32_t* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int32_t val_a = a[tid];
        int32_t val_b = b[tid];
        int32_t dot = gfx906::wave_dot4_i8(val_a, val_b);
        
        if (gfx906::is_wave_leader()) {
            output[tid / gfx906::WAVE_SIZE] = dot;
        }
    }
}

__global__ void test_block_reduce(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (tid < n) ? input[tid] : 0.0f;
    
    float block_sum = gfx906::block_reduce_sum(value);
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = block_sum;
    }
}

// Test functions
bool test_reduce_sum() {
    const int n = 256;  // 4 waves
    const int n_waves = n / gfx906::WAVE_SIZE;
    
    float *d_input, *d_output;
    std::vector<float> h_input(n);
    std::vector<float> h_output(n_waves);
    
    // Initialize input
    for (int i = 0; i < n; i++) {
        h_input[i] = i % 64;  // Each wave gets 0-63
    }
    
    HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, n_waves * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), n * sizeof(float), hipMemcpyHostToDevice));
    
    test_wave_reduce_sum<<<1, n>>>(d_input, d_output, n);
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, n_waves * sizeof(float), hipMemcpyDeviceToHost));
    
    // Verify results
    float expected = 0;
    for (int i = 0; i < 64; i++) expected += i;
    
    bool passed = true;
    for (int i = 0; i < n_waves; i++) {
        if (std::abs(h_output[i] - expected) > 1e-5) {
            printf("Wave %d: Expected %.2f, got %.2f\n", i, expected, h_output[i]);
            passed = false;
        }
    }
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    
    printf("test_reduce_sum: %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_broadcast() {
    const int n = 64;  // 1 wave
    const int src_lane = 10;
    
    float *d_input, *d_output;
    std::vector<float> h_input(n);
    std::vector<float> h_output(n);
    
    // Initialize input
    for (int i = 0; i < n; i++) {
        h_input[i] = i;
    }
    
    HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, n * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), n * sizeof(float), hipMemcpyHostToDevice));
    
    test_wave_broadcast<<<1, n>>>(d_input, d_output, n, src_lane);
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, n * sizeof(float), hipMemcpyDeviceToHost));
    
    // Verify results - all should have value from src_lane
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(h_output[i] - h_input[src_lane]) > 1e-5) {
            printf("Lane %d: Expected %.2f, got %.2f\n", i, h_input[src_lane], h_output[i]);
            passed = false;
        }
    }
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    
    printf("test_broadcast: %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_shuffle() {
    const int n = 64;  // 1 wave
    
    float *d_input, *d_output;
    std::vector<float> h_input(n);
    std::vector<float> h_output(n);
    
    // Initialize input
    for (int i = 0; i < n; i++) {
        h_input[i] = i;
    }
    
    HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, n * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), n * sizeof(float), hipMemcpyHostToDevice));
    
    test_wave_shuffle<<<1, n>>>(d_input, d_output, n);
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, n * sizeof(float), hipMemcpyDeviceToHost));
    
    // Verify results - each lane should have value from (lane+1)%64
    bool passed = true;
    for (int i = 0; i < n; i++) {
        float expected = h_input[(i + 1) % n];
        if (std::abs(h_output[i] - expected) > 1e-5) {
            printf("Lane %d: Expected %.2f, got %.2f\n", i, expected, h_output[i]);
            passed = false;
        }
    }
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    
    printf("test_shuffle: %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_prefix_sum() {
    const int n = 64;  // 1 wave
    
    float *d_input, *d_output;
    std::vector<float> h_input(n);
    std::vector<float> h_output(n);
    
    // Initialize input
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // All ones for easy verification
    }
    
    HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, n * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), n * sizeof(float), hipMemcpyHostToDevice));
    
    test_wave_prefix_sum<<<1, n>>>(d_input, d_output, n);
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, n * sizeof(float), hipMemcpyDeviceToHost));
    
    // Verify results - should be 1, 2, 3, ..., 64
    bool passed = true;
    for (int i = 0; i < n; i++) {
        float expected = i + 1;
        if (std::abs(h_output[i] - expected) > 1e-5) {
            printf("Lane %d: Expected %.2f, got %.2f\n", i, expected, h_output[i]);
            passed = false;
        }
    }
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    
    printf("test_prefix_sum: %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_reduce_max() {
    const int n = 128;  // 2 waves
    const int n_waves = n / gfx906::WAVE_SIZE;
    
    float *d_input, *d_output;
    std::vector<float> h_input(n);
    std::vector<float> h_output(n_waves);
    
    // Initialize input with increasing values
    for (int i = 0; i < n; i++) {
        h_input[i] = i;
    }
    
    HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, n_waves * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), n * sizeof(float), hipMemcpyHostToDevice));
    
    test_wave_reduce_max<<<1, n>>>(d_input, d_output, n);
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, n_waves * sizeof(float), hipMemcpyDeviceToHost));
    
    // Verify results
    bool passed = true;
    for (int i = 0; i < n_waves; i++) {
        float expected = (i + 1) * 64 - 1;  // Max value in each wave
        if (std::abs(h_output[i] - expected) > 1e-5) {
            printf("Wave %d: Expected %.2f, got %.2f\n", i, expected, h_output[i]);
            passed = false;
        }
    }
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    
    printf("test_reduce_max: %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_block_reduce() {
    const int block_size = 256;
    const int n = block_size * 4;  // 4 blocks
    
    float *d_input, *d_output;
    std::vector<float> h_input(n);
    std::vector<float> h_output(4);
    
    // Initialize input
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }
    
    HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, 4 * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), n * sizeof(float), hipMemcpyHostToDevice));
    
    test_block_reduce<<<4, block_size>>>(d_input, d_output, n);
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, 4 * sizeof(float), hipMemcpyDeviceToHost));
    
    // Verify results
    bool passed = true;
    for (int i = 0; i < 4; i++) {
        float expected = block_size;
        if (std::abs(h_output[i] - expected) > 1e-5) {
            printf("Block %d: Expected %.2f, got %.2f\n", i, expected, h_output[i]);
            passed = false;
        }
    }
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    
    printf("test_block_reduce: %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

int main() {
    // Check if we have a GFX906 device
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        printf("No HIP devices found!\n");
        return 1;
    }
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Testing on device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Warp size: %d\n", prop.warpSize);
    
    // Check if this is GFX906
    if (prop.warpSize != 64) {
        printf("Warning: This device doesn't appear to be GFX906 (warp size != 64)\n");
    }
    
    printf("\nRunning GFX906 Wave Primitives Tests:\n");
    printf("=====================================\n");
    
    bool all_passed = true;
    all_passed &= test_reduce_sum();
    all_passed &= test_broadcast();
    all_passed &= test_shuffle();
    all_passed &= test_prefix_sum();
    all_passed &= test_reduce_max();
    all_passed &= test_block_reduce();
    
    printf("\n=====================================\n");
    printf("Overall result: %s\n", all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    
    return all_passed ? 0 : 1;
}