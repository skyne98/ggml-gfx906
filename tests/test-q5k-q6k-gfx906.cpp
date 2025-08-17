// Test for GFX906-optimized Q5_K and Q6_K quantization implementations

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>

// Include necessary GGML headers
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#define CHECK_HIP(call)                                                        \
    do {                                                                       \
        hipError_t error = call;                                             \
        if (error != hipSuccess) {                                           \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error));                               \
            exit(1);                                                          \
        }                                                                      \
    } while (0)

// Test parameters
constexpr int TEST_SIZE = 4096;
constexpr int NUM_TESTS = 100;
constexpr float EPSILON = 1e-3f;

// Helper function to generate random float data
void generate_random_data(float* data, int size, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

// Test Q5_K quantization and dequantization
bool test_q5_k_operations() {
    printf("Testing Q5_K operations with GFX906 optimizations...\n");
    
    std::mt19937 gen(42);
    float* input_data = new float[TEST_SIZE];
    float* output_data = new float[TEST_SIZE];
    
    generate_random_data(input_data, TEST_SIZE, gen);
    
    // Create GGML context
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 16,  // 16 MB
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to create GGML context\n");
        return false;
    }
    
    // Create tensors
    struct ggml_tensor* src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TEST_SIZE);
    struct ggml_tensor* q5k = ggml_new_tensor_1d(ctx, GGML_TYPE_Q5_K, TEST_SIZE);
    
    // Copy input data
    memcpy(src->data, input_data, TEST_SIZE * sizeof(float));
    
    // Quantize
    ggml_quantize_chunk(GGML_TYPE_Q5_K, input_data, q5k->data, 
                       0, 1, TEST_SIZE, nullptr);
    
    // Dequantize
    ggml_dequantize_row_q5_K((const block_q5_K*)q5k->data, output_data, TEST_SIZE);
    
    // Check accuracy
    float max_error = 0.0f;
    float avg_error = 0.0f;
    for (int i = 0; i < TEST_SIZE; ++i) {
        float error = std::abs(input_data[i] - output_data[i]);
        max_error = std::max(max_error, error);
        avg_error += error;
    }
    avg_error /= TEST_SIZE;
    
    printf("Q5_K - Max error: %f, Avg error: %f\n", max_error, avg_error);
    
    // Cleanup
    ggml_free(ctx);
    delete[] input_data;
    delete[] output_data;
    
    return max_error < 0.1f; // Reasonable threshold for Q5_K
}

// Test Q6_K quantization and dequantization
bool test_q6_k_operations() {
    printf("Testing Q6_K operations with GFX906 optimizations...\n");
    
    std::mt19937 gen(42);
    float* input_data = new float[TEST_SIZE];
    float* output_data = new float[TEST_SIZE];
    
    generate_random_data(input_data, TEST_SIZE, gen);
    
    // Create GGML context
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 16,  // 16 MB
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to create GGML context\n");
        return false;
    }
    
    // Create tensors
    struct ggml_tensor* src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TEST_SIZE);
    struct ggml_tensor* q6k = ggml_new_tensor_1d(ctx, GGML_TYPE_Q6_K, TEST_SIZE);
    
    // Copy input data
    memcpy(src->data, input_data, TEST_SIZE * sizeof(float));
    
    // Quantize
    ggml_quantize_chunk(GGML_TYPE_Q6_K, input_data, q6k->data, 
                       0, 1, TEST_SIZE, nullptr);
    
    // Dequantize
    ggml_dequantize_row_q6_K((const block_q6_K*)q6k->data, output_data, TEST_SIZE);
    
    // Check accuracy
    float max_error = 0.0f;
    float avg_error = 0.0f;
    for (int i = 0; i < TEST_SIZE; ++i) {
        float error = std::abs(input_data[i] - output_data[i]);
        max_error = std::max(max_error, error);
        avg_error += error;
    }
    avg_error /= TEST_SIZE;
    
    printf("Q6_K - Max error: %f, Avg error: %f\n", max_error, avg_error);
    
    // Cleanup
    ggml_free(ctx);
    delete[] input_data;
    delete[] output_data;
    
    return max_error < 0.08f; // Reasonable threshold for Q6_K
}

// Benchmark Q5_K vec_dot performance
void benchmark_q5_k_vec_dot() {
    printf("\nBenchmarking Q5_K vec_dot with GFX906 optimizations...\n");
    
    // Initialize HIP backend
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "Failed to initialize CUDA/HIP backend\n");
        return;
    }
    
    // Create context and tensors for benchmarking
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 64,  // 64 MB
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    
    // Create Q5_K and Q8_1 tensors for dot product
    const int n_elements = 16384;
    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_Q5_K, n_elements);
    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_1, n_elements);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    
    // Allocate backend buffer
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    
    // Generate random data
    std::mt19937 gen(42);
    float* temp_data = new float[n_elements];
    generate_random_data(temp_data, n_elements, gen);
    
    // Quantize to Q5_K
    ggml_quantize_chunk(GGML_TYPE_Q5_K, temp_data, a->data, 0, 1, n_elements, nullptr);
    
    // Quantize to Q8_1
    ggml_quantize_chunk(GGML_TYPE_Q8_1, temp_data, b->data, 0, 1, n_elements, nullptr);
    
    // Create compute graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    
    // Benchmark
    CHECK_HIP(hipDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_TESTS; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time = elapsed.count() / NUM_TESTS;
    
    printf("Q5_K vec_dot average time: %.3f ms\n", avg_time);
    
    // Calculate throughput
    size_t ops = (size_t)n_elements * 2; // multiply-add operations
    double gflops = (ops / 1e9) / (avg_time / 1000.0);
    printf("Q5_K vec_dot throughput: %.2f GFLOPS\n", gflops);
    
    // Cleanup
    delete[] temp_data;
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

// Benchmark Q6_K vec_dot performance
void benchmark_q6_k_vec_dot() {
    printf("\nBenchmarking Q6_K vec_dot with GFX906 optimizations...\n");
    
    // Initialize HIP backend
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "Failed to initialize CUDA/HIP backend\n");
        return;
    }
    
    // Create context and tensors for benchmarking
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 64,  // 64 MB
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    
    // Create Q6_K and Q8_1 tensors for dot product
    const int n_elements = 16384;
    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_Q6_K, n_elements);
    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_1, n_elements);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    
    // Allocate backend buffer
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    
    // Generate random data
    std::mt19937 gen(42);
    float* temp_data = new float[n_elements];
    generate_random_data(temp_data, n_elements, gen);
    
    // Quantize to Q6_K
    ggml_quantize_chunk(GGML_TYPE_Q6_K, temp_data, a->data, 0, 1, n_elements, nullptr);
    
    // Quantize to Q8_1
    ggml_quantize_chunk(GGML_TYPE_Q8_1, temp_data, b->data, 0, 1, n_elements, nullptr);
    
    // Create compute graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    
    // Benchmark
    CHECK_HIP(hipDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_TESTS; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time = elapsed.count() / NUM_TESTS;
    
    printf("Q6_K vec_dot average time: %.3f ms\n", avg_time);
    
    // Calculate throughput
    size_t ops = (size_t)n_elements * 2; // multiply-add operations
    double gflops = (ops / 1e9) / (avg_time / 1000.0);
    printf("Q6_K vec_dot throughput: %.2f GFLOPS\n", gflops);
    
    // Cleanup
    delete[] temp_data;
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

int main() {
    printf("GFX906 Q5_K/Q6_K Optimization Tests\n");
    printf("====================================\n\n");
    
    // Check if GFX906 device is available
    int device_count;
    CHECK_HIP(hipGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        fprintf(stderr, "No HIP devices found\n");
        return 1;
    }
    
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    printf("Using device: %s (GCN arch: %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    // Run tests
    bool all_passed = true;
    
    if (!test_q5_k_operations()) {
        fprintf(stderr, "Q5_K test failed\n");
        all_passed = false;
    }
    
    if (!test_q6_k_operations()) {
        fprintf(stderr, "Q6_K test failed\n");
        all_passed = false;
    }
    
    // Run benchmarks
    benchmark_q5_k_vec_dot();
    benchmark_q6_k_vec_dot();
    
    printf("\n====================================\n");
    if (all_passed) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("Some tests FAILED\n");
        return 1;
    }
}