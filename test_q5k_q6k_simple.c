// Simple test for Q5_K and Q6_K quantization on GFX906
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#define TEST_SIZE 4096
#define EPSILON 0.1f

void generate_random_data(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

int test_q5_k() {
    printf("Testing Q5_K quantization...\n");
    
    float* input = (float*)malloc(TEST_SIZE * sizeof(float));
    float* output = (float*)malloc(TEST_SIZE * sizeof(float));
    
    generate_random_data(input, TEST_SIZE);
    
    // Get quantized size
    size_t q5k_size = ggml_row_size(GGML_TYPE_Q5_K, TEST_SIZE);
    void* quantized = malloc(q5k_size);
    
    // Quantize
    ggml_quantize_chunk(GGML_TYPE_Q5_K, input, quantized, 0, 1, TEST_SIZE, NULL);
    
    // Dequantize
    ggml_dequantize_row_q5_K(quantized, output, TEST_SIZE);
    
    // Check accuracy
    float max_error = 0.0f;
    float avg_error = 0.0f;
    for (int i = 0; i < TEST_SIZE; ++i) {
        float error = fabsf(input[i] - output[i]);
        if (error > max_error) max_error = error;
        avg_error += error;
    }
    avg_error /= TEST_SIZE;
    
    printf("Q5_K - Max error: %f, Avg error: %f\n", max_error, avg_error);
    
    free(input);
    free(output);
    free(quantized);
    
    return max_error < EPSILON ? 0 : 1;
}

int test_q6_k() {
    printf("Testing Q6_K quantization...\n");
    
    float* input = (float*)malloc(TEST_SIZE * sizeof(float));
    float* output = (float*)malloc(TEST_SIZE * sizeof(float));
    
    generate_random_data(input, TEST_SIZE);
    
    // Get quantized size
    size_t q6k_size = ggml_row_size(GGML_TYPE_Q6_K, TEST_SIZE);
    void* quantized = malloc(q6k_size);
    
    // Quantize
    ggml_quantize_chunk(GGML_TYPE_Q6_K, input, quantized, 0, 1, TEST_SIZE, NULL);
    
    // Dequantize
    ggml_dequantize_row_q6_K(quantized, output, TEST_SIZE);
    
    // Check accuracy
    float max_error = 0.0f;
    float avg_error = 0.0f;
    for (int i = 0; i < TEST_SIZE; ++i) {
        float error = fabsf(input[i] - output[i]);
        if (error > max_error) max_error = error;
        avg_error += error;
    }
    avg_error /= TEST_SIZE;
    
    printf("Q6_K - Max error: %f, Avg error: %f\n", max_error, avg_error);
    
    free(input);
    free(output);
    free(quantized);
    
    return max_error < EPSILON ? 0 : 1;
}

int test_q5k_q6k_gpu() {
    printf("\nTesting Q5_K and Q6_K on GPU (GFX906)...\n");
    
    // Initialize CUDA/HIP backend
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        printf("Failed to initialize CUDA/HIP backend\n");
        return 1;
    }
    
    // Create context
    struct ggml_init_params params = {
        .mem_size = 128 * 1024 * 1024,  // 128 MB
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to create context\n");
        ggml_backend_free(backend);
        return 1;
    }
    
    // Test Q5_K matrix multiplication
    printf("Creating Q5_K tensors for GPU test...\n");
    struct ggml_tensor* a_q5k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, 512, 512);
    struct ggml_tensor* b_q8_1 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 512, 512);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a_q5k, b_q8_1);
    
    // Allocate backend buffer
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    
    // Generate test data
    float* temp = (float*)malloc(512 * 512 * sizeof(float));
    generate_random_data(temp, 512 * 512);
    
    // Set tensor data through backend
    ggml_backend_tensor_set(a_q5k, temp, 0, 512 * 512 * sizeof(float));
    ggml_backend_tensor_set(b_q8_1, temp, 0, 512 * 512 * sizeof(float));
    
    // Create graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    
    // Compute
    printf("Running Q5_K matrix multiplication on GPU...\n");
    ggml_backend_graph_compute(backend, graph);
    
    printf("✓ Q5_K GPU computation completed\n");
    
    // Test Q6_K
    printf("Creating Q6_K tensors for GPU test...\n");
    struct ggml_tensor* a_q6k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 512, 512);
    ggml_backend_tensor_set(a_q6k, temp, 0, 512 * 512 * sizeof(float));
    
    struct ggml_tensor* c2 = ggml_mul_mat(ctx, a_q6k, b_q8_1);
    struct ggml_cgraph* graph2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph2, c2);
    
    printf("Running Q6_K matrix multiplication on GPU...\n");
    ggml_backend_graph_compute(backend, graph2);
    
    printf("✓ Q6_K GPU computation completed\n");
    
    // Cleanup
    free(temp);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    
    return 0;
}

int main() {
    printf("GFX906 Q5_K/Q6_K Test\n");
    printf("=====================\n\n");
    
    srand(42);
    
    int result = 0;
    
    // CPU tests
    result |= test_q5_k();
    result |= test_q6_k();
    
    // GPU tests
    result |= test_q5k_q6k_gpu();
    
    printf("\n=====================\n");
    if (result == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("Some tests FAILED ✗\n");
    }
    
    return result;
}