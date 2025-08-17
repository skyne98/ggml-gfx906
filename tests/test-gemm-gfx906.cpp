#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifdef GGML_USE_HIP
#    include <hip/hip_fp16.h>
#    include <hip/hip_runtime.h>
#    define cudaStream_t           hipStream_t
#    define cudaMalloc             hipMalloc
#    define cudaFree               hipFree
#    define cudaMemcpy             hipMemcpy
#    define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#    define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#    define cudaDeviceSynchronize  hipDeviceSynchronize
#    define cudaStreamCreate       hipStreamCreate
#    define cudaStreamDestroy      hipStreamDestroy
#    define cudaEvent_t            hipEvent_t
#    define cudaEventCreate        hipEventCreate
#    define cudaEventRecord        hipEventRecord
#    define cudaEventSynchronize   hipEventSynchronize
#    define cudaEventElapsedTime   hipEventElapsedTime
#    define cudaEventDestroy       hipEventDestroy
#else
#    include <cuda_fp16.h>
#    include <cuda_runtime.h>
#endif

// Forward declarations of the GEMM functions
extern "C" {
void ggml_cuda_gemm_gfx906_f32(const float * A,
                               const float * B,
                               float *       C,
                               const int     M,
                               const int     N,
                               const int     K,
                               const float   alpha,
                               const float   beta,
                               cudaStream_t  stream);

void ggml_cuda_gemm_gfx906_f16(const half * A,
                               const half * B,
                               half *       C,
                               const int    M,
                               const int    N,
                               const int    K,
                               const float  alpha,
                               const float  beta,
                               cudaStream_t stream);
}

// Helper function to initialize matrix with random values
template <typename T> void init_matrix(T * matrix, int rows, int cols, float scale = 1.0f) {
    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> dist(-scale, scale);

    for (int i = 0; i < rows * cols; i++) {
        if constexpr (std::is_same_v<T, half>) {
            matrix[i] = __float2half(dist(gen));
        } else {
            matrix[i] = dist(gen);
        }
    }
}

// CPU reference implementation for validation
void gemm_cpu_ref(const float * A, const float * B, float * C, int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// Validation function
bool validate_result(const float * gpu_result, const float * cpu_result, int size, float tolerance = 1e-3f) {
    float max_error   = 0.0f;
    int   error_count = 0;

    for (int i = 0; i < size; i++) {
        float error = std::abs(gpu_result[i] - cpu_result[i]);
        max_error   = std::max(max_error, error);
        if (error > tolerance) {
            error_count++;
            if (error_count <= 10) {  // Print first 10 errors
                printf(
                    "Error at index %d: GPU = %.6f, CPU = %.6f, diff = %.6f\n", i, gpu_result[i], cpu_result[i], error);
            }
        }
    }

    printf("Max error: %.6f, Error count: %d/%d (%.2f%%)\n", max_error, error_count, size, 100.0f * error_count / size);

    return error_count == 0;
}

// Test FP32 GEMM
void test_gemm_f32(int M, int N, int K) {
    printf("\n=== Testing FP32 GEMM (M=%d, N=%d, K=%d) ===\n", M, N, K);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    float * h_A     = (float *) malloc(size_A);
    float * h_B     = (float *) malloc(size_B);
    float * h_C     = (float *) malloc(size_C);
    float * h_C_ref = (float *) malloc(size_C);

    // Initialize matrices
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    memset(h_C, 0, size_C);
    memset(h_C_ref, 0, size_C);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < 5; i++) {
        ggml_cuda_gemm_gfx906_f32(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, stream);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_iterations = 100;
    cudaEventRecord(start, stream);

    for (int i = 0; i < num_iterations; i++) {
        ggml_cuda_gemm_gfx906_f32(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, stream);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate performance metrics
    float  avg_time_ms = milliseconds / num_iterations;
    double flops       = 2.0 * M * N * K;
    double tflops      = (flops * num_iterations) / (milliseconds * 1e9);

    printf("Average time: %.3f ms\n", avg_time_ms);
    printf("Performance: %.2f TFLOPS\n", tflops);

    // Copy result back
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Compute CPU reference
    gemm_cpu_ref(h_A, h_B, h_C_ref, M, N, K, 1.0f, 0.0f);

    // Validate
    bool passed = validate_result(h_C, h_C_ref, M * N);
    printf("Validation: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
}

// Test FP16 GEMM
void test_gemm_f16(int M, int N, int K) {
    printf("\n=== Testing FP16 GEMM (M=%d, N=%d, K=%d) ===\n", M, N, K);

    size_t size_A       = M * K * sizeof(half);
    size_t size_B       = K * N * sizeof(half);
    size_t size_C       = M * N * sizeof(half);
    size_t size_C_float = M * N * sizeof(float);

    // Allocate host memory
    half *  h_A       = (half *) malloc(size_A);
    half *  h_B       = (half *) malloc(size_B);
    half *  h_C       = (half *) malloc(size_C);
    float * h_C_float = (float *) malloc(size_C_float);
    float * h_A_float = (float *) malloc(M * K * sizeof(float));
    float * h_B_float = (float *) malloc(K * N * sizeof(float));
    float * h_C_ref   = (float *) malloc(size_C_float);

    // Initialize matrices
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    memset(h_C, 0, size_C);
    memset(h_C_ref, 0, size_C_float);

    // Convert to float for CPU reference
    for (int i = 0; i < M * K; i++) {
        h_A_float[i] = __half2float(h_A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        h_B_float[i] = __half2float(h_B[i]);
    }

    // Allocate device memory
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < 5; i++) {
        ggml_cuda_gemm_gfx906_f16(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, stream);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_iterations = 100;
    cudaEventRecord(start, stream);

    for (int i = 0; i < num_iterations; i++) {
        ggml_cuda_gemm_gfx906_f16(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, stream);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate performance metrics
    float  avg_time_ms = milliseconds / num_iterations;
    double flops       = 2.0 * M * N * K;
    double tflops      = (flops * num_iterations) / (milliseconds * 1e9);

    printf("Average time: %.3f ms\n", avg_time_ms);
    printf("Performance: %.2f TFLOPS\n", tflops);

    // Copy result back
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Convert result to float for validation
    for (int i = 0; i < M * N; i++) {
        h_C_float[i] = __half2float(h_C[i]);
    }

    // Compute CPU reference
    gemm_cpu_ref(h_A_float, h_B_float, h_C_ref, M, N, K, 1.0f, 0.0f);

    // Validate (with higher tolerance for FP16)
    bool passed = validate_result(h_C_float, h_C_ref, M * N, 1e-2f);
    printf("Validation: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_float);
    free(h_A_float);
    free(h_B_float);
    free(h_C_ref);
}

int main(int argc, char ** argv) {
    printf("=== GFX906 GEMM Performance Test ===\n");

    // Test different matrix sizes
    struct TestCase {
        int M, N, K;
    };

    TestCase test_cases[] = {
        { 512,  512,  512  }, // Small
        { 1024, 1024, 1024 }, // Medium
        { 2048, 2048, 2048 }, // Large
        { 4096, 4096, 1024 }, // Wide
        { 1024, 4096, 4096 }, // Tall
    };

    for (const auto & tc : test_cases) {
        test_gemm_f32(tc.M, tc.N, tc.K);
        test_gemm_f16(tc.M, tc.N, tc.K);
    }

    printf("\n=== All tests completed ===\n");

    return 0;
}
