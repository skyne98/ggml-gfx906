#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <chrono>

// Our GEMM function
void ggml_cuda_gemm_gfx906_f32(const float * A, const float * B, float * C,
                               const int M, const int N, const int K,
                               const float alpha, const float beta, hipStream_t stream);

#define HIP_CHECK(call) do { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error: %s\n", hipGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("Quick GFX906 GEMM Performance Test\n");
    
    const int M = 1024, N = 1024, K = 1024;
    const int iterations = 10;
    
    // Allocate memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    
    // Initialize with dummy data
    HIP_CHECK(hipMemset(d_A, 1, M * K * sizeof(float)));
    HIP_CHECK(hipMemset(d_B, 1, K * N * sizeof(float)));
    HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
    
    // Events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    printf("Testing %dx%dx%d matrix with %d iterations\n", M, N, K, iterations);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        ggml_cuda_gemm_gfx906_f32(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, nullptr);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        ggml_cuda_gemm_gfx906_f32(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, nullptr);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    
    // Calculate TFLOPS
    double ops = 2.0 * M * N * K * iterations;
    double tflops = (ops / 1e12) / (ms / 1000.0);
    
    printf("Results:\n");
    printf("  Time: %.3f ms total, %.3f ms per GEMM\n", ms, ms/iterations);
    printf("  Performance: %.2f TFLOPS\n", tflops);
    printf("  Efficiency: %.1f%% of 13.1 TFLOPS peak\n", (tflops/13.1)*100.0);
    
    // Test hipBLAS for comparison
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup hipBLAS
    for (int i = 0; i < 3; i++) {
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark hipBLAS
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    tflops = (ops / 1e12) / (ms / 1000.0);
    
    printf("\nhipBLAS comparison:\n");
    printf("  Time: %.3f ms total, %.3f ms per GEMM\n", ms, ms/iterations);
    printf("  Performance: %.2f TFLOPS\n", tflops);
    printf("  Efficiency: %.1f%% of 13.1 TFLOPS peak\n", (tflops/13.1)*100.0);
    
    // Cleanup
    hipblasDestroy(handle);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    
    return 0;
}