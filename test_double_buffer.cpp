#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <vector>

#include "src/ggml-cuda/gemm_gfx906_optimized.cuh"
#include "src/ggml-cuda/gemm_gfx906_double_buffer.cuh"

#define HIP_CHECK(call) do { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error: %s\n", hipGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const int iterations = 100;
    
    printf("=== GFX906 Double-Buffer GEMM Test ===\n");
    printf("Matrix size: %dx%dx%d\n", M, N, K);
    printf("Iterations: %d\n\n", iterations);
    
    // Test double-buffered FP16
    {
        printf("Testing Double-Buffered FP16 kernel...\n");
        
        size_t size_A = M * K * sizeof(__half);
        size_t size_B = K * N * sizeof(__half);
        size_t size_C = M * N * sizeof(__half);
        
        __half *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, size_A));
        HIP_CHECK(hipMalloc(&d_B, size_B));
        HIP_CHECK(hipMalloc(&d_C, size_C));
        
        std::vector<__half> h_A(M * K, __float2half(0.01f));
        std::vector<__half> h_B(K * N, __float2half(0.01f));
        std::vector<__half> h_C(M * N, __float2half(0.0f));
        
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice));
        
        dim3 grid((N + DB_TILE_N - 1) / DB_TILE_N,
                  (M + DB_TILE_M - 1) / DB_TILE_M);
        dim3 block(DB_THREADS);
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            gemm_f16_gfx906_double_buffer<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            gemm_f16_gfx906_double_buffer<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        
        double ops = 2.0 * M * N * K * iterations;
        double tflops = (ops / 1e12) / (ms / 1000.0);
        
        printf("  Time: %.3f ms total, %.3f ms per GEMM\n", ms, ms/iterations);
        printf("  Performance: %.2f TFLOPS\n", tflops);
        printf("  Efficiency: %.1f%% of 26.3 TFLOPS FP16 peak\n\n", (tflops/26.3)*100.0);
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }
    
    // Test double-buffered FP32
    {
        printf("Testing Double-Buffered FP32 kernel...\n");
        
        size_t size_A = M * K * sizeof(float);
        size_t size_B = K * N * sizeof(float);
        size_t size_C = M * N * sizeof(float);
        
        float *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, size_A));
        HIP_CHECK(hipMalloc(&d_B, size_B));
        HIP_CHECK(hipMalloc(&d_C, size_C));
        
        std::vector<float> h_A(M * K, 0.01f);
        std::vector<float> h_B(K * N, 0.01f);
        std::vector<float> h_C(M * N, 0.0f);
        
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice));
        
        dim3 grid((N + DB_TILE_N - 1) / DB_TILE_N,
                  (M + DB_TILE_M - 1) / DB_TILE_M);
        dim3 block(DB_THREADS);
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            gemm_f32_gfx906_double_buffer<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            gemm_f32_gfx906_double_buffer<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        
        double ops = 2.0 * M * N * K * iterations;
        double tflops = (ops / 1e12) / (ms / 1000.0);
        
        printf("  Time: %.3f ms total, %.3f ms per GEMM\n", ms, ms/iterations);
        printf("  Performance: %.2f TFLOPS\n", tflops);
        printf("  Efficiency: %.1f%% of 13.1 TFLOPS FP32 peak\n\n", (tflops/13.1)*100.0);
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }
    
    // Compare with non-double-buffered optimized version
    {
        printf("Comparison with Non-Double-Buffered Optimized (FP16)...\n");
        
        size_t size_A = M * K * sizeof(__half);
        size_t size_B = K * N * sizeof(__half);
        size_t size_C = M * N * sizeof(__half);
        
        __half *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, size_A));
        HIP_CHECK(hipMalloc(&d_B, size_B));
        HIP_CHECK(hipMalloc(&d_C, size_C));
        
        std::vector<__half> h_A(M * K, __float2half(0.01f));
        std::vector<__half> h_B(K * N, __float2half(0.01f));
        
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
        
        dim3 grid((N + OPT_TILE_N - 1) / OPT_TILE_N,
                  (M + OPT_TILE_M - 1) / OPT_TILE_M);
        dim3 block(OPT_THREADS);
        
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            gemm_f16_gfx906_optimized<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        
        double ops = 2.0 * M * N * K * iterations;
        double tflops = (ops / 1e12) / (ms / 1000.0);
        
        printf("  Non-double-buffered: %.2f TFLOPS (%.3f ms per GEMM)\n", tflops, ms/iterations);
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }
    
    return 0;
}