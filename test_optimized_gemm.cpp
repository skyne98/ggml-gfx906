#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <random>

// Include the optimized kernel header
#include "src/ggml-cuda/gemm-gfx906-optimized.cuh"

#define HIP_CHECK(call) do { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Initialize matrix with random values
template<typename T>
void init_matrix(T* matrix, int size, float scale = 1.0f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (int i = 0; i < size; i++) {
        if constexpr (std::is_same_v<T, __half>) {
            matrix[i] = __float2half(dist(gen));
        } else {
            matrix[i] = dist(gen);
        }
    }
}

// Test FP16 optimized kernel
void test_fp16_optimized(int M, int N, int K, int iterations = 100) {
    printf("\n=== Testing Optimized FP16 GEMM %dx%dx%d ===\n", M, N, K);
    
    size_t size_A = M * K * sizeof(__half);
    size_t size_B = K * N * sizeof(__half);
    size_t size_C = M * N * sizeof(__half);
    
    // Allocate host memory
    std::vector<__half> h_A(M * K);
    std::vector<__half> h_B(K * N);
    std::vector<__half> h_C(M * N, __float2half(0.0f));
    
    // Initialize matrices
    init_matrix(h_A.data(), M * K);
    init_matrix(h_B.data(), K * N);
    
    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice));
    
    // Create events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 10; i++) {
        launch_gemm_f16_gfx906_optimized(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, nullptr);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    printf("Running %d iterations...\n", iterations);
    HIP_CHECK(hipEventRecord(start));
    
    for (int i = 0; i < iterations; i++) {
        launch_gemm_f16_gfx906_optimized(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, nullptr);
    }
    
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    
    // Calculate TFLOPS
    double ops_per_gemm = 2.0 * M * N * K;
    double total_ops = ops_per_gemm * iterations;
    double time_seconds = milliseconds / 1000.0;
    double tflops = (total_ops / 1e12) / time_seconds;
    
    printf("Results:\n");
    printf("  Total time: %.3f ms\n", milliseconds);
    printf("  Time per GEMM: %.3f ms\n", milliseconds / iterations);
    printf("  Performance: %.2f TFLOPS\n", tflops);
    
    // Calculate efficiency (FP16 peak is ~26.3 TFLOPS on GFX906)
    double peak_tflops = 26.3;
    double efficiency = (tflops / peak_tflops) * 100.0;
    printf("  Efficiency: %.1f%% of FP16 peak\n", efficiency);
    
    // Cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

// Test FP32 optimized kernel
void test_fp32_optimized(int M, int N, int K, int iterations = 100) {
    printf("\n=== Testing Optimized FP32 GEMM %dx%dx%d ===\n", M, N, K);
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    
    // Initialize matrices
    init_matrix(h_A.data(), M * K);
    init_matrix(h_B.data(), K * N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice));
    
    // Create events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 10; i++) {
        launch_gemm_f32_gfx906_optimized(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, nullptr);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    printf("Running %d iterations...\n", iterations);
    HIP_CHECK(hipEventRecord(start));
    
    for (int i = 0; i < iterations; i++) {
        launch_gemm_f32_gfx906_optimized(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, nullptr);
    }
    
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    
    // Calculate TFLOPS
    double ops_per_gemm = 2.0 * M * N * K;
    double total_ops = ops_per_gemm * iterations;
    double time_seconds = milliseconds / 1000.0;
    double tflops = (total_ops / 1e12) / time_seconds;
    
    printf("Results:\n");
    printf("  Total time: %.3f ms\n", milliseconds);
    printf("  Time per GEMM: %.3f ms\n", milliseconds / iterations);
    printf("  Performance: %.2f TFLOPS\n", tflops);
    
    // Calculate efficiency (FP32 peak is ~13.1 TFLOPS on GFX906)
    double peak_tflops = 13.1;
    double efficiency = (tflops / peak_tflops) * 100.0;
    printf("  Efficiency: %.1f%% of FP32 peak\n", efficiency);
    
    // Cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

// Compare with hipBLAS
void test_hipblas_reference(int M, int N, int K, int iterations = 100) {
    printf("\n=== hipBLAS Reference %dx%dx%d ===\n", M, N, K);
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    
    init_matrix(h_A.data(), M * K);
    init_matrix(h_B.data(), K * N);
    
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));
    
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice));
    
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipEventRecord(start));
    
    for (int i = 0; i < iterations; i++) {
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    
    double ops_per_gemm = 2.0 * M * N * K;
    double total_ops = ops_per_gemm * iterations;
    double time_seconds = milliseconds / 1000.0;
    double tflops = (total_ops / 1e12) / time_seconds;
    
    printf("Results:\n");
    printf("  Total time: %.3f ms\n", milliseconds);
    printf("  Time per GEMM: %.3f ms\n", milliseconds / iterations);
    printf("  Performance: %.2f TFLOPS\n", tflops);
    printf("  Efficiency: %.1f%% of FP32 peak\n", (tflops/13.1)*100.0);
    
    hipblasDestroy(handle);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

int main() {
    // Print device info
    int device;
    HIP_CHECK(hipGetDevice(&device));
    
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    
    printf("=== GFX906 Optimized GEMM Performance Test ===\n");
    printf("Device: %s\n", props.name);
    printf("Compute Units: %d\n", props.multiProcessorCount);
    printf("Max Threads per Block: %d\n", props.maxThreadsPerBlock);
    printf("Shared Memory per Block: %zu KB\n", props.sharedMemPerBlock / 1024);
    printf("Memory Clock: %d MHz\n", props.memoryClockRate / 1000);
    printf("Memory Bus Width: %d bits\n", props.memoryBusWidth);
    
    double peak_bandwidth = (props.memoryClockRate / 1000.0) * (props.memoryBusWidth / 8.0) * 2.0 / 1000.0;
    printf("Theoretical Memory Bandwidth: %.1f GB/s\n", peak_bandwidth);
    printf("\n");
    
    // Test sizes
    struct TestSize {
        int M, N, K;
        int iterations;
    };
    
    TestSize test_sizes[] = {
        {512, 512, 512, 500},
        {1024, 1024, 1024, 200},
        {2048, 2048, 2048, 50},
    };
    
    for (const auto& size : test_sizes) {
        // Test optimized FP32
        test_fp32_optimized(size.M, size.N, size.K, size.iterations);
        
        // Test optimized FP16
        test_fp16_optimized(size.M, size.N, size.K, size.iterations);
        
        // Compare with hipBLAS
        test_hipblas_reference(size.M, size.N, size.K, size.iterations);
        
        printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    }
    
    printf("All tests completed!\n");
    return 0;
}