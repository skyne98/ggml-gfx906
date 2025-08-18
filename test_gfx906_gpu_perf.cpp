#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <random>

// Forward declaration of our GEMM function (C++ linkage)
void ggml_cuda_gemm_gfx906_f32(const float * A,
                               const float * B,
                               float *       C,
                               const int     M,
                               const int     N,
                               const int     K,
                               const float   alpha,
                               const float   beta,
                               hipStream_t   stream);

// Helper to check HIP errors
#define HIP_CHECK(call) do { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Initialize matrix with random values
void init_matrix(float* matrix, int size) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < size; i++) {
        matrix[i] = dist(gen);
    }
}

// Benchmark function
void benchmark_gemm(int M, int N, int K, int iterations = 100) {
    printf("\n=== Benchmarking %dx%dx%d ===\n", M, N, K);
    
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
        ggml_cuda_gemm_gfx906_f32(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, nullptr);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    printf("Running %d iterations...\n", iterations);
    HIP_CHECK(hipEventRecord(start));
    
    for (int i = 0; i < iterations; i++) {
        ggml_cuda_gemm_gfx906_f32(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, nullptr);
    }
    
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    
    // Calculate TFLOPS
    double ops_per_gemm = 2.0 * M * N * K;  // 2 ops per multiply-add
    double total_ops = ops_per_gemm * iterations;
    double time_seconds = milliseconds / 1000.0;
    double tflops = (total_ops / 1e12) / time_seconds;
    
    printf("Results:\n");
    printf("  Total time: %.3f ms\n", milliseconds);
    printf("  Time per GEMM: %.3f ms\n", milliseconds / iterations);
    printf("  Performance: %.2f TFLOPS\n", tflops);
    
    // Calculate efficiency
    double peak_tflops = 13.1;  // GFX906 FP32 peak
    double efficiency = (tflops / peak_tflops) * 100.0;
    printf("  Efficiency: %.1f%% of peak\n", efficiency);
    
    // Memory bandwidth
    double gb_transferred = (size_A + size_B + size_C) * iterations / 1e9;
    double bandwidth = gb_transferred / time_seconds;
    printf("  Memory bandwidth: %.1f GB/s\n", bandwidth);
    
    // Cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

// Test with hipBLAS for comparison
void benchmark_hipblas(int M, int N, int K, int iterations = 100) {
    printf("\n=== hipBLAS Reference %dx%dx%d ===\n", M, N, K);
    
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
    
    // Create hipBLAS handle
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    
    // Create events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 10; i++) {
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    printf("Running %d iterations...\n", iterations);
    HIP_CHECK(hipEventRecord(start));
    
    for (int i = 0; i < iterations; i++) {
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
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
    
    // Calculate efficiency
    double peak_tflops = 13.1;
    double efficiency = (tflops / peak_tflops) * 100.0;
    printf("  Efficiency: %.1f%% of peak\n", efficiency);
    
    // Cleanup
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
    
    printf("=== GFX906 GEMM Performance Benchmark ===\n");
    printf("Device: %s\n", props.name);
    printf("Compute Units: %d\n", props.multiProcessorCount);
    printf("Max Threads per Block: %d\n", props.maxThreadsPerBlock);
    printf("Shared Memory per Block: %zu KB\n", props.sharedMemPerBlock / 1024);
    printf("Memory Clock: %d MHz\n", props.memoryClockRate / 1000);
    printf("Memory Bus Width: %d bits\n", props.memoryBusWidth);
    
    double peak_bandwidth = (props.memoryClockRate / 1000.0) * (props.memoryBusWidth / 8.0) * 2.0 / 1000.0;
    printf("Theoretical Memory Bandwidth: %.1f GB/s\n", peak_bandwidth);
    printf("\n");
    
    // Test different matrix sizes
    struct TestSize {
        int M, N, K;
        int iterations;
    };
    
    TestSize test_sizes[] = {
        {512, 512, 512, 1000},
        {1024, 1024, 1024, 500},
        {2048, 2048, 2048, 100},
        {4096, 4096, 4096, 20},
        {8192, 8192, 8192, 5}
    };
    
    for (const auto& size : test_sizes) {
        // Test our implementation
        benchmark_gemm(size.M, size.N, size.K, size.iterations);
        
        // Compare with hipBLAS
        benchmark_hipblas(size.M, size.N, size.K, size.iterations);
        
        printf("\n");
    }
    
    printf("Benchmark complete!\n");
    return 0;
}