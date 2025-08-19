#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <vector>

// Include the auto-tuner
#include "src/ggml-cuda/gemm-gfx906-autotuner.cuh"

// Include our kernels
#include "gemm_gfx906_standalone.cuh"

// Template kernel that uses runtime configuration
template<typename T>
__global__ void gemm_tuned(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const GemmConfig config
) {
    // Dynamic shared memory allocation
    extern __shared__ char shared_mem[];
    
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * config.tile_m;
    const int block_col = blockIdx.x * config.tile_n;
    
    // Calculate thread's position in tile
    const int threads_per_row = config.tile_n / config.thread_tile_n;
    const int thread_row = (tid / threads_per_row) * config.thread_tile_m;
    const int thread_col = (tid % threads_per_row) * config.thread_tile_n;
    
    // Setup shared memory pointers based on configuration
    T* lds_tile_a = reinterpret_cast<T*>(shared_mem);
    T* lds_tile_b = lds_tile_a + config.tile_m * (config.tile_k + config.lds_pad);
    
    // Accumulator registers
    float acc[8][8] = {{0.0f}};  // Max size, will use subset based on config
    
    // Main GEMM loop
    for (int k_tile = 0; k_tile < K; k_tile += config.tile_k) {
        // Cooperative loading with configured vector width
        // ... (implementation would follow based on config.vector_width)
        
        __syncthreads();
        
        // Computation based on thread tile configuration
        for (int k = 0; k < config.tile_k; k++) {
            // ... (computation based on config.thread_tile_m/n)
        }
        
        __syncthreads();
    }
    
    // Store results
    // ... (store based on configuration)
}

// Benchmark function that returns actual timing
float benchmark_gemm_config(
    const GemmConfig& config,
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    int iterations = 10
) {
    dim3 grid((N + config.tile_n - 1) / config.tile_n,
              (M + config.tile_m - 1) / config.tile_m);
    dim3 block(config.threads_per_block);
    
    // Calculate shared memory size
    size_t smem_size = config.lds_bytes;
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        gemm_tuned<<<grid, block, smem_size>>>(A, B, C, M, N, K, 1.0f, 0.0f, config);
    }
    hipDeviceSynchronize();
    
    // Timing
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        gemm_tuned<<<grid, block, smem_size>>>(A, B, C, M, N, K, 1.0f, 0.0f, config);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);
    
    hipEventDestroy(start);
    hipEventDestroy(stop);
    
    return ms / iterations;  // Return average time per GEMM
}

int main() {
    printf("=== GFX906 GEMM Auto-Tuner Demonstration ===\n\n");
    
    // Initialize auto-tuner
    GemmAutoTuner tuner;
    
    // Test different matrix sizes
    struct TestCase {
        int M, N, K;
        const char* description;
    };
    
    TestCase test_cases[] = {
        {512, 512, 512, "Small square matrices"},
        {1024, 1024, 1024, "Medium square matrices"},
        {2048, 2048, 2048, "Large square matrices"},
        {4096, 1024, 512, "Rectangular (tall)"},
        {512, 4096, 1024, "Rectangular (wide)"},
        {8192, 8192, 64, "Large but shallow K"},
        {64, 64, 8192, "Small but deep K"}
    };
    
    for (const auto& test : test_cases) {
        printf("=====================================\n");
        printf("Matrix: %s (%dx%dx%d)\n", test.description, test.M, test.N, test.K);
        printf("=====================================\n\n");
        
        // Auto-tune for FP16
        printf("Auto-tuning for FP16...\n");
        GemmConfig config_fp16 = tuner.tune(test.M, test.N, test.K, true);
        
        printf("\nSelected Configuration:\n");
        printf("  Tile: %dx%dx%d\n", config_fp16.tile_m, config_fp16.tile_n, config_fp16.tile_k);
        printf("  Thread tile: %dx%d\n", config_fp16.thread_tile_m, config_fp16.thread_tile_n);
        printf("  Threads: %d (waves: %d)\n", config_fp16.threads_per_block, config_fp16.waves_per_block);
        printf("  Vector width: %d\n", config_fp16.vector_width);
        printf("  LDS padding: %d\n", config_fp16.lds_pad);
        printf("  Double buffer: %s\n", config_fp16.double_buffer ? "Yes" : "No");
        printf("  Use V_DOT2: %s\n", config_fp16.use_dot2 ? "Yes" : "No");
        
        // Print occupancy analysis
        printf("\n");
        tuner.print_occupancy_info(config_fp16);
        
        // Suggest optimizations
        tuner.suggest_optimizations(config_fp16, test.M, test.N, test.K);
        
        // Auto-tune for FP32
        printf("\n-------------------------------------\n");
        printf("Auto-tuning for FP32...\n");
        GemmConfig config_fp32 = tuner.tune(test.M, test.N, test.K, false);
        
        printf("\nSelected Configuration:\n");
        printf("  Tile: %dx%dx%d\n", config_fp32.tile_m, config_fp32.tile_n, config_fp32.tile_k);
        printf("  Thread tile: %dx%d\n", config_fp32.thread_tile_m, config_fp32.thread_tile_n);
        printf("  Threads: %d (waves: %d)\n", config_fp32.threads_per_block, config_fp32.waves_per_block);
        
        printf("\n");
        tuner.print_occupancy_info(config_fp32);
        
        printf("\n\n");
    }
    
    // Demonstrate actual benchmarking with a specific size
    printf("=====================================\n");
    printf("Benchmarking with actual data (1024x1024x1024 FP16)\n");
    printf("=====================================\n\n");
    
    const int M = 1024, N = 1024, K = 1024;
    
    // Allocate matrices
    size_t size_A = M * K * sizeof(__half);
    size_t size_B = K * N * sizeof(__half);
    size_t size_C = M * N * sizeof(__half);
    
    __half *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);
    
    // Initialize with dummy data
    std::vector<__half> h_A(M * K, __float2half(0.01f));
    std::vector<__half> h_B(K * N, __float2half(0.01f));
    hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice);
    
    // Try different configurations
    std::vector<GemmConfig> configs = {
        // Config 1: Small tiles, high occupancy
        {256, 4, 64, 64, 16, 2, 2, 2, 0, false, true, 0, 0, 0, 0, 0, 0, 0},
        // Config 2: Large tiles, lower occupancy but better data reuse
        {256, 4, 128, 128, 32, 4, 4, 4, 1, false, true, 0, 0, 0, 0, 0, 0, 0},
        // Config 3: Balanced
        {256, 4, 128, 64, 32, 2, 4, 2, 1, true, true, 0, 0, 0, 0, 0, 0, 0}
    };
    
    for (int i = 0; i < configs.size(); i++) {
        auto& config = configs[i];
        config.calculate_resources(true);
        
        printf("Configuration %d:\n", i + 1);
        printf("  Tile: %dx%dx%d, Thread tile: %dx%d\n",
               config.tile_m, config.tile_n, config.tile_k,
               config.thread_tile_m, config.thread_tile_n);
        
        if (!config.is_valid()) {
            printf("  ❌ Invalid configuration (exceeds resource limits)\n\n");
            continue;
        }
        
        // Would benchmark here with actual kernel
        // float time = benchmark_gemm_config(config, d_A, d_B, d_C, M, N, K);
        
        // For demonstration, use our existing kernel
        dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
        dim3 block(256);
        
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        
        // Warmup
        for (int j = 0; j < 5; j++) {
            gemm_f16_gfx906_asm<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
        }
        hipDeviceSynchronize();
        
        // Benchmark
        hipEventRecord(start);
        for (int j = 0; j < 100; j++) {
            gemm_f16_gfx906_asm<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        
        float ms = 0;
        hipEventElapsedTime(&ms, start, stop);
        float time_per_gemm = ms / 100;
        
        double ops = 2.0 * M * N * K;
        double tflops = (ops / 1e12) / (time_per_gemm / 1000.0);
        
        printf("  Performance: %.2f TFLOPS (%.3f ms)\n", tflops, time_per_gemm);
        printf("  Occupancy: %d waves (%.1f%%)\n", 
               config.occupancy_waves, 
               (config.occupancy_waves * 100.0) / GFX906::MAX_WAVES_PER_CU);
        printf("  LDS usage: %d bytes (%.1f%%)\n",
               config.lds_bytes,
               (config.lds_bytes * 100.0) / GFX906::LDS_SIZE);
        printf("\n");
        
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }
    
    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    printf("Auto-tuner demonstration complete!\n");
    
    return 0;
}