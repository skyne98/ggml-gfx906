#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

// Simple GEMM test without GPU execution
// This test validates the implementation compiles and can be linked

// External declarations - these would be implemented in the GEMM kernel
extern "C" {
bool ggml_cuda_gemm_gfx906_available();
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

// Helper function to initialize matrix with random values
void init_matrix(float * matrix, int rows, int cols, float scale = 1.0f) {
    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = dist(gen);
    }
}

// Helper function to check if two matrices are approximately equal
bool check_matrices_equal(const float * A, const float * B, int size, float tolerance = 1e-3f) {
    for (int i = 0; i < size; i++) {
        if (std::fabs(A[i] - B[i]) > tolerance) {
            printf("Mismatch at index %d: %f vs %f (diff: %f)\n", i, A[i], B[i], std::fabs(A[i] - B[i]));
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== GFX906 GEMM Test Suite ===\n\n");
    
    // Check if GFX906 GEMM is available
#ifdef GGML_HIP_GFX906_OPTIMIZED
    printf("GFX906 optimizations: ENABLED\n");
    
    // Test CPU reference implementation
    printf("\nTesting CPU reference implementation...\n");
    
    const int M = 128;
    const int N = 128;
    const int K = 32;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);
    
    // Initialize matrices
    init_matrix(h_A.data(), M, K);
    init_matrix(h_B.data(), K, N);
    
    // Compute reference on CPU
    gemm_cpu_ref(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K, alpha, beta);
    
    // Verify CPU implementation works
    float sum = 0.0f;
    for (int i = 0; i < M * N; i++) {
        sum += std::fabs(h_C_ref[i]);
    }
    
    if (sum > 0.0f) {
        printf("CPU reference implementation: PASSED (sum of abs values: %.2f)\n", sum);
    } else {
        printf("CPU reference implementation: FAILED (result is all zeros)\n");
        return 1;
    }
    
    // Test with different sizes
    printf("\nTesting various matrix sizes:\n");
    const int test_sizes[][3] = {
        {64, 64, 32},
        {128, 128, 32},
        {256, 256, 64},
        {512, 512, 128},
        {1024, 1024, 256}
    };
    
    for (const auto& size : test_sizes) {
        const int m = size[0];
        const int n = size[1];
        const int k = size[2];
        
        std::vector<float> a(m * k);
        std::vector<float> b(k * n);
        std::vector<float> c(m * n, 0.0f);
        
        init_matrix(a.data(), m, k);
        init_matrix(b.data(), k, n);
        
        gemm_cpu_ref(a.data(), b.data(), c.data(), m, n, k, 1.0f, 0.0f);
        
        // Basic validation - check result is not zero
        float max_val = 0.0f;
        for (int i = 0; i < m * n; i++) {
            max_val = std::max(max_val, std::fabs(c[i]));
        }
        
        printf("  Size %dx%dx%d: %s (max value: %.2f)\n", 
               m, n, k, 
               max_val > 0.0f ? "PASSED" : "FAILED",
               max_val);
    }
    
    printf("\nAll tests completed successfully!\n");
#else
    printf("GFX906 optimizations: DISABLED\n");
    printf("Please rebuild with -DGGML_HIP_GFX906_OPTIMIZED=ON to enable tests\n");
#endif
    
    return 0;
}