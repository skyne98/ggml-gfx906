#include <hip/hip_runtime.h>
#include <stdio.h>

// Test basic __shfl_xor
__global__ void test_shfl_xor(float* output) {
    int tid = threadIdx.x;
    float value = tid;
    
    if (tid < 64) {
        // Original value
        output[tid] = value;
        
        // Test __shfl_xor with 1
        float xor1 = __shfl_xor(value, 1, 64);
        output[64 + tid] = xor1;
        
        // Test __shfl_xor with 32
        float xor32 = __shfl_xor(value, 32, 64);
        output[128 + tid] = xor32;
    }
}

int main() {
    float *d_output, h_output[192];
    
    hipMalloc(&d_output, 192 * sizeof(float));
    
    test_shfl_xor<<<1, 64>>>(d_output);
    hipDeviceSynchronize();
    
    hipMemcpy(h_output, d_output, 192 * sizeof(float), hipMemcpyDeviceToHost);
    
    printf("Testing __shfl_xor:\n");
    printf("===================\n");
    
    // Check XOR 1
    bool xor1_correct = true;
    for (int i = 0; i < 64; i++) {
        float expected = (i % 2 == 0) ? i + 1 : i - 1;
        if (h_output[64 + i] != expected) {
            xor1_correct = false;
            printf("XOR 1 failed at lane %d: expected %.0f, got %.0f\n", 
                   i, expected, h_output[64 + i]);
            break;
        }
    }
    printf("XOR 1 (swap adjacent): %s\n", xor1_correct ? "PASSED" : "FAILED");
    
    // Check XOR 32
    bool xor32_correct = true;
    for (int i = 0; i < 32; i++) {
        if (h_output[128 + i] != i + 32) {
            xor32_correct = false;
            printf("XOR 32 failed at lane %d: expected %.0f, got %.0f\n", 
                   i, (float)(i + 32), h_output[128 + i]);
            break;
        }
        if (h_output[128 + i + 32] != i) {
            xor32_correct = false;
            printf("XOR 32 failed at lane %d: expected %.0f, got %.0f\n", 
                   i + 32, (float)i, h_output[128 + i + 32]);
            break;
        }
    }
    printf("XOR 32 (swap halves): %s\n", xor32_correct ? "PASSED" : "FAILED");
    
    hipFree(d_output);
    return 0;
}