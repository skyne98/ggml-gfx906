#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_warp_functions.h>
#include <stdio.h>

// Test using HIP's swizzle wrapper
__global__ void test_hip_swizzle(float* output) {
    int tid = threadIdx.x;
    float value = tid;
    
    if (tid < 64) {
        // Original value
        output[tid] = value;
        
        // Use HIP's ds_swizzle with XOR pattern
        // Pattern for XOR butterfly: 0x1F is the "mode", offset is the XOR mask
        // For XOR mode with mask 1: pattern = (0x1F << 10) | 1 = 0x7C01
        float swizzled1 = __hip_ds_swizzlef(value, 0x7C01);
        output[64 + tid] = swizzled1;
        
        // For XOR mode with mask 32: pattern = (0x1F << 10) | 32 = 0x7C20
        float swizzled32 = __hip_ds_swizzlef(value, 0x7C20);
        output[128 + tid] = swizzled32;
    }
}

int main() {
    float *d_output, h_output[192];
    
    hipMalloc(&d_output, 192 * sizeof(float));
    
    test_hip_swizzle<<<1, 64>>>(d_output);
    hipDeviceSynchronize();
    
    hipMemcpy(h_output, d_output, 192 * sizeof(float), hipMemcpyDeviceToHost);
    
    printf("Testing HIP swizzle wrapper:\n");
    printf("============================\n");
    
    // Check XOR 1 (swap adjacent)
    bool xor1_correct = true;
    for (int i = 0; i < 64; i++) {
        float expected = (i % 2 == 0) ? i + 1 : i - 1;
        if (h_output[64 + i] != expected) {
            xor1_correct = false;
            break;
        }
    }
    printf("XOR 1 (swap adjacent): %s\n", xor1_correct ? "PASSED" : "FAILED");
    
    // Check XOR 32 (swap halves)
    bool xor32_correct = true;
    for (int i = 0; i < 32; i++) {
        if (h_output[128 + i] != i + 32) xor32_correct = false;
        if (h_output[128 + i + 32] != i) xor32_correct = false;
    }
    printf("XOR 32 (swap halves): %s\n", xor32_correct ? "PASSED" : "FAILED");
    
    // If failed, show what we got
    if (!xor32_correct) {
        printf("\nExpected for XOR 32: lanes 0-31 should have 32-63, lanes 32-63 should have 0-31\n");
        printf("Got:\n");
        for (int i = 0; i < 64; i++) {
            if (i % 8 == 0) printf("\n");
            printf("%4.0f ", h_output[128 + i]);
        }
        printf("\n");
    }
    
    hipFree(d_output);
    return 0;
}