#include <hip/hip_runtime.h>
#include <stdio.h>

// Test DS_SWIZZLE_B32 with correct Vega ISA encoding
// Based on Vega 7nm ISA documentation section 12.13.1
__global__ void test_vega_swizzle(float* output) {
    int tid = threadIdx.x;
    float value = tid;
    
    if (tid < 64) {
        // Original value
        output[tid] = value;
        
        // XOR mode: offset[15] = 0 (32-thread swizzle mode)
        // Pattern: [14:10] = AND mask, [9:5] = OR mask, [4:0] = XOR mask
        // For XOR with 1: AND=0x1F, OR=0x00, XOR=0x01
        // Pattern = (0 << 15) | (0x1F << 10) | (0x00 << 5) | 0x01 = 0x7C01
        float swizzled1 = __builtin_amdgcn_ds_swizzle(value, 0x7C01);
        output[64 + tid] = swizzled1;
        
        // For XOR with 32: AND=0x1F, OR=0x00, XOR=0x20
        // Pattern = (0 << 15) | (0x1F << 10) | (0x00 << 5) | 0x20 = 0x7C20
        float swizzled32 = __builtin_amdgcn_ds_swizzle(value, 0x7C20);
        output[128 + tid] = swizzled32;
        
        // Try XOR with 16
        float swizzled16 = __builtin_amdgcn_ds_swizzle(value, 0x7C10);
        output[192 + tid] = swizzled16;
    }
}

int main() {
    float *d_output, h_output[256];
    
    hipMalloc(&d_output, 256 * sizeof(float));
    
    test_vega_swizzle<<<1, 64>>>(d_output);
    hipDeviceSynchronize();
    
    hipMemcpy(h_output, d_output, 256 * sizeof(float), hipMemcpyDeviceToHost);
    
    printf("Testing Vega DS_SWIZZLE_B32 with correct ISA encoding:\n");
    printf("=======================================================\n");
    
    // Check XOR 1 (swap adjacent)
    bool xor1_correct = true;
    for (int i = 0; i < 32; i++) {  // Note: 32-thread groups!
        float expected = (i % 2 == 0) ? i + 1 : i - 1;
        if (h_output[64 + i] != expected) {
            xor1_correct = false;
            printf("XOR 1 failed at lane %d: expected %.0f, got %.0f\n", 
                   i, expected, h_output[64 + i]);
            break;
        }
    }
    // Check second 32-thread group
    for (int i = 32; i < 64; i++) {
        float expected = (i % 2 == 0) ? i + 1 : i - 1;
        if (h_output[64 + i] != expected) {
            xor1_correct = false;
            printf("XOR 1 failed at lane %d: expected %.0f, got %.0f\n", 
                   i, expected, h_output[64 + i]);
            break;
        }
    }
    printf("XOR 1 (swap adjacent): %s\n", xor1_correct ? "PASSED" : "FAILED");
    
    // Check XOR 16 (swap between 16-thread groups within 32)
    bool xor16_correct = true;
    for (int group = 0; group < 2; group++) {  // Two 32-thread groups
        int base = group * 32;
        for (int i = 0; i < 16; i++) {
            float expected = base + i + 16;  // First 16 should get second 16
            if (h_output[192 + base + i] != expected) {
                xor16_correct = false;
                printf("XOR 16 failed at lane %d: expected %.0f, got %.0f\n", 
                       base + i, expected, h_output[192 + base + i]);
                break;
            }
        }
        for (int i = 16; i < 32; i++) {
            float expected = base + i - 16;  // Second 16 should get first 16
            if (h_output[192 + base + i] != expected) {
                xor16_correct = false;
                printf("XOR 16 failed at lane %d: expected %.0f, got %.0f\n", 
                       base + i, expected, h_output[192 + base + i]);
                break;
            }
        }
    }
    printf("XOR 16 (swap 16-thread groups): %s\n", xor16_correct ? "PASSED" : "FAILED");
    
    // Note: XOR 32 won't work as expected because swizzle works in 32-thread groups
    // So lanes 0-31 can't access lanes 32-63 with this mode
    printf("\nNote: XOR 32 cannot swap between wave halves (0-31 with 32-63)\n");
    printf("      because DS_SWIZZLE_B32 operates on 32-thread groups.\n");
    
    hipFree(d_output);
    return 0;
}