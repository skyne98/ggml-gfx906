#include <hip/hip_runtime.h>
#include <stdio.h>

// Test inline GCN assembly for DS_SWIZZLE_B32
__global__ void test_inline_asm_swizzle(float* output) {
    int tid = threadIdx.x;
    float value = tid;
    
    if (tid < 64) {
        // Original value
        output[tid] = value;
        
        // Test inline assembly DS_SWIZZLE with XOR 1
        float result1;
        asm volatile(
            "ds_swizzle_b32 %0, %1 offset:0x0041\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(result1)
            : "v"(value)
        );
        output[64 + tid] = result1;
        
        // Test XOR 16
        float result16;
        asm volatile(
            "ds_swizzle_b32 %0, %1 offset:0x0050\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(result16)
            : "v"(value)
        );
        output[128 + tid] = result16;
        
        // Test REVERSEX32
        float reverse;
        asm volatile(
            "ds_swizzle_b32 %0, %1 offset:0x001F\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(reverse)
            : "v"(value)
        );
        output[192 + tid] = reverse;
        
        // Test DS_BPERMUTE_B32 (broadcast from lane 5)
        float bperm;
        int src_addr = 5 * 4;  // Lane 5, byte address
        asm volatile(
            "ds_bpermute_b32 %0, %1, %2\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(bperm)
            : "v"(src_addr), "v"(value)
        );
        output[256 + tid] = bperm;
    }
}

int main() {
    float *d_output, h_output[320];
    
    hipMalloc(&d_output, 320 * sizeof(float));
    
    test_inline_asm_swizzle<<<1, 64>>>(d_output);
    hipDeviceSynchronize();
    
    hipMemcpy(h_output, d_output, 320 * sizeof(float), hipMemcpyDeviceToHost);
    
    printf("Testing Inline GCN Assembly DS_SWIZZLE_B32:\n");
    printf("===========================================\n\n");
    
    // Check XOR 1
    printf("XOR 1 (swap adjacent):\n");
    printf("First 8 values: ");
    for (int i = 0; i < 8; i++) {
        printf("%.0f ", h_output[64 + i]);
    }
    printf("\nExpected:       1 0 3 2 5 4 7 6\n");
    
    // Check XOR 16
    printf("\nXOR 16 (swap 16-thread groups):\n");
    printf("Lane 0-3:   %.0f %.0f %.0f %.0f\n",
           h_output[128 + 0], h_output[128 + 1], h_output[128 + 2], h_output[128 + 3]);
    printf("Lane 16-19: %.0f %.0f %.0f %.0f\n",
           h_output[128 + 16], h_output[128 + 17], h_output[128 + 18], h_output[128 + 19]);
    printf("Expected lane 0-3:   16 17 18 19\n");
    printf("Expected lane 16-19: 0 1 2 3\n");
    
    // Check REVERSEX32
    printf("\nREVERSEX32 (reverse within 32-thread groups):\n");
    printf("Lane 0-3:   %.0f %.0f %.0f %.0f\n",
           h_output[192 + 0], h_output[192 + 1], h_output[192 + 2], h_output[192 + 3]);
    printf("Lane 28-31: %.0f %.0f %.0f %.0f\n",
           h_output[192 + 28], h_output[192 + 29], h_output[192 + 30], h_output[192 + 31]);
    printf("Expected lane 0-3:   31 30 29 28\n");
    printf("Expected lane 28-31: 3 2 1 0\n");
    
    // Check DS_BPERMUTE (broadcast from lane 5)
    printf("\nDS_BPERMUTE (broadcast from lane 5):\n");
    printf("First 8 values: ");
    for (int i = 0; i < 8; i++) {
        printf("%.0f ", h_output[256 + i]);
    }
    printf("\nExpected:       5 5 5 5 5 5 5 5\n");
    
    hipFree(d_output);
    return 0;
}