#include <hip/hip_runtime.h>
#include <stdio.h>

// Test DS_SWIZZLE_B32 with special swizzle modes for Group of 32
// Based on Vega ISA documentation
__global__ void test_vega_special_modes(float* output) {
    int tid = threadIdx.x;
    float value = tid;
    
    if (tid < 64) {
        // Original value
        output[tid] = value;
        
        // Try predefined swizzle patterns for Group of 32 mode
        // These are special patterns mentioned in the ISA doc
        
        // SWAP modes - swap data within groups
        // SWAPX1: swap adjacent threads (XOR with 1)
        // Pattern for SWAPX1 in Group of 32 mode
        float swap1 = __builtin_amdgcn_ds_swizzle(value, 0x0041);  // SWAPX1
        output[64 + tid] = swap1;
        
        // SWAPX2: swap pairs (XOR with 2)
        float swap2 = __builtin_amdgcn_ds_swizzle(value, 0x0042);  // SWAPX2
        output[128 + tid] = swap2;
        
        // SWAPX4: swap quads (XOR with 4)
        float swap4 = __builtin_amdgcn_ds_swizzle(value, 0x0044);  // SWAPX4
        output[192 + tid] = swap4;
        
        // SWAPX8: swap groups of 8 (XOR with 8)
        float swap8 = __builtin_amdgcn_ds_swizzle(value, 0x0048);  // SWAPX8
        output[256 + tid] = swap8;
        
        // SWAPX16: swap groups of 16 (XOR with 16)
        float swap16 = __builtin_amdgcn_ds_swizzle(value, 0x0050);  // SWAPX16
        output[320 + tid] = swap16;
        
        // REVERSEX32: reverse within 32-thread groups
        float reverse = __builtin_amdgcn_ds_swizzle(value, 0x001F);  // REVERSEX32
        output[384 + tid] = reverse;
        
        // BCASTX2: broadcast from first thread of each pair
        float bcast2 = __builtin_amdgcn_ds_swizzle(value, 0x0061);  // BCASTX2
        output[448 + tid] = bcast2;
        
        // BCASTX4: broadcast from first thread of each quad
        float bcast4 = __builtin_amdgcn_ds_swizzle(value, 0x0063);  // BCASTX4
        output[512 + tid] = bcast4;
    }
}

int main() {
    float *d_output, *h_output;
    h_output = (float*)malloc(576 * sizeof(float));
    
    hipMalloc(&d_output, 576 * sizeof(float));
    
    test_vega_special_modes<<<1, 64>>>(d_output);
    hipDeviceSynchronize();
    
    hipMemcpy(h_output, d_output, 576 * sizeof(float), hipMemcpyDeviceToHost);
    
    printf("Testing Vega DS_SWIZZLE_B32 Special Modes:\n");
    printf("==========================================\n\n");
    
    // Test SWAPX1 (swap adjacent)
    printf("SWAPX1 (swap adjacent threads):\n");
    printf("First 8 values: ");
    for (int i = 0; i < 8; i++) {
        printf("%.0f ", h_output[64 + i]);
    }
    printf("\nExpected:       1 0 3 2 5 4 7 6\n");
    
    // Test SWAPX16 (swap 16-thread groups)
    printf("\nSWAPX16 (swap 16-thread groups):\n");
    printf("Lane 0-3:   %.0f %.0f %.0f %.0f\n", 
           h_output[320 + 0], h_output[320 + 1], h_output[320 + 2], h_output[320 + 3]);
    printf("Lane 16-19: %.0f %.0f %.0f %.0f\n",
           h_output[320 + 16], h_output[320 + 17], h_output[320 + 18], h_output[320 + 19]);
    printf("Expected lane 0-3:   16 17 18 19\n");
    printf("Expected lane 16-19: 0 1 2 3\n");
    
    // Test REVERSEX32 (reverse within 32-thread groups)
    printf("\nREVERSEX32 (reverse within 32-thread groups):\n");
    printf("Lane 0-3:   %.0f %.0f %.0f %.0f\n",
           h_output[384 + 0], h_output[384 + 1], h_output[384 + 2], h_output[384 + 3]);
    printf("Lane 28-31: %.0f %.0f %.0f %.0f\n",
           h_output[384 + 28], h_output[384 + 29], h_output[384 + 30], h_output[384 + 31]);
    printf("Expected lane 0-3:   31 30 29 28\n");
    printf("Expected lane 28-31: 3 2 1 0\n");
    
    // Test BCASTX4 (broadcast from first of each quad)
    printf("\nBCASTX4 (broadcast from first thread of each quad):\n");
    printf("Lane 0-7: ");
    for (int i = 0; i < 8; i++) {
        printf("%.0f ", h_output[512 + i]);
    }
    printf("\nExpected: 0 0 0 0 4 4 4 4\n");
    
    free(h_output);
    hipFree(d_output);
    return 0;
}