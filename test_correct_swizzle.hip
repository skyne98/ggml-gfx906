#include <hip/hip_runtime.h>
#include <stdio.h>

// Test DS_SWIZZLE_B32 with CORRECT encoding based on ISA pseudocode
// For 32-thread mode (offset[15] == 0):
// offset[14:10] = xor_mask
// offset[9:5] = or_mask
// offset[4:0] = and_mask
__global__ void test_correct_swizzle(float* output) {
    int tid = threadIdx.x;
    float value = tid;
    
    if (tid < 64) {
        // Original value
        output[tid] = value;
        
        // SWAPX1: xor_mask=0x01, or_mask=0x00, and_mask=0x1f
        // offset = (0 << 15) | (0x01 << 10) | (0x00 << 5) | 0x1f = 0x041F
        float swap1;
        asm volatile(
            "ds_swizzle_b32 %0, %1 offset:0x041F\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(swap1)
            : "v"(value)
        );
        output[64 + tid] = swap1;
        
        // SWAPX2: xor_mask=0x02, or_mask=0x00, and_mask=0x1f
        // offset = (0 << 15) | (0x02 << 10) | (0x00 << 5) | 0x1f = 0x081F
        float swap2;
        asm volatile(
            "ds_swizzle_b32 %0, %1 offset:0x081F\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(swap2)
            : "v"(value)
        );
        output[128 + tid] = swap2;
        
        // SWAPX4: xor_mask=0x04, or_mask=0x00, and_mask=0x1f
        // offset = (0 << 15) | (0x04 << 10) | (0x00 << 5) | 0x1f = 0x101F
        float swap4;
        asm volatile(
            "ds_swizzle_b32 %0, %1 offset:0x101F\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(swap4)
            : "v"(value)
        );
        output[192 + tid] = swap4;
        
        // SWAPX8: xor_mask=0x08, or_mask=0x00, and_mask=0x1f
        // offset = (0 << 15) | (0x08 << 10) | (0x00 << 5) | 0x1f = 0x201F
        float swap8;
        asm volatile(
            "ds_swizzle_b32 %0, %1 offset:0x201F\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(swap8)
            : "v"(value)
        );
        output[256 + tid] = swap8;
        
        // SWAPX16: xor_mask=0x10, or_mask=0x00, and_mask=0x1f
        // offset = (0 << 15) | (0x10 << 10) | (0x00 << 5) | 0x1f = 0x401F
        float swap16;
        asm volatile(
            "ds_swizzle_b32 %0, %1 offset:0x401F\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(swap16)
            : "v"(value)
        );
        output[320 + tid] = swap16;
        
        // REVERSEX32: xor_mask=0x1f, or_mask=0x00, and_mask=0x1f
        // offset = (0 << 15) | (0x1f << 10) | (0x00 << 5) | 0x1f = 0x7C1F
        float reverse32;
        asm volatile(
            "ds_swizzle_b32 %0, %1 offset:0x7C1F\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(reverse32)
            : "v"(value)
        );
        output[384 + tid] = reverse32;
        
        // BCASTX4: xor_mask=0x00, or_mask=thread, and_mask=0x1c
        // For broadcasting from thread 0 of each quad: or_mask=0x00
        // offset = (0 << 15) | (0x00 << 10) | (0x00 << 5) | 0x1c = 0x001C
        float bcast4;
        asm volatile(
            "ds_swizzle_b32 %0, %1 offset:0x001C\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(bcast4)
            : "v"(value)
        );
        output[448 + tid] = bcast4;
        
        // Test Quad Permute mode (offset[15] == 1)
        // Example: Rotate within quad - thread 0->1, 1->2, 2->3, 3->0
        // offset[1:0] = 1 (thread 0 gets from thread 1)
        // offset[3:2] = 2 (thread 1 gets from thread 2)
        // offset[5:4] = 3 (thread 2 gets from thread 3)
        // offset[7:6] = 0 (thread 3 gets from thread 0)
        // offset = (1 << 15) | (0 << 6) | (3 << 4) | (2 << 2) | 1 = 0x8039
        float quad_rotate;
        asm volatile(
            "ds_swizzle_b32 %0, %1 offset:0x8039\n\t"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(quad_rotate)
            : "v"(value)
        );
        output[512 + tid] = quad_rotate;
    }
}

int main() {
    float *d_output, *h_output;
    h_output = (float*)malloc(576 * sizeof(float));
    
    hipMalloc(&d_output, 576 * sizeof(float));
    
    test_correct_swizzle<<<1, 64>>>(d_output);
    hipDeviceSynchronize();
    
    hipMemcpy(h_output, d_output, 576 * sizeof(float), hipMemcpyDeviceToHost);
    
    printf("Testing DS_SWIZZLE_B32 with CORRECT ISA Encoding:\n");
    printf("==================================================\n\n");
    
    // Test SWAPX1
    printf("SWAPX1 (XOR with 1, swap adjacent):\n");
    bool swap1_correct = true;
    for (int i = 0; i < 8; i++) {
        printf("%.0f ", h_output[64 + i]);
        float expected = (i % 2 == 0) ? i + 1 : i - 1;
        if (h_output[64 + i] != expected) swap1_correct = false;
    }
    printf("\nExpected: 1 0 3 2 5 4 7 6\n");
    printf("Status: %s\n\n", swap1_correct ? "PASSED" : "FAILED");
    
    // Test SWAPX16
    printf("SWAPX16 (XOR with 16, swap 16-thread groups):\n");
    printf("Lane 0-3:   %.0f %.0f %.0f %.0f\n",
           h_output[320 + 0], h_output[320 + 1], h_output[320 + 2], h_output[320 + 3]);
    printf("Lane 16-19: %.0f %.0f %.0f %.0f\n",
           h_output[320 + 16], h_output[320 + 17], h_output[320 + 18], h_output[320 + 19]);
    printf("Expected lane 0-3:   16 17 18 19\n");
    printf("Expected lane 16-19: 0 1 2 3\n");
    
    bool swap16_correct = true;
    for (int i = 0; i < 16; i++) {
        if (h_output[320 + i] != i + 16) swap16_correct = false;
        if (h_output[320 + i + 16] != i) swap16_correct = false;
    }
    printf("Status: %s\n\n", swap16_correct ? "PASSED" : "FAILED");
    
    // Test REVERSEX32
    printf("REVERSEX32 (reverse within 32-thread groups):\n");
    printf("Lane 0-3:   %.0f %.0f %.0f %.0f\n",
           h_output[384 + 0], h_output[384 + 1], h_output[384 + 2], h_output[384 + 3]);
    printf("Lane 28-31: %.0f %.0f %.0f %.0f\n",
           h_output[384 + 28], h_output[384 + 29], h_output[384 + 30], h_output[384 + 31]);
    printf("Expected lane 0-3:   31 30 29 28\n");
    printf("Expected lane 28-31: 3 2 1 0\n");
    
    bool reverse_correct = true;
    for (int i = 0; i < 32; i++) {
        if (h_output[384 + i] != 31 - i) reverse_correct = false;
        if (h_output[384 + i + 32] != 63 - i) reverse_correct = false;
    }
    printf("Status: %s\n\n", reverse_correct ? "PASSED" : "FAILED");
    
    // Test BCASTX4
    printf("BCASTX4 (broadcast from first of each quad):\n");
    printf("Lane 0-7: ");
    for (int i = 0; i < 8; i++) {
        printf("%.0f ", h_output[448 + i]);
    }
    printf("\nExpected: 0 0 0 0 4 4 4 4\n");
    
    bool bcast4_correct = true;
    for (int i = 0; i < 64; i++) {
        float expected = (i / 4) * 4;  // First thread of each quad
        if (h_output[448 + i] != expected) bcast4_correct = false;
    }
    printf("Status: %s\n\n", bcast4_correct ? "PASSED" : "FAILED");
    
    // Test Quad Permute
    printf("Quad Permute (rotate within quad):\n");
    printf("Lane 0-7: ");
    for (int i = 0; i < 8; i++) {
        printf("%.0f ", h_output[512 + i]);
    }
    printf("\nExpected: 1 2 3 0 5 6 7 4\n");
    
    bool quad_correct = true;
    for (int i = 0; i < 64; i++) {
        int quad_id = i / 4;
        int lane_in_quad = i % 4;
        float expected = quad_id * 4 + ((lane_in_quad + 1) % 4);
        if (h_output[512 + i] != expected) quad_correct = false;
    }
    printf("Status: %s\n\n", quad_correct ? "PASSED" : "FAILED");
    
    // Overall result
    bool all_passed = swap1_correct && swap16_correct && reverse_correct && 
                      bcast4_correct && quad_correct;
    printf("==================================================\n");
    printf("Overall: %s\n", all_passed ? "ALL TESTS PASSED!" : "Some tests failed");
    
    free(h_output);
    hipFree(d_output);
    return all_passed ? 0 : 1;
}