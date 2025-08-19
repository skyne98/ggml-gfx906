#include <hip/hip_runtime.h>
#include <cstdio>

// Simple test kernel to verify launch
__global__ void test_kernel(float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = idx * 2.0f;
    }
}

#define HIP_CHECK(call) do { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error: %s\n", hipGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

int main() {
    const int size = 1024;
    float *d_output;
    
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(float)));
    HIP_CHECK(hipMemset(d_output, 0, size * sizeof(float)));
    
    // Launch test kernel
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    printf("Launching kernel with grid(%d) block(%d)\n", grid.x, block.x);
    test_kernel<<<grid, block>>>(d_output, size);
    
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    
    // Check results
    float h_output[10];
    HIP_CHECK(hipMemcpy(h_output, d_output, 10 * sizeof(float), hipMemcpyDeviceToHost));
    
    printf("First 10 outputs: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n");
    
    HIP_CHECK(hipFree(d_output));
    
    printf("Kernel launch successful!\n");
    return 0;
}