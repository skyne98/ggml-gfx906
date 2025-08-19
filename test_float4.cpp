#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <cstdio>

__global__ void test_float4(float4* out) {
    int idx = threadIdx.x;
    if (idx == 0) {
        float4 val = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
        out[0] = val;
    }
}

int main() {
    float4 *d_out;
    hipMalloc(&d_out, sizeof(float4));
    
    test_float4<<<1, 32>>>(d_out);
    hipDeviceSynchronize();
    
    float4 h_out;
    hipMemcpy(&h_out, d_out, sizeof(float4), hipMemcpyDeviceToHost);
    
    printf("float4 test: %.1f %.1f %.1f %.1f\n", h_out.x, h_out.y, h_out.z, h_out.w);
    
    hipFree(d_out);
    return 0;
}
