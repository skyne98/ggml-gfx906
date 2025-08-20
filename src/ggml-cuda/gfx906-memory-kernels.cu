// GFX906 Memory Copy Kernel Implementation
// Actual kernel definitions that can be called from other compilation units

#ifdef GGML_HIP_GFX906_OPTIMIZED

#include "common.cuh"
#include "gfx906-memory-isa.cuh"

namespace gfx906 {
namespace memory_isa {

// Actual kernel that wraps the device function
__global__ void memcpy_dwordx4_kernel(float * __restrict__ dst, const float * __restrict__ src, size_t n_float4) {
    memcpy_dwordx4_device(dst, src, n_float4);
}

} // namespace memory_isa
} // namespace gfx906

#endif // GGML_HIP_GFX906_OPTIMIZED