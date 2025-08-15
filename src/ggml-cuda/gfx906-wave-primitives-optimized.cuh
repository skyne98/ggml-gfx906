#pragma once

// GFX906 Wave-level Primitives with Native DS_SWIZZLE_B32
// This file provides optimized wave operations using correct DS_SWIZZLE encoding

#ifdef GGML_HIP_GFX906_OPTIMIZED

#include <hip/hip_runtime.h>

namespace gfx906 {

// Wave size constant for GCN architecture
static constexpr int WAVE_SIZE = 64;

// ================================
// DS_SWIZZLE_B32 Helper Functions
// ================================

// DS_SWIZZLE with correct encoding for 32-thread mode
// offset[15] = 0 (32-thread mode)
// offset[14:10] = xor_mask
// offset[9:5] = or_mask
// offset[4:0] = and_mask
__device__ __forceinline__ float ds_swizzle_xor(float value, int xor_mask) {
    float result;
    // Build offset: xor_mask in bits [14:10], and_mask = 0x1f for full range
    int offset = (xor_mask << 10) | 0x1f;
    
    switch(xor_mask) {
        case 1:  // SWAPX1
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x041F\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        case 2:  // SWAPX2
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x081F\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        case 4:  // SWAPX4
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x101F\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        case 8:  // SWAPX8
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x201F\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        case 16: // SWAPX16
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x401F\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        default:
            result = value; // No swizzle
    }
    return result;
}

// ================================
// Optimized Wave Reduction
// ================================

// Wave reduce sum using native DS_SWIZZLE_B32
template <typename T>
__device__ __forceinline__ T wave_reduce_sum_optimized(T value) {
    static_assert(sizeof(T) == 4, "DS_SWIZZLE only supports 32-bit types");
    
    // Reduction within 32-thread groups using DS_SWIZZLE
    value += ds_swizzle_xor(value, 1);   // SWAPX1
    value += ds_swizzle_xor(value, 2);   // SWAPX2
    value += ds_swizzle_xor(value, 4);   // SWAPX4
    value += ds_swizzle_xor(value, 8);   // SWAPX8
    value += ds_swizzle_xor(value, 16);  // SWAPX16
    
    // For full 64-thread reduction, use __shfl_xor for cross-wave communication
    // DS_SWIZZLE only works within 32-thread groups
    value += __shfl_xor(value, 32, WAVE_SIZE);
    
    return value;
}

// Specialized version for half2
__device__ __forceinline__ __half2 wave_reduce_sum_optimized(__half2 value) {
    // Convert to float for DS_SWIZZLE
    float2 f = __half22float2(value);
    
    // Reduce each component
    f.x = wave_reduce_sum_optimized(f.x);
    f.y = wave_reduce_sum_optimized(f.y);
    
    return __float22half2_rn(f);
}

// Wave reduce max using DS_SWIZZLE
template <typename T>
__device__ __forceinline__ T wave_reduce_max_optimized(T value) {
    T shuffled;
    
    shuffled = ds_swizzle_xor(value, 1);
    value = (value > shuffled) ? value : shuffled;
    
    shuffled = ds_swizzle_xor(value, 2);
    value = (value > shuffled) ? value : shuffled;
    
    shuffled = ds_swizzle_xor(value, 4);
    value = (value > shuffled) ? value : shuffled;
    
    shuffled = ds_swizzle_xor(value, 8);
    value = (value > shuffled) ? value : shuffled;
    
    shuffled = ds_swizzle_xor(value, 16);
    value = (value > shuffled) ? value : shuffled;
    
    // Cross-wave communication
    shuffled = __shfl_xor(value, 32, WAVE_SIZE);
    value = (value > shuffled) ? value : shuffled;
    
    return value;
}

// ================================
// Broadcast Operations
// ================================

// Broadcast using DS_BPERMUTE_B32 (confirmed working)
template <typename T>
__device__ __forceinline__ T wave_broadcast_optimized(T value, int src_lane) {
    static_assert(sizeof(T) == 4, "DS_BPERMUTE only supports 32-bit types");
    
    T result;
    int addr = src_lane * 4;  // Byte address
    
    asm volatile(
        "ds_bpermute_b32 %0, %1, %2\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(addr), "v"(value)
    );
    return result;
}

// Broadcast from first lane using readfirstlane
template <typename T>
__device__ __forceinline__ T wave_broadcast_first_optimized(T value) {
    return __builtin_amdgcn_readfirstlane(value);
}

// ================================
// Special Swizzle Patterns
// ================================

// Reverse within 32-thread groups
__device__ __forceinline__ float ds_swizzle_reverse32(float value) {
    float result;
    // REVERSEX32: xor_mask=0x1f, or_mask=0x00, and_mask=0x1f
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0x7C1F\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result) : "v"(value)
    );
    return result;
}

// Broadcast from first thread of each quad
__device__ __forceinline__ float ds_swizzle_bcast4(float value) {
    float result;
    // BCASTX4: xor_mask=0x00, or_mask=0x00, and_mask=0x1c
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0x001C\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result) : "v"(value)
    );
    return result;
}

// Quad permute mode for full data sharing within quads
__device__ __forceinline__ float ds_swizzle_quad_perm(float value, int perm) {
    float result;
    // offset[15] = 1 (quad mode)
    // perm encodes the permutation pattern for 4 threads
    int offset = 0x8000 | (perm & 0xFF);
    
    // Since we need compile-time constants, use specific patterns
    switch(perm) {
        case 0x39: // Rotate: 0->1, 1->2, 2->3, 3->0
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x8039\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        case 0x4E: // Swap pairs: 0<->1, 2<->3
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x804E\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        case 0x1B: // Reverse: 0<->3, 1<->2
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x801B\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        default:
            result = value;
    }
    return result;
}

// ================================
// Prefix Operations
// ================================

// Inclusive prefix sum using DS_SWIZZLE
template <typename T>
__device__ __forceinline__ T wave_prefix_sum_inclusive_optimized(T value) {
    T shuffled;
    
    // Use DS_SWIZZLE for efficient prefix computation
    // This is more complex but shows the pattern
    
    // Step 1: Adjacent pairs
    shuffled = ds_swizzle_xor(value, 1);
    if (threadIdx.x & 1) value += shuffled;
    
    // Step 2: Groups of 4
    shuffled = ds_swizzle_xor(value, 2);
    if (threadIdx.x & 2) value += shuffled;
    
    // Step 3: Groups of 8
    shuffled = ds_swizzle_xor(value, 4);
    if (threadIdx.x & 4) value += shuffled;
    
    // Step 4: Groups of 16
    shuffled = ds_swizzle_xor(value, 8);
    if (threadIdx.x & 8) value += shuffled;
    
    // Step 5: Groups of 32
    shuffled = ds_swizzle_xor(value, 16);
    if (threadIdx.x & 16) value += shuffled;
    
    // Step 6: Cross wave boundary
    shuffled = __shfl_xor(value, 32, WAVE_SIZE);
    if (threadIdx.x & 32) value += shuffled;
    
    return value;
}

// ================================
// Utility Functions
// ================================

// Get current lane ID within wave
__device__ __forceinline__ int __lane_id() {
    return threadIdx.x & (WAVE_SIZE - 1);
}

// Get wave ID within block
__device__ __forceinline__ int __wave_id() {
    return threadIdx.x / WAVE_SIZE;
}

// Check if current thread is wave leader
__device__ __forceinline__ bool is_wave_leader() {
    return (__lane_id() == 0);
}

} // namespace gfx906

// ================================
// Compatibility Macros
// ================================

// Map to optimized implementations when available
#ifdef __gfx906__
    #define wave_reduce_sum      gfx906::wave_reduce_sum_optimized
    #define wave_reduce_max      gfx906::wave_reduce_max_optimized
    #define wave_broadcast       gfx906::wave_broadcast_optimized
    #define wave_broadcast_first gfx906::wave_broadcast_first_optimized
    #define wave_prefix_sum      gfx906::wave_prefix_sum_inclusive_optimized
#else
    // Fallback to standard implementations
    #include "gfx906-wave-primitives.cuh"
#endif

#endif // GGML_HIP_GFX906_OPTIMIZED