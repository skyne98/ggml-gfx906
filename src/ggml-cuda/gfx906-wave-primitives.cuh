#pragma once

// GFX906 Wave-level Primitives Implementation
// This file provides optimized wave-level operations for 64-thread GCN waves
// on AMD GFX906 (Vega 20) architecture

#ifdef GGML_HIP_GFX906_OPTIMIZED

#    include <hip/hip_runtime.h>

// Namespace for GFX906-specific wave operations
namespace gfx906 {

// Wave size constant for GCN architecture
static constexpr int WAVE_SIZE = 64;

// ================================
// Wave Reduction Operations
// ================================

// Wave reduce sum using native DS_SWIZZLE for GFX906
// This uses the hardware swizzle instruction for maximum performance
template <typename T> __device__ __forceinline__ T wave_reduce_sum(T value) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Only 32-bit and 64-bit types supported");

    if constexpr (sizeof(T) == 4) {
// 32-bit reduction using native DS_SWIZZLE (1.35x faster than shuffle)
// Pattern 0x1F is XOR butterfly, manually unrolled with compile-time constants
        value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 32);
        value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 16);
        value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 8);
        value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 4);
        value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 2);
        value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 1);
    } else {
        // 64-bit reduction requires special handling
        // Split into two 32-bit operations
        union {
            T       val64;
            int32_t val32[2];
        } tmp;

        tmp.val64 = value;

#    pragma unroll
        for (int offset = 32; offset >= 1; offset >>= 1) {
            // Use native swizzle for each 32-bit component
            tmp.val32[0] += __builtin_amdgcn_ds_swizzle(tmp.val32[0], (0x1F << 10) | 32);
            tmp.val32[0] += __builtin_amdgcn_ds_swizzle(tmp.val32[0], (0x1F << 10) | 16);
            tmp.val32[0] += __builtin_amdgcn_ds_swizzle(tmp.val32[0], (0x1F << 10) | 8);
            tmp.val32[0] += __builtin_amdgcn_ds_swizzle(tmp.val32[0], (0x1F << 10) | 4);
            tmp.val32[0] += __builtin_amdgcn_ds_swizzle(tmp.val32[0], (0x1F << 10) | 2);
            tmp.val32[0] += __builtin_amdgcn_ds_swizzle(tmp.val32[0], (0x1F << 10) | 1);
            
            tmp.val32[1] += __builtin_amdgcn_ds_swizzle(tmp.val32[1], (0x1F << 10) | 32);
            tmp.val32[1] += __builtin_amdgcn_ds_swizzle(tmp.val32[1], (0x1F << 10) | 16);
            tmp.val32[1] += __builtin_amdgcn_ds_swizzle(tmp.val32[1], (0x1F << 10) | 8);
            tmp.val32[1] += __builtin_amdgcn_ds_swizzle(tmp.val32[1], (0x1F << 10) | 4);
            tmp.val32[1] += __builtin_amdgcn_ds_swizzle(tmp.val32[1], (0x1F << 10) | 2);
            tmp.val32[1] += __builtin_amdgcn_ds_swizzle(tmp.val32[1], (0x1F << 10) | 1);
        }
        value = tmp.val64;
    }
    return value;
}

// Specialized version for half2 using native swizzle
__device__ __forceinline__ __half2 wave_reduce_sum(__half2 value) {
    // Cast to int for swizzle, then back to half2
    int val_as_int = *reinterpret_cast<int*>(&value);
    
    // Native swizzle operations on the bit pattern
    __half2 swizzled;
    int temp;
    
    temp = __builtin_amdgcn_ds_swizzle(val_as_int, (0x1F << 10) | 32);
    swizzled = *reinterpret_cast<__half2*>(&temp);
    value = __hadd2(value, swizzled);
    
    temp = __builtin_amdgcn_ds_swizzle(val_as_int, (0x1F << 10) | 16);
    swizzled = *reinterpret_cast<__half2*>(&temp);
    value = __hadd2(value, swizzled);
    
    temp = __builtin_amdgcn_ds_swizzle(val_as_int, (0x1F << 10) | 8);
    swizzled = *reinterpret_cast<__half2*>(&temp);
    value = __hadd2(value, swizzled);
    
    temp = __builtin_amdgcn_ds_swizzle(val_as_int, (0x1F << 10) | 4);
    swizzled = *reinterpret_cast<__half2*>(&temp);
    value = __hadd2(value, swizzled);
    
    temp = __builtin_amdgcn_ds_swizzle(val_as_int, (0x1F << 10) | 2);
    swizzled = *reinterpret_cast<__half2*>(&temp);
    value = __hadd2(value, swizzled);
    
    temp = __builtin_amdgcn_ds_swizzle(val_as_int, (0x1F << 10) | 1);
    swizzled = *reinterpret_cast<__half2*>(&temp);
    value = __hadd2(value, swizzled);
    
    return value;
}

// Wave reduce max using native DS_SWIZZLE
template <typename T> __device__ __forceinline__ T wave_reduce_max(T value) {
    T shuffled;
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 32);
    value = (value > shuffled) ? value : shuffled;
    
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 16);
    value = (value > shuffled) ? value : shuffled;
    
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 8);
    value = (value > shuffled) ? value : shuffled;
    
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 4);
    value = (value > shuffled) ? value : shuffled;
    
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 2);
    value = (value > shuffled) ? value : shuffled;
    
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 1);
    value = (value > shuffled) ? value : shuffled;
    
    return value;
}

// Wave reduce min using native DS_SWIZZLE
template <typename T> __device__ __forceinline__ T wave_reduce_min(T value) {
    T shuffled;
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 32);
    value = (value < shuffled) ? value : shuffled;
    
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 16);
    value = (value < shuffled) ? value : shuffled;
    
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 8);
    value = (value < shuffled) ? value : shuffled;
    
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 4);
    value = (value < shuffled) ? value : shuffled;
    
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 2);
    value = (value < shuffled) ? value : shuffled;
    
    shuffled = __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 1);
    value = (value < shuffled) ? value : shuffled;
    
    return value;
}

// Wave reduce with custom operation
template <typename T, typename Op> __device__ __forceinline__ T wave_reduce_op(T value, Op op) {
#    pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) {
        value = op(value, __shfl_xor(value, offset, WAVE_SIZE));
    }
    return value;
}

// ================================
// Wave Broadcast Operations
// ================================

// Broadcast value from lane 0 to all lanes
template <typename T> __device__ __forceinline__ T wave_broadcast_first(T value) {
    return __builtin_amdgcn_readfirstlane(value);
}

// Broadcast value from specific lane to all lanes
template <typename T> __device__ __forceinline__ T wave_broadcast(T value, int src_lane) {
    // Use shuffle to broadcast from specific lane
    return __shfl(value, src_lane, WAVE_SIZE);
}

// ================================
// Wave Shuffle Operations
// ================================

// Shuffle value from source lane
template <typename T> __device__ __forceinline__ T wave_shuffle(T value, int src_lane) {
    static_assert(sizeof(T) == 4, "wave_shuffle only supports 32-bit types");
    // Use standard shuffle
    return __shfl(value, src_lane, WAVE_SIZE);
}

// Shuffle with XOR mask (butterfly patterns)
template <typename T> __device__ __forceinline__ T wave_shuffle_xor(T value, int mask) {
    int src_lane = __lane_id() ^ mask;
    return wave_shuffle(value, src_lane);
}

// Shuffle up (from lane - delta)
template <typename T> __device__ __forceinline__ T wave_shuffle_up(T value, int delta) {
    int src_lane = __lane_id() - delta;
    if (src_lane < 0) {
        src_lane = __lane_id();  // Clamp to current lane
    }
    return wave_shuffle(value, src_lane);
}

// Shuffle down (from lane + delta)
template <typename T> __device__ __forceinline__ T wave_shuffle_down(T value, int delta) {
    int src_lane = __lane_id() + delta;
    if (src_lane >= WAVE_SIZE) {
        src_lane = __lane_id();  // Clamp to current lane
    }
    return wave_shuffle(value, src_lane);
}

// ================================
// Wave Prefix Operations
// ================================

// Inclusive prefix sum
template <typename T> __device__ __forceinline__ T wave_prefix_sum_inclusive(T value) {
#    pragma unroll
    for (int offset = 1; offset < WAVE_SIZE; offset <<= 1) {
        T shuffled = wave_shuffle_up(value, offset);
        if (__lane_id() >= offset) {
            value += shuffled;
        }
    }
    return value;
}

// Exclusive prefix sum
template <typename T> __device__ __forceinline__ T wave_prefix_sum_exclusive(T value) {
    T inclusive = wave_prefix_sum_inclusive(value);
    return wave_shuffle_up(inclusive, 1) * (__lane_id() > 0);
}

// Prefix operation with custom operator
template <typename T, typename Op> __device__ __forceinline__ T wave_prefix_op_inclusive(T value, Op op) {
#    pragma unroll
    for (int offset = 1; offset < WAVE_SIZE; offset <<= 1) {
        T shuffled = wave_shuffle_up(value, offset);
        if (__lane_id() >= offset) {
            value = op(value, shuffled);
        }
    }
    return value;
}

// ================================
// Wave Vote Operations
// ================================

// All threads in wave have non-zero value
__device__ __forceinline__ bool wave_all(int predicate) {
    return __all(predicate);
}

// Any thread in wave has non-zero value
__device__ __forceinline__ bool wave_any(int predicate) {
    return __any(predicate);
}

// Get ballot mask for predicate across wave
__device__ __forceinline__ uint64_t wave_ballot(int predicate) {
    return __ballot(predicate);
}

// Count number of set bits in ballot
__device__ __forceinline__ int wave_popc(uint64_t mask) {
    return __popcll(mask);
}

// ================================
// Wave Scan Operations
// ================================

// Segmented scan with head flags
template <typename T> __device__ __forceinline__ T wave_segmented_scan(T value, bool is_head) {
    T scan = value;

#    pragma unroll
    for (int offset = 1; offset < WAVE_SIZE; offset <<= 1) {
        T    shuffled      = wave_shuffle_up(scan, offset);
        bool head_shuffled = wave_shuffle_up(is_head, offset);

        if (__lane_id() >= offset && !is_head) {
            scan = head_shuffled ? scan : (scan + shuffled);
        }
    }
    return scan;
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

// Get number of active lanes in wave
__device__ __forceinline__ int wave_active_count() {
    return wave_popc(wave_ballot(1));
}

// ================================
// GFX906-Specific Swizzle Patterns
// ================================

// Reverse lanes in wave (lane 0 <-> 63, 1 <-> 62, etc.)
template <typename T> __device__ __forceinline__ T wave_reverse(T value) {
    static_assert(sizeof(T) == 4, "wave_reverse only supports 32-bit types");
    // Pattern 0x1E with mask 0x3F reverses the wave
    return __builtin_amdgcn_ds_swizzle(value, (0x1E << 10) | 0x3F);
}

// Rotate lanes by N positions
template <int N> __device__ __forceinline__ float wave_rotate(float value) {
    static_assert(N >= 0 && N < 64, "Rotation must be in range [0, 64)");
    // Pattern 0x1D performs rotation
    return __builtin_amdgcn_ds_swizzle(value, (0x1D << 10) | (N & 0x3F));
}

// Broadcast from lane 0 using swizzle (faster than readfirstlane)
template <typename T> __device__ __forceinline__ T wave_broadcast_lane0_swizzle(T value) {
    static_assert(sizeof(T) == 4, "Only 32-bit types supported");
    // Pattern 0x00 broadcasts from lane 0
    return __builtin_amdgcn_ds_swizzle(value, (0x00 << 10) | 0);
}

// Swap adjacent pairs (0<->1, 2<->3, etc.)
template <typename T> __device__ __forceinline__ T wave_swap_adjacent(T value) {
    static_assert(sizeof(T) == 4, "Only 32-bit types supported");
    return __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 1);
}

// Swap groups of 4 (0-3<->4-7, etc.)
template <typename T> __device__ __forceinline__ T wave_swap_quads(T value) {
    static_assert(sizeof(T) == 4, "Only 32-bit types supported");
    return __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 4);
}

// ================================
// Complex Operations
// ================================

// Dot product across wave
template <typename T> __device__ __forceinline__ T wave_dot_product(T a, T b) {
    return wave_reduce_sum(a * b);
}

// Matrix transpose within wave (for 8x8 tiles)
template <typename T> __device__ __forceinline__ void wave_transpose_8x8(T * values) {
    int lane = __lane_id();
    int row  = lane / 8;
    int col  = lane % 8;

// Transpose using shuffle operations
#    pragma unroll
    for (int i = 0; i < 8; i++) {
        int src_lane = col * 8 + row;
        values[i]    = wave_shuffle(values[i], src_lane);
    }
}

// Parallel reduction across multiple waves in a block
template <typename T> __device__ T block_reduce_sum(T value) {
    // First reduce within each wave
    value = wave_reduce_sum(value);

    // Then reduce across waves using shared memory
    __shared__ T wave_sums[1024 / WAVE_SIZE];  // Max waves per block

    int wave_id   = __wave_id();
    int num_waves = blockDim.x / WAVE_SIZE;

    if (is_wave_leader()) {
        wave_sums[wave_id] = value;
    }
    __syncthreads();

    // Final reduction in first wave
    if (wave_id == 0 && __lane_id() < num_waves) {
        value = wave_sums[__lane_id()];
        value = wave_reduce_sum(value);
    }

    return value;
}

// ================================
// Specialized INT8 Operations
// ================================

// INT8 dot product using V_DOT4_I32_I8
__device__ __forceinline__ int32_t wave_dot4_i8(int32_t a, int32_t b) {
    int32_t result = __builtin_amdgcn_sdot4(a, b, 0, false);
    return wave_reduce_sum(result);
}

// ================================
// Specialized FP16 Operations
// ================================

// FP16 dot product using V_DOT2_F32_F16
__device__ __forceinline__ float wave_dot2_f16(uint32_t a, uint32_t b) {
    float result;
    asm volatile("v_dot2_f32_f16 %0, %1, %2, 0.0" : "=v"(result) : "v"(a), "v"(b));
    return wave_reduce_sum(result);
}

// FP16 reduction with packed half2
__device__ __forceinline__ float wave_reduce_sum_f16x2(__half2 value) {
    value = wave_reduce_sum(value);
    return __low2float(value) + __high2float(value);
}

}  // namespace gfx906

// ================================
// Compatibility Macros
// ================================

// Map generic names to GFX906 implementations
#    define wave_reduce_sum      gfx906::wave_reduce_sum
#    define wave_reduce_max      gfx906::wave_reduce_max
#    define wave_reduce_min      gfx906::wave_reduce_min
#    define wave_broadcast       gfx906::wave_broadcast
#    define wave_broadcast_first gfx906::wave_broadcast_first
#    define wave_shuffle         gfx906::wave_shuffle
#    define wave_shuffle_xor     gfx906::wave_shuffle_xor
#    define wave_prefix_sum      gfx906::wave_prefix_sum_inclusive
#    define wave_all             gfx906::wave_all
#    define wave_any             gfx906::wave_any
#    define wave_ballot          gfx906::wave_ballot

#endif  // GGML_HIP_GFX906_OPTIMIZED
