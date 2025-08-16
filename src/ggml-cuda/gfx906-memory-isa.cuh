#pragma once

// GFX906 Memory Access ISA-Level Implementation
// Low-level GCN assembly implementations for vectorized loads, coalesced access,
// and LDS double buffering using Vega 20nm ISA primitives

#ifdef GGML_HIP_GFX906_OPTIMIZED

#    include "gfx906-config.cuh"

#    include <hip/hip_runtime.h>

namespace gfx906 {
namespace memory_isa {

// ================================
// Vectorized 128-bit Loads (DWORDX4)
// ================================

// BUFFER_LOAD_DWORDX4 - Load 4 consecutive dwords (128 bits)
__device__ __forceinline__ float4 buffer_load_dwordx4(const float * __restrict__ addr, uint32_t offset = 0) {
    float4 result;

    // Inline assembly for BUFFER_LOAD_DWORDX4
    // v[dst:dst+3] = buffer_load_dwordx4 v[addr], s[desc:desc+3], offset
    asm volatile(
        "buffer_load_dwordx4 %0, %1, s[0:3], %2 offen\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(result)
        : "v"(addr), "i"(offset)
        : "memory");

    return result;
}

// GLOBAL_LOAD_DWORDX4 - Global memory 128-bit load
__device__ __forceinline__ float4 global_load_dwordx4(const float * __restrict__ addr) {
    float4 result;

    // Ensure 16-byte alignment for optimal performance
    assert(((uintptr_t) addr & 0xF) == 0);

    // Inline assembly for GLOBAL_LOAD_DWORDX4
    asm volatile(
        "global_load_dwordx4 %0, %1, off\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(result)
        : "v"(addr)
        : "memory");

    return result;
}

// FLAT_LOAD_DWORDX4 - Flat address space 128-bit load
__device__ __forceinline__ float4 flat_load_dwordx4(const float * __restrict__ addr) {
    float4 result;

    asm volatile(
        "flat_load_dwordx4 %0, %1\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(result)
        : "v"(addr)
        : "memory");

    return result;
}

// Load 8 half-precision values (128 bits) as packed data
__device__ __forceinline__ uint4 global_load_half8(const __half * __restrict__ addr) {
    uint4 result;

    asm volatile(
        "global_load_dwordx4 %0, %1, off\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(result)
        : "v"(addr)
        : "memory");

    return result;
}

// ================================
// Coalesced Memory Access Patterns
// ================================

// Calculate coalesced address for current thread
__device__ __forceinline__ uintptr_t calculate_coalesced_address(const void * base_addr, size_t element_size) {
    // Get thread ID within wavefront (0-63)
    const uint32_t lane_id = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));

    // Calculate per-thread address for coalescing
    // address_for_thread_N = base_address + (N * element_size)
    return (uintptr_t) base_addr + (lane_id * element_size);
}

// Coalesced DWORD load across wavefront
__device__ __forceinline__ float coalesced_load_dword(const float * __restrict__ base_addr) {
    // Each thread loads from its coalesced position
    const float * thread_addr = base_addr + (__builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u)));

    float result;
    asm volatile(
        "global_load_dword %0, %1, off\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(result)
        : "v"(thread_addr)
        : "memory");

    return result;
}

// Coalesced DWORDX4 load across wavefront (256 bytes total)
__device__ __forceinline__ float4 coalesced_load_dwordx4(const float * __restrict__ base_addr) {
    // Each thread loads 16 bytes, wavefront loads 1024 bytes total
    const uint32_t lane_id     = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
    const float *  thread_addr = base_addr + (lane_id * 4);

    return global_load_dwordx4(thread_addr);
}

// ================================
// LDS Double Buffering with ISA
// ================================

// Direct LDS load bypassing VGPRs (asynchronous)
__device__ __forceinline__ void buffer_load_to_lds_async(const void * __restrict__ buffer_addr,
                                                         void * __restrict__ lds_addr,
                                                         uint32_t num_dwords) {
    // BUFFER_STORE_LDS_DWORD - Actually loads from buffer to LDS
    // This is asynchronous and increments lgkmcnt
    asm volatile("buffer_load_dword %0, %1, s[0:3], 0 offen lds\n\t" : : "v"(lds_addr), "v"(buffer_addr) : "memory");
}

// Wait for LDS operations to complete
__device__ __forceinline__ void wait_for_lds() {
    // S_WAITCNT lgkmcnt(0) - Wait for all LDS operations
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// Workgroup barrier synchronization
__device__ __forceinline__ void workgroup_barrier() {
    // S_BARRIER - Synchronize all wavefronts in workgroup
    asm volatile("s_barrier" ::: "memory");
}

// LDS double buffer implementation with ISA primitives
template <typename T, int BUFFER_SIZE> class LDSDoubleBufferISA {
  private:
    // Two buffers in LDS
    __shared__ T buffer_a[BUFFER_SIZE];
    __shared__ T buffer_b[BUFFER_SIZE];
    bool         current_is_a;

  public:
    __device__ LDSDoubleBufferISA() : current_is_a(true) {}

    // Get current compute buffer
    __device__ __forceinline__ T * get_compute_buffer() { return current_is_a ? buffer_a : buffer_b; }

    // Get current prefetch buffer
    __device__ __forceinline__ T * get_prefetch_buffer() { return current_is_a ? buffer_b : buffer_a; }

    // Async prefetch to next buffer using MUBUF with LDS=1
    __device__ __forceinline__ void prefetch_async(const T * __restrict__ global_addr, int elements) {
        T *       lds_buffer = get_prefetch_buffer();
        const int tid        = threadIdx.x;
        const int stride     = blockDim.x;

        // Each thread loads elements cooperatively
        for (int i = tid; i < elements; i += stride) {
            // Use MUBUF instruction with LDS bit set
            asm volatile("buffer_load_dword %0, %1, s[0:3], 0 offen lds\n\t"
                         :
                         : "v"(&lds_buffer[i]), "v"(&global_addr[i])
                         : "memory");
        }
    }

    // Synchronize and swap buffers
    __device__ __forceinline__ void sync_and_swap() {
        // Wait for LDS prefetch to complete
        wait_for_lds();

        // Synchronize all threads in workgroup
        workgroup_barrier();

        // Swap buffers
        current_is_a = !current_is_a;
    }

    // Initial prefetch before main loop
    __device__ __forceinline__ void initial_prefetch(const T * __restrict__ global_addr, int elements) {
        prefetch_async(global_addr, elements);
        wait_for_lds();
        workgroup_barrier();
    }
};

// ================================
// Optimized Memory Copy Kernels
// ================================

// High-bandwidth memory copy using DWORDX4
__global__ void memcpy_dwordx4_kernel(float * __restrict__ dst, const float * __restrict__ src, size_t n_float4) {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Each thread handles one float4 (128 bits)
    for (size_t i = tid; i < n_float4; i += stride) {
        // Calculate addresses
        const float * src_addr = src + (i * 4);
        float *       dst_addr = dst + (i * 4);

        // Load 128 bits
        float4 data = global_load_dwordx4(src_addr);

        // Store 128 bits
        asm volatile(
            "global_store_dwordx4 %0, %1, off\n\t"
            "s_waitcnt vmcnt(0)"
            :
            : "v"(dst_addr), "v"(data)
            : "memory");
    }
}

// ================================
// Prefetch Instructions
// ================================

// L2 cache prefetch hint
__device__ __forceinline__ void prefetch_l2_isa(const void * addr) {
    // GLC (globally coherent) and SLC (system level coherent) flags
    // help with L2 cache utilization
    asm volatile("global_load_dword v[0:0], %0, off glc slc\n\t" : : "v"(addr) : "v0", "memory");
}

// Prefetch with non-temporal hint (bypass cache)
__device__ __forceinline__ void prefetch_nt_isa(const void * addr) {
    // NT (non-temporal) flag bypasses cache for streaming data
    asm volatile("global_load_dword v[0:0], %0, off nt\n\t" : : "v"(addr) : "v0", "memory");
}

// ================================
// Memory Fence Instructions
// ================================

// Full memory fence
__device__ __forceinline__ void memory_fence_isa() {
    // S_WAITCNT waits for all memory operations
    asm volatile(
        "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
        "buffer_gl0_inv\n\t"
        "buffer_gl1_inv" ::
            : "memory");
}

// Vector memory fence only
__device__ __forceinline__ void vmem_fence_isa() {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

// LDS memory fence only
__device__ __forceinline__ void lds_fence_isa() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

}  // namespace memory_isa
}  // namespace gfx906

#endif  // GGML_HIP_GFX906_OPTIMIZED
