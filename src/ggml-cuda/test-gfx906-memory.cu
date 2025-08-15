// Test suite for GFX906 memory optimization patterns
// Verifies vectorized loads, coalesced access, and LDS double buffering

#ifdef GGML_HIP_GFX906_OPTIMIZED

#    include "gfx906-config.cuh"
#    include "gfx906-memory-isa.cuh"
#    include "gfx906-memory-patterns.cuh"

#    include <assert.h>
#    include <hip/hip_runtime.h>
#    include <stdio.h>

#    include <chrono>

using namespace gfx906;

// Test data size configurations
constexpr size_t TEST_SIZE_MB     = 256;
constexpr size_t TEST_SIZE_BYTES  = TEST_SIZE_MB * 1024 * 1024;
constexpr size_t TEST_SIZE_FLOATS = TEST_SIZE_BYTES / sizeof(float);
constexpr int    ITERATIONS       = 100;

// ================================
// Test Kernels
// ================================

// Baseline memory copy (unoptimized)
__global__ void baseline_copy_kernel(float * __restrict__ dst, const float * __restrict__ src, size_t n) {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// Optimized copy using vectorized loads
__global__ void vectorized_copy_kernel(float * __restrict__ dst, const float * __restrict__ src, size_t n) {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Process float4 chunks
    const size_t n_float4 = n / 4;

    for (size_t i = tid; i < n_float4; i += stride) {
        float4 data                              = memory::load_float4_aligned(src + i * 4);
        *reinterpret_cast<float4 *>(dst + i * 4) = data;
    }
}

// ISA-level optimized copy
__global__ void isa_dwordx4_copy_kernel(float * __restrict__ dst, const float * __restrict__ src, size_t n) {
    const int    tid      = blockIdx.x * blockDim.x + threadIdx.x;
    const int    stride   = gridDim.x * blockDim.x;
    const size_t n_float4 = n / 4;

    for (size_t i = tid; i < n_float4; i += stride) {
        float4 data = memory_isa::global_load_dwordx4(src + i * 4);

        asm volatile("global_store_dwordx4 %0, %1, off\n\t" : : "v"(dst + i * 4), "v"(data) : "memory");
    }
}

// Matrix transpose with coalesced access
__global__ void coalesced_transpose_kernel(float * __restrict__ dst,
                                           const float * __restrict__ src,
                                           int rows,
                                           int cols) {
    __shared__ float tile[32][33];  // +1 for bank conflict avoidance

    const int block_row  = blockIdx.y;
    const int block_col  = blockIdx.x;
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    // Coalesced read from global memory
    int src_row = block_row * 32 + thread_row;
    int src_col = block_col * 32 + thread_col;

    if (src_row < rows && src_col < cols) {
        tile[thread_row][thread_col] = src[src_row * cols + src_col];
    }

    __syncthreads();

    // Coalesced write to global memory (transposed)
    int dst_row = block_col * 32 + thread_row;
    int dst_col = block_row * 32 + thread_col;

    if (dst_row < cols && dst_col < rows) {
        dst[dst_row * rows + dst_col] = tile[thread_col][thread_row];
    }
}

// LDS double buffering test kernel
template <int TILE_SIZE>
__global__ void lds_double_buffer_kernel(float * __restrict__ dst, const float * __restrict__ src, size_t n) {
    memory_isa::LDSDoubleBufferISA<float, TILE_SIZE> buffer;

    const int tid             = threadIdx.x;
    const int block_id        = blockIdx.x;
    const int tiles_per_block = n / (gridDim.x * TILE_SIZE);

    // Initial prefetch
    size_t tile_start = block_id * tiles_per_block * TILE_SIZE;
    if (tile_start < n) {
        buffer.initial_prefetch(src + tile_start, TILE_SIZE);
    }

    // Main loop with double buffering
    for (int tile = 0; tile < tiles_per_block - 1; tile++) {
        size_t current_offset = tile_start + tile * TILE_SIZE;
        size_t next_offset    = current_offset + TILE_SIZE;

        // Prefetch next tile
        if (next_offset < n) {
            buffer.prefetch_async(src + next_offset, TILE_SIZE);
        }

        // Process current tile
        float * compute_buffer = buffer.get_compute_buffer();
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            if (current_offset + i < n) {
                // Simple processing: square the value
                float val               = compute_buffer[i];
                dst[current_offset + i] = val * val;
            }
        }

        // Sync and swap buffers
        buffer.sync_and_swap();
    }

    // Process last tile
    size_t last_offset = tile_start + (tiles_per_block - 1) * TILE_SIZE;
    if (last_offset < n) {
        float * compute_buffer = buffer.get_compute_buffer();
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            if (last_offset + i < n) {
                float val            = compute_buffer[i];
                dst[last_offset + i] = val * val;
            }
        }
    }
}

// ================================
// Test Functions
// ================================

void test_memory_bandwidth() {
    printf("\n=== Memory Bandwidth Test ===\n");
    printf("Test size: %zu MB\n", TEST_SIZE_MB);

    // Allocate device memory
    float *d_src, *d_dst;
    hipMalloc(&d_src, TEST_SIZE_BYTES);
    hipMalloc(&d_dst, TEST_SIZE_BYTES);

    // Initialize source data
    hipMemset(d_src, 1, TEST_SIZE_BYTES);

    // Configure kernel launches
    const int block_size = 256;
    const int grid_size  = GFX906_NUM_CUS * 16;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Test 1: Baseline copy
    {
        hipEventRecord(start);
        for (int i = 0; i < ITERATIONS; i++) {
            baseline_copy_kernel<<<grid_size, block_size>>>(d_dst, d_src, TEST_SIZE_FLOATS);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float elapsed_ms;
        hipEventElapsedTime(&elapsed_ms, start, stop);

        size_t total_bytes = TEST_SIZE_BYTES * ITERATIONS * 2;  // Read + Write
        auto   stats       = memory::calculate_bandwidth(total_bytes, elapsed_ms);

        printf("\nBaseline Copy:\n");
        stats.print();
    }

    // Test 2: Vectorized copy
    {
        hipEventRecord(start);
        for (int i = 0; i < ITERATIONS; i++) {
            vectorized_copy_kernel<<<grid_size, block_size>>>(d_dst, d_src, TEST_SIZE_FLOATS);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float elapsed_ms;
        hipEventElapsedTime(&elapsed_ms, start, stop);

        size_t total_bytes = TEST_SIZE_BYTES * ITERATIONS * 2;
        auto   stats       = memory::calculate_bandwidth(total_bytes, elapsed_ms);

        printf("\nVectorized Copy (float4):\n");
        stats.print();
    }

    // Test 3: ISA-level DWORDX4 copy
    {
        hipEventRecord(start);
        for (int i = 0; i < ITERATIONS; i++) {
            isa_dwordx4_copy_kernel<<<grid_size, block_size>>>(d_dst, d_src, TEST_SIZE_FLOATS);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float elapsed_ms;
        hipEventElapsedTime(&elapsed_ms, start, stop);

        size_t total_bytes = TEST_SIZE_BYTES * ITERATIONS * 2;
        auto   stats       = memory::calculate_bandwidth(total_bytes, elapsed_ms);

        printf("\nISA DWORDX4 Copy:\n");
        stats.print();
    }

    // Cleanup
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_src);
    hipFree(d_dst);
}

void test_coalesced_access() {
    printf("\n=== Coalesced Access Test (Matrix Transpose) ===\n");

    const int    matrix_size  = 4096;
    const size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);

    float *d_src, *d_dst;
    hipMalloc(&d_src, matrix_bytes);
    hipMalloc(&d_dst, matrix_bytes);

    // Initialize with pattern
    hipMemset(d_src, 1, matrix_bytes);

    dim3 block(32, 32);
    dim3 grid(matrix_size / 32, matrix_size / 32);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int i = 0; i < 10; i++) {
        coalesced_transpose_kernel<<<grid, block>>>(d_dst, d_src, matrix_size, matrix_size);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float elapsed_ms;
    hipEventElapsedTime(&elapsed_ms, start, stop);

    size_t total_bytes = matrix_bytes * 10 * 2;  // 10 iterations, read + write
    auto   stats       = memory::calculate_bandwidth(total_bytes, elapsed_ms);

    printf("Matrix size: %dx%d\n", matrix_size, matrix_size);
    stats.print();

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_src);
    hipFree(d_dst);
}

void test_lds_double_buffering() {
    printf("\n=== LDS Double Buffering Test ===\n");

    const size_t test_size = 16 * 1024 * 1024 / sizeof(float);  // 16 MB
    const int    tile_size = 1024;                              // Elements per tile

    float *d_src, *d_dst;
    hipMalloc(&d_src, test_size * sizeof(float));
    hipMalloc(&d_dst, test_size * sizeof(float));

    // Initialize
    hipMemset(d_src, 1, test_size * sizeof(float));

    const int block_size = 256;
    const int grid_size  = 60;  // One block per CU

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    lds_double_buffer_kernel<tile_size><<<grid_size, block_size>>>(d_dst, d_src, test_size);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float elapsed_ms;
    hipEventElapsedTime(&elapsed_ms, start, stop);

    size_t total_bytes = test_size * sizeof(float) * 2;
    auto   stats       = memory::calculate_bandwidth(total_bytes, elapsed_ms);

    printf("Tile size: %d elements\n", tile_size);
    printf("LDS usage: %zu KB per tile\n", tile_size * sizeof(float) * 2 / 1024);
    stats.print();

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_src);
    hipFree(d_dst);
}

// ================================
// Main Test Runner
// ================================

int main() {
    // Check for GFX906 device
    int device_count;
    hipGetDeviceCount(&device_count);

    if (device_count == 0) {
        printf("No HIP devices found!\n");
        return 1;
    }

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);

    printf("Device: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);

    // Check if this is GFX906
    if (props.gcnArch != 906) {
        printf("Warning: Not running on GFX906 (Vega 20nm)\n");
        printf("Current architecture: gfx%d\n", props.gcnArch);
    }

    printf("\nTheoretical Peak Bandwidth: %.1f GB/s\n", (float) GFX906_HBM2_BANDWIDTH);
    printf("Target Sustained Bandwidth: 900+ GB/s\n");

    // Run tests
    test_memory_bandwidth();
    test_coalesced_access();
    test_lds_double_buffering();

    printf("\n=== Test Complete ===\n");

    return 0;
}

#else  // !GGML_HIP_GFX906_OPTIMIZED

#    include <stdio.h>

int main() {
    printf("GFX906 optimizations not enabled. Build with -DGGML_HIP_GFX906_OPTIMIZED=ON\n");
    return 1;
}

#endif  // GGML_HIP_GFX906_OPTIMIZED
