# Q4_0 Quantization Optimization for GFX906

## Overview

This document describes the optimized Q4_0 quantization kernels for AMD GFX906 (MI50/MI60) GPUs, implementing efficient 4-bit quantized operations using hardware-specific instructions.

## Key Optimizations

### 1. V_DOT8_I32_I4 Instruction

The core optimization leverages the GFX906's `V_DOT8_I32_I4` instruction, which performs an 8-way dot product on signed 4-bit integers in a single cycle:

```cpp
// Performs: result = sum(a[i] * b[i]) for i=0..7
// where a[i] and b[i] are 4-bit signed integers
asm volatile("v_dot8_i32_i4 %0, %1, %2, 0" : "=v"(result) : "v"(a), "v"(b));
```

This instruction provides:
- 8x throughput improvement over scalar operations
- Single-cycle execution for 8 multiply-accumulate operations
- Native support for Q4_0's 4-bit quantization format

### 2. Fused Dequantize-and-GEMV Kernel

Instead of separate dequantization and GEMV operations, we implement a fused kernel that:
- Loads quantized weights directly
- Performs dot products using V_DOT8_I32_I4
- Applies scaling factors in registers
- Minimizes HBM traffic

Benefits:
- Reduces memory bandwidth by ~87.5% (4-bit vs 32-bit)
- Eliminates intermediate storage
- Improves cache utilization

### 3. LDS Optimization

The kernel uses the 64KB Local Data Share (LDS) per CU for:
- **Input Vector Caching**: Reused across matrix rows
- **Scale Factor Storage**: Shared across workgroup

Memory access pattern:
```
HBM → LDS (cooperative load) → Registers (compute) → HBM (output)
```

### 4. Optimal Thread Configuration

- **Workgroup Size**: 256 threads (4 wavefronts)
- **Waves per CU**: 4-8 for optimal occupancy
- **Thread Mapping**: One thread per output element

This configuration ensures:
- Maximum hardware utilization
- Efficient memory coalescing
- Minimal synchronization overhead

## Performance Targets

| Metric | Target | Achieved |
|--------|---------|----------|
| Dequantization Throughput | 80 GB/s | TBD |
| GEMV Performance | 50+ GFLOPS | TBD |
| Memory Efficiency | >85% | TBD |

## Implementation Files

- `ggml/src/ggml-cuda/q4_0-gfx906.cuh`: Optimized kernel implementations
- `ggml/src/ggml-cuda/vecdotq.cuh`: Integration with vector dot product operations
- `ggml/tests/test-q4_0-gfx906.cpp`: Unit tests and benchmarks

## Building and Testing

### Build with GFX906 Optimizations

```bash
cmake -B build \
    -DGGML_HIP=ON \
    -DGGML_HIP_GFX906_OPTIMIZED=ON \
    -DAMDGPU_TARGETS=gfx906
cmake --build build --config Release
```

### Run Benchmarks

```bash
./benchmark_q4_0.sh
```

### Profile with ROCm

```bash
rocprof --stats ./build/bin/llama-cli -m model.q4_0.gguf -p "test" -n 100
```

## Hardware Requirements

- AMD GFX906 GPU (Instinct MI50/MI60)
- ROCm 5.0 or later
- HIP compiler with GFX906 support

## Technical Details

### Q4_0 Format

Each Q4_0 block contains:
- 32 quantized 4-bit values (16 bytes)
- 1 FP16 scale factor (2 bytes)
- Total: 18 bytes per 32 values

### Memory Layout

Values are packed as:
```
[v0:v1][v2:v3]...[v30:v31]
```
Where each pair is stored in one byte, with values offset by 8.

### Instruction Sequence

1. Load 8 packed 4-bit values (32 bits)
2. Load corresponding 8 int8 values from input
3. Execute V_DOT8_I32_I4 instruction
4. Accumulate results
5. Apply scaling at the end

## Future Optimizations

1. **Tensor Core Support**: Investigate using matrix core units
2. **Mixed Precision**: Combine with FP16 operations
3. **Kernel Fusion**: Integrate with other operators
4. **Dynamic Dispatch**: Runtime selection based on matrix size

## References

- AMD "Vega" 7nm ISA Documentation
- GCN Assembly Programming Guide
- ROCm Performance Tuning Guide