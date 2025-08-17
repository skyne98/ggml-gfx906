#!/bin/bash

# Benchmark script for GFX906 GEMM kernels
# Tests the optimized 128x128x32 tiled GEMM implementation

set -e

echo "=== GFX906 GEMM Benchmark Script ==="
echo "Building with GFX906 optimizations..."

# Clean previous build
rm -rf build_gemm_test
mkdir -p build_gemm_test
cd build_gemm_test

# Configure with GFX906 optimizations
cmake .. \
    -DGGML_HIP=ON \
    -DGGML_HIP_GFX906_OPTIMIZED=ON \
    -DAMDGPU_TARGETS=gfx906 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=ON

# Build the test
make test-gemm-gfx906 -j$(nproc)

# Run the benchmark
echo ""
echo "=== Running GEMM Benchmark ==="
./tests/test-gemm-gfx906

# Check performance
echo ""
echo "=== Performance Analysis ==="
echo "Target: 4-5 TFLOPS for GFX906 (AMD Instinct MI50)"
echo "Expected efficiency: 60-75% of theoretical peak (6.6 TFLOPS FP32)"

cd ..

echo ""
echo "=== Benchmark Complete ==="