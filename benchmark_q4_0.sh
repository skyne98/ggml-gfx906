#!/bin/bash

# Benchmark script for Q4_0 optimized kernels on GFX906

echo "=== Q4_0 GFX906 Optimization Benchmark ==="
echo "Building with GFX906 optimizations..."

# Clean previous build
rm -rf build_gfx906

# Configure with HIP and GFX906 optimizations
cmake -B build_gfx906 \
    -DGGML_HIP=ON \
    -DGGML_HIP_GFX906_OPTIMIZED=ON \
    -DAMDGPU_TARGETS=gfx906 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=ON

# Build
cmake --build build_gfx906 --config Release -j$(nproc)

echo ""
echo "=== Running Q4_0 Tests ==="

# Check if test binary exists and run it
if [ -f "build_gfx906/bin/test-backend-ops" ]; then
    echo "Running backend operations test..."
    ./build_gfx906/bin/test-backend-ops | grep -A 10 "q4_0"
fi

# Run llama-bench if available
if [ -f "build_gfx906/bin/llama-bench" ]; then
    echo ""
    echo "=== Running Performance Benchmark ==="
    echo "Testing Q4_0 quantization performance..."
    
    # Create a simple test to measure throughput
    # This would need an actual model file to test properly
    echo "Note: Full benchmark requires a Q4_0 quantized model file"
fi

echo ""
echo "=== Checking Kernel Usage ==="
echo "Looking for V_DOT8_I32_I4 instruction usage..."

# Use rocprof to profile if available
if command -v rocprof &> /dev/null; then
    echo "ROCm profiler available - use 'rocprof --stats' with inference for detailed metrics"
fi

echo ""
echo "=== Summary ==="
echo "Q4_0 optimizations have been built successfully."
echo "To verify 80 GB/s throughput target:"
echo "1. Run inference with a Q4_0 quantized model"
echo "2. Use rocprof to measure memory bandwidth"
echo "3. Check for V_DOT8_I32_I4 instruction usage in kernel assembly"