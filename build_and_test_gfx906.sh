#!/bin/bash

# Build and test script for GFX906 memory optimizations
# This script builds the project with GFX906 optimizations and runs memory tests

set -e

echo "=== Building with GFX906 Memory Optimizations ==="

# Clean previous build
rm -rf build_gfx906
mkdir -p build_gfx906
cd build_gfx906

# Configure with GFX906 optimizations
echo "Configuring CMake..."
cmake .. \
    -DGGML_HIP=ON \
    -DGGML_HIP_GFX906_OPTIMIZED=ON \
    -DAMDGPU_TARGETS=gfx906 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=ON

# Build
echo "Building..."
cmake --build . --config Release -j$(nproc)

# Run the memory optimization test if it exists
if [ -f "ggml/src/ggml-cuda/test-gfx906-memory" ]; then
    echo ""
    echo "=== Running GFX906 Memory Optimization Tests ==="
    ./ggml/src/ggml-cuda/test-gfx906-memory
else
    echo "Warning: test-gfx906-memory not found. Checking for alternative locations..."
    find . -name "test-gfx906-memory" -type f -executable 2>/dev/null | while read test_exe; do
        echo "Found test at: $test_exe"
        echo "Running..."
        $test_exe
    done
fi

# Run standard tests
echo ""
echo "=== Running Standard Tests ==="
ctest -L memory || true
ctest -L optimization || true

echo ""
echo "=== Build and Test Complete ==="
echo "To manually run the memory test:"
echo "  ./build_gfx906/ggml/src/ggml-cuda/test-gfx906-memory"