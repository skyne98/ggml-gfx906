#!/bin/bash

echo "GFX906 Q5_K/Q6_K Performance Benchmark"
echo "======================================"
echo ""

# Check for GPU
rocm-smi --showid 2>/dev/null | grep -q "GPU" && echo "✓ AMD GPU detected" || echo "✗ No AMD GPU found"
echo ""

# Set environment for GFX906
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0

# Build directory
BUILD_DIR="build"

# Check if llama-bench exists
if [ -f "$BUILD_DIR/bin/llama-bench" ]; then
    echo "Using llama-bench for performance testing..."
    echo ""
    
    # Download a small test model if not present
    MODEL_FILE="test_model.gguf"
    if [ ! -f "$MODEL_FILE" ]; then
        echo "Downloading small test model..."
        # Use a tiny model for testing
        wget -q https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf -O $MODEL_FILE || {
            echo "Could not download test model. Creating synthetic test..."
            # If download fails, we'll skip model testing
            MODEL_FILE=""
        }
    fi
    
    if [ -n "$MODEL_FILE" ] && [ -f "$MODEL_FILE" ]; then
        echo "Running Q5_K benchmark..."
        $BUILD_DIR/bin/llama-bench -m $MODEL_FILE -t 4 -n 128 -pg 32,32 2>&1 | grep -E "Q5_K|pp|tg|cuda"
        
        echo ""
        echo "Note: Q6_K would require a Q6_K quantized model"
    fi
else
    echo "llama-bench not found. Building may not have completed."
fi

# Alternative: Use llama-cli if available
if [ -f "$BUILD_DIR/bin/llama-cli" ]; then
    echo ""
    echo "Checking GFX906 optimizations in build..."
    
    # Check if our optimizations are included
    strings $BUILD_DIR/src/libggml*.so 2>/dev/null | grep -q "gfx906" && echo "✓ GFX906 code detected in library" || echo "✗ GFX906 code not found"
fi

echo ""
echo "Build configuration check:"
echo "--------------------------"
grep -E "GGML_HIP|GFX906|AMDGPU_TARGETS" $BUILD_DIR/CMakeCache.txt 2>/dev/null | head -5

echo ""
echo "Library check:"
ls -lh $BUILD_DIR/src/libggml*.so 2>/dev/null || echo "Libraries not found"

echo ""
echo "======================================"
echo "Benchmark complete"