#!/bin/bash

# Script to download Jina-Embeddings-v3 ONNX model from HuggingFace
# Requirements: git-lfs

set -e

echo "=========================================="
echo "Download Jina-Embeddings-v3 ONNX Model Files"
echo "=========================================="
echo ""

# Check git-lfs
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Error: git-lfs is not installed"
    echo ""
    echo "Install git-lfs:"
    echo "  Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  macOS: brew install git-lfs"
    echo "  Windows: Download from https://git-lfs.github.com/"
    echo ""
    exit 1
fi

TARGET_DIR="model_repository/jina-embeddings-v3/1"

echo "Target directory: $TARGET_DIR"
echo ""

# Create temp directory
TEMP_DIR=$(mktemp -d)
echo "Downloading to temp directory: $TEMP_DIR"
echo ""

cd "$TEMP_DIR"

# Clone with sparse checkout
echo "Cloning repository (sparse checkout)..."
git clone --depth 1 --filter=blob:none --sparse https://huggingface.co/jinaai/jina-embeddings-v3

cd jina-embeddings-v3

# Checkout only onnx directory
git sparse-checkout set onnx

# Pull LFS files
echo ""
echo "Downloading LFS files (this may take a while)..."
git lfs pull --include "onnx/*"

# Copy files
echo ""
echo "Copying files to $TARGET_DIR..."

cd onnx
for file in *; do
    cp "$file" "$OLDPWD/$OLDPWD/$TARGET_DIR/"
    echo "  ✓ $file"
done

# Cleanup
cd "$OLDPWD/$OLDPWD"
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Download completed!"
echo "=========================================="
echo ""

# Verify
python3 verify_model_files.py

echo ""
echo "Next steps:"
echo "  1. docker-compose build"
echo "  2. docker-compose up -d"
echo ""

