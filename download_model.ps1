# Script to download Jina-Embeddings-v3 ONNX model from HuggingFace
# Requirements: git-lfs

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Download Jina-Embeddings-v3 ONNX Model Files" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check git-lfs
try {
    git lfs version | Out-Null
} catch {
    Write-Host "❌ Error: git-lfs is not installed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Download and install git-lfs from:"
    Write-Host "  https://git-lfs.github.com/"
    Write-Host ""
    exit 1
}

$TARGET_DIR = "model_repository\jina-embeddings-v3\1"

Write-Host "Target directory: $TARGET_DIR"
Write-Host ""

# Create temp directory
$TEMP_DIR = New-Item -ItemType Directory -Path ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), [System.IO.Path]::GetRandomFileName()))
Write-Host "Downloading to temp directory: $TEMP_DIR"
Write-Host ""

Push-Location $TEMP_DIR

# Clone with sparse checkout
Write-Host "Cloning repository (sparse checkout)..." -ForegroundColor Yellow
git clone --depth 1 --filter=blob:none --sparse https://huggingface.co/jinaai/jina-embeddings-v3

Set-Location jina-embeddings-v3

# Checkout only onnx directory
git sparse-checkout set onnx

# Pull LFS files
Write-Host ""
Write-Host "Downloading LFS files (this may take a while)..." -ForegroundColor Yellow
git lfs pull --include "onnx/*"

# Copy files
Write-Host ""
Write-Host "Copying files to $TARGET_DIR..." -ForegroundColor Yellow

$sourceDir = "onnx"
Get-ChildItem $sourceDir | ForEach-Object {
    Copy-Item $_.FullName -Destination (Join-Path $PSScriptRoot $TARGET_DIR) -Force
    Write-Host "  ✓ $($_.Name)" -ForegroundColor Green
}

# Cleanup
Pop-Location
Remove-Item -Recurse -Force $TEMP_DIR

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Download completed!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Verify
python verify_model_files.py

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. docker-compose build"
Write-Host "  2. docker-compose up -d"
Write-Host ""

