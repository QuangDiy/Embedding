# Quick start script for Jina-Embeddings-v3 Embedding Service (Windows PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Jina-Embeddings-v3 Embedding Service - Quick Start" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if all required files exist
Write-Host "Checking model files..." -ForegroundColor Yellow
$verifyResult = python verify_model_files.py 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Error: Missing required model files!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run the verification script to see details:"
    Write-Host "  python verify_model_files.py"
    Write-Host ""
    exit 1
}

Write-Host "✓ All model files found" -ForegroundColor Green
Write-Host ""

# Check Docker
try {
    docker --version | Out-Null
    Write-Host "✓ Docker is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Docker is not installed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Check Docker Compose
try {
    docker-compose --version | Out-Null
    Write-Host "✓ Docker Compose is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Docker Compose is not installed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Build images
Write-Host "Building Docker images..." -ForegroundColor Yellow
docker-compose build

Write-Host ""
Write-Host "✓ Images built successfully" -ForegroundColor Green
Write-Host ""

# Start services
Write-Host "Starting services..." -ForegroundColor Yellow
docker-compose up -d

Write-Host ""
Write-Host "✓ Services started" -ForegroundColor Green
Write-Host ""

# Wait for services to be ready
Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check Triton
Write-Host -NoNewline "Checking Triton Server... "
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8002/v2/health/ready" -UseBasicParsing -TimeoutSec 5
    Write-Host "✓" -ForegroundColor Green
} catch {
    Write-Host "❌" -ForegroundColor Red
    Write-Host "Triton Server is not ready. Check logs with: docker-compose logs triton"
}

# Check API
Write-Host -NoNewline "Checking API Service... "
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "✓" -ForegroundColor Green
} catch {
    Write-Host "❌" -ForegroundColor Red
    Write-Host "API Service is not ready. Check logs with: docker-compose logs api"
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Services are running!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "API Endpoint: http://localhost:8000/v1/embeddings"
Write-Host "Health Check: http://localhost:8000/health"
Write-Host "Triton HTTP:  http://localhost:8002"
Write-Host ""
Write-Host "To test the API:"
Write-Host "  python example_usage.py"
Write-Host ""
Write-Host "To view logs:"
Write-Host "  docker-compose logs -f"
Write-Host ""
Write-Host "To stop services:"
Write-Host "  docker-compose down"
Write-Host ""

