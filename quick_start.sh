#!/bin/bash

# Quick start script for Jina-Embeddings-v3 Embedding Service

set -e

echo "=========================================="
echo "Jina-Embeddings-v3 Embedding Service - Quick Start"
echo "=========================================="
echo ""

# Check if all required files exist
echo "Checking model files..."
if ! python3 verify_model_files.py > /dev/null 2>&1; then
    echo ""
    echo "❌ Error: Missing required model files!"
    echo ""
    echo "Please run the verification script to see details:"
    echo "  python3 verify_model_files.py"
    echo ""
    exit 1
fi

echo "✓ All model files found"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    exit 1
fi

echo "✓ Docker is installed"
echo ""

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed"
    exit 1
fi

echo "✓ Docker Compose is installed"
echo ""

# Build images
echo "Building Docker images..."
docker-compose build

echo ""
echo "✓ Images built successfully"
echo ""

# Start services
echo "Starting services..."
docker-compose up -d

echo ""
echo "✓ Services started"
echo ""

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check Triton
echo -n "Checking Triton Server... "
if curl -s http://localhost:8002/v2/health/ready > /dev/null; then
    echo "✓"
else
    echo "❌"
    echo "Triton Server is not ready. Check logs with: docker-compose logs triton"
fi

# Check API
echo -n "Checking API Service... "
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓"
else
    echo "❌"
    echo "API Service is not ready. Check logs with: docker-compose logs api"
fi

echo ""
echo "=========================================="
echo "Services are running!"
echo "=========================================="
echo ""
echo "API Endpoint: http://localhost:8000/v1/embeddings"
echo "Health Check: http://localhost:8000/health"
echo "Triton HTTP:  http://localhost:8002"
echo ""
echo "To test the API:"
echo "  python example_usage.py"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""

