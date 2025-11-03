# Jina AI ONNX Service with Triton Server

Deploy Jina AI models (Embeddings v3 + Reranker v2) on Triton Inference Server with OpenAI-compatible APIs.

## API Key Configuration

### Sử dụng API Key trong Requests

Service sử dụng **Bearer Token Authentication**:

**Authorization Header:**
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -d '{"input": "Hello, world!", "model": "jina-embeddings-v3"}'
```

### Sử dụng trong Code

**Python với requests:**
```python
import requests

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-api-key-here"
}

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    headers=headers,
    json={
        "input": "Hello, world!",
        "model": "jina-embeddings-v3"
    }
)
```

**Python với OpenAI SDK:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="your-secret-api-key-here",
    base_url="http://localhost:8000/v1"
)

response = client.embeddings.create(
    input="Hello, world!",
    model="jina-embeddings-v3"
)
```

### Lưu ý Bảo mật

⚠️ **QUAN TRỌNG:**
- **KHÔNG** commit file `.env` vào git
- Thêm `.env` vào `.gitignore`
- Sử dụng API key mạnh (ít nhất 32 ký tự ngẫu nhiên)
- Xoay vòng API key định kỳ
- Sử dụng HTTPS trong production

**Tạo API key mạnh:**
```bash
# Linux/Mac
openssl rand -hex 32

# hoặc
python3 -c "import secrets; print(secrets.token_hex(32))"
```

## Quick Start

### Build & Run with Docker Compose

```bash
# Build and start services
docker compose up --build -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Build & Run with Docker (Single Container)

```bash
# Build image
docker build -t embedding-service .

# Run container
docker run -p 8000:8000 embedding-service
```

## 🔧 Available Services

After starting, services will be available at:
- **API Service**: http://localhost:8000
- **Triton HTTP**: http://localhost:8002
- **Triton gRPC**: http://localhost:8001
- **Metrics**: http://localhost:8003

## Test the Service

**Không có API key (mặc định):**
```bash
# Health check (không cần API key)
curl http://localhost:8000/health

# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "model": "jina-embeddings-v3", "task": "text-matching"}'

# Rerank
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "documents": ["AI and ML", "Cooking recipes"], "model": "jina-reranker-v2"}'
```

**Với API key (Bearer Token):**
```bash
# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"input": "Hello, world!", "model": "jina-embeddings-v3"}'

# Rerank
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"query": "machine learning", "documents": ["AI and ML", "Cooking recipes"], "model": "jina-reranker-v2"}'

# Test script
python test_api.py
```

## API Endpoints

### Embeddings API (OpenAI-compatible)

**POST** `/v1/embeddings`

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    json={
        "input": "Hello, world!",
        "model": "jina-embeddings-v3",
        "task": "text-matching"
    }
)
print(response.json())
```

**Task types:** `retrieval.query`, `retrieval.passage`, `text-matching`, `classification`, `separation`

### Rerank API

**POST** `/v1/rerank`

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/rerank",
    json={
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a branch of AI.",
            "Python is a programming language.",
            "Deep learning is a subset of ML."
        ],
        "model": "jina-reranker-v2",
        "top_n": 2
    }
)
print(response.json())
```

## Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific tests
pytest tests/test_services.py -v
```

## 🔗 Resources

- [Jina-Embeddings-v3 Model](https://huggingface.co/jinaai/jina-embeddings-v3)
- [Jina-Reranker-v2 Model](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
- [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)
