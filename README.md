# Jina AI ONNX Service with Triton Server

Deploy Jina AI models (Embeddings v3 + Reranker v2) on Triton Inference Server with OpenAI-compatible APIs.

## Quick Start

### Build & Run with Docker Compose

```bash
# Build and start services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
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

## 🧪 Test the Service

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "model": "jina-embeddings-v3", "task": "text-matching"}'

curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "documents": ["AI and ML", "Cooking recipes"], "model": "jina-reranker-v2"}'

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

## 🔗 Resources

- [Jina-Embeddings-v3 Model](https://huggingface.co/jinaai/jina-embeddings-v3)
- [Jina-Reranker-v2 Model](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
- [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)
