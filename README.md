# Jina-Embeddings-v3 ONNX Embedding Service with Triton Server

Deploy jinaai/jina-embeddings-v3 ONNX model on Triton Inference Server with an OpenAI-compatible embedding API, optimized for CPU.

## 📋 Requirements

- Docker & Docker Compose
- ONNX model files for jinaai/jina-embeddings-v3
- At least 8GB RAM available
- 10GB free disk space

## 🏗️ Project Structure

```
triton/
├── model_repository/
│   └── jina-embeddings-v3/
│       ├── 1/                     # ← PLACE ALL MODEL FILES HERE
│       │   ├── model.onnx         # ONNX model structure
│       │   ├── model.onnx_data    # Model weights (large file) ⭐ REQUIRED!
│       │   ├── tokenizer.json
│       │   ├── tokenizer_config.json
│       │   ├── special_tokens_map.json
│       │   └── config.json
│       └── config.pbtxt
├── api/
│   ├── app.py                     # Main API application
│   ├── triton_client.py           # Triton client
│   └── requirements.txt
├── docker-compose.yml
├── Dockerfile.api
└── README.md
```

## ⚠️ Important: Model Files

Jina-Embeddings-v3 ONNX model consists of **6 FILES** (not just model.onnx!):

1. ✅ `model.onnx` - ONNX model structure
2. ✅ `model.onnx_data` - Model weights (large file) **← CRITICAL!**
3. ✅ `tokenizer.json` - Tokenizer vocabulary
4. ✅ `tokenizer_config.json` - Tokenizer configuration
5. ✅ `special_tokens_map.json` - Special tokens
6. ✅ `config.json` - Model configuration

**All 6 files are REQUIRED!**

## 🚀 Quick Start

### Step 1: Download Model Files

**Option A - Automated (Recommended):**

```bash
# Linux/macOS
bash download_model.sh

# Windows PowerShell
.\download_model.ps1
```

**Option B - Manual:**
1. Visit: https://huggingface.co/jinaai/jina-embeddings-v3/tree/main/onnx
2. Download all 6 files
3. Place them in `model_repository/jina-embeddings-v3/1/`

### Step 2: Verify Model Files

```bash
python verify_model_files.py
```

You should see: `✅ ALL FILES ARE READY!`

### Step 3: Start Services

**Quick Start (Recommended):**

```bash
# Linux/macOS
bash quick_start.sh

# Windows PowerShell
.\quick_start.ps1
```

**Manual Start:**

```bash
# Build Docker images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

Services will be available at:
- **API Service**: http://localhost:8000
- **Triton HTTP**: http://localhost:8002
- **Triton gRPC**: http://localhost:8001
- **Metrics**: http://localhost:8003

### Step 4: Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Check Triton Server
curl http://localhost:8002/v2/health/ready

# Test embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "model": "jina-embeddings-v3", "task": "text-matching"}'

# Run example script
python example_usage.py
```

## 📡 API Usage

### OpenAI-Compatible Endpoint

The API is compatible with OpenAI embeddings format.

**Endpoint:** `POST /v1/embeddings`

### Example with cURL

```bash
# Single text
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "jina-embeddings-v3",
    "task": "text-matching"
  }'

# Multiple texts (batching)
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello, world!", "How are you?", "Goodbye!"],
    "model": "jina-embeddings-v3",
    "task": "text-matching"
  }'
```

### Example with Python (requests)

```python
import requests

url = "http://localhost:8000/v1/embeddings"

# Single text
response = requests.post(
    url,
    json={
        "input": "Hello, world!",
        "model": "jina-embeddings-v3",
        "task": "text-matching"
    }
)

result = response.json()
embedding = result['data'][0]['embedding']
print(f"Embedding dimension: {len(embedding)}")

# Multiple texts (batching)
response = requests.post(
    url,
    json={
        "input": [
            "What is machine learning?",
            "Triton Inference Server is awesome",
            "FastAPI makes APIs easy"
        ],
        "model": "jina-embeddings-v3",
        "task": "text-matching"
    }
)

result = response.json()
print(f"Generated {len(result['data'])} embeddings")
for i, item in enumerate(result['data']):
    print(f"Text {i}: dimension = {len(item['embedding'])}")
```

### Example with OpenAI Python Client

```python
from openai import OpenAI

# Point to local service using base_url
client = OpenAI(
    api_key="dummy-key",  # API key not required for local
    base_url="http://localhost:8000/v1"
)

response = client.embeddings.create(
    input="Hello, world!",
    model="jina-embeddings-v3"
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],
      "index": 0
    }
  ],
  "model": "jina-embeddings-v3",
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3
  }
}
```

### Task Types

Jina-Embeddings-v3 supports 5 different task types via LoRA adapters:

- **`retrieval.query`** (default): For search queries
- **`retrieval.passage`**: For documents to be retrieved
- **`text-matching`**: For similarity/matching tasks
- **`classification`**: For classification tasks
- **`separation`**: For separation tasks

**Example with task:**

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is machine learning?",
    "model": "jina-embeddings-v3",
    "task": "retrieval.query"
  }'
```

### Additional Endpoints

```bash
# List available models
curl http://localhost:8000/v1/models

# Health check
curl http://localhost:8000/health

# Root endpoint
curl http://localhost:8000/
```

## 🔧 Configuration

### Dynamic Batching

Triton is configured with dynamic batching in `config.pbtxt`:
- Max batch size: 32
- Preferred batch sizes: 4, 8, 16
- Max queue delay: 100 microseconds

To adjust, edit `model_repository/jina-embeddings-v3/config.pbtxt`:

```protobuf
max_batch_size: 16  # Reduce if running out of memory

dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 100
}
```

### CPU Optimization

The service is optimized for CPU:
- ONNX Runtime CPU execution provider
- Instance count: 1 (increase for more CPU cores)

To increase instances, edit `config.pbtxt`:

```protobuf
instance_group [
  {
    count: 2  # Increase for more parallelism
    kind: KIND_CPU
  }
]
```

### Environment Variables

In `docker-compose.yml`, you can configure:

```yaml
environment:
  - TRITON_URL=triton:8000
  - MODEL_NAME=jina-embeddings-v3
  - TOKENIZER_PATH=/tokenizer
```

## 🐛 Troubleshooting

### ❌ Model Fails to Load

**Check Triton logs:**
```bash
docker-compose logs triton
```

**Verify all model files exist:**
```bash
python verify_model_files.py
```

**Common issues:**
- Missing `model.onnx_data` file (2.27 GB)
- Missing tokenizer files
- Input/output names in `config.pbtxt` don't match ONNX model

### ❌ API Returns 500 Error

**Check API logs:**
```bash
docker-compose logs api
```

**Test Triton directly:**
```bash
curl http://localhost:8002/v2/models/jina-embeddings-v3/ready
```

**Common issues:**
- Triton server not ready
- Tokenizer files missing
- Model not loaded correctly

### ❌ "External Data Not Found" Error

This means `model.onnx_data` file is missing. This is the main 2.27 GB file containing model weights.

**Solution:**
```bash
# Verify the file exists and is correct size
ls -lh model_repository/jina-embeddings-v3/1/model.onnx_data
# Should show the large weights file
```

### ❌ Out of Memory

**Solution:** Reduce batch size in `config.pbtxt`:
```protobuf
max_batch_size: 16  # Reduce from 32 to 16 or 8
```

### ❌ Slow First Request

First request may be slow due to:
- Tokenizer loading
- Model warm-up
- ONNX Runtime optimization

This is normal - subsequent requests will be faster.

## 📊 Performance Monitoring

### Triton Metrics

Triton provides Prometheus metrics at port 8003:

```bash
# Get all metrics
curl http://localhost:8003/metrics

# Key metrics:
# - nv_inference_request_success
# - nv_inference_request_failure
# - nv_inference_queue_duration_us
# - nv_inference_compute_infer_duration_us
```

### Model Statistics

```bash
# Get model stats
curl http://localhost:8002/v2/models/jina-embeddings-v3/stats
```

### Expected Performance (CPU)

- Single text: ~50-200ms
- Batch of 8: ~100-400ms
- Batch of 32: ~300-800ms

*Actual performance depends on CPU, text length, and configuration*

## 🛑 Stop Services

```bash
# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop and remove images
docker-compose down --rmi all
```

## 🔐 Production Deployment

### Security Considerations

**⚠️ Default setup has NO authentication!**

For production deployment, implement:

1. **Authentication** - Add API keys or JWT tokens
2. **HTTPS/TLS** - Use reverse proxy (nginx, traefik)
3. **Rate Limiting** - Prevent abuse
4. **Firewall Rules** - Restrict network access
5. **Monitoring** - Setup alerts and logging

### Example: Add Authentication

```python
# api/app.py
from fastapi import Security, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return credentials

@app.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    token = Depends(verify_token)
):
    # ... rest of code
```

### Example: Add Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/embeddings")
@limiter.limit("100/minute")
async def create_embeddings(request: Request, ...):
    # ... code
```

### Using Reverse Proxy (nginx)

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📝 Notes

1. **Task Selection**: Jina-Embeddings-v3 uses LoRA adapters for different tasks. Choose the appropriate task type for your use case.

2. **Tokenization**: Uses XLM-RoBERTa tokenizer from HuggingFace. The tokenizer is automatically loaded from model files.

3. **Mean Pooling**: The model outputs `last_hidden_state` which is automatically mean-pooled and normalized by the client.

4. **Model Input/Output**: Ensure input/output names and shapes in `config.pbtxt` match your actual ONNX model.

5. **Memory**: Jina-Embeddings-v3 is a large model - ensure at least 8GB RAM available (16GB+ recommended for production).

6. **Batch Processing**: Use batching for better throughput. Dynamic batching in Triton automatically optimizes batch sizes.

## 🔗 Resources

- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
- [Jina-Embeddings-v3 Model](https://huggingface.co/jinaai/jina-embeddings-v3)
- [Jina-Embeddings-v3 Paper](https://arxiv.org/abs/2409.10173)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)
- [ONNX Runtime](https://onnxruntime.ai/)

## 📄 License

MIT License - Free to use and modify for your needs.

---

## Quick Reference Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Check health
curl http://localhost:8000/health

# Test embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "jina-embeddings-v3", "task": "text-matching"}'

# Verify model files
python verify_model_files.py

# Run examples
python example_usage.py

# Monitor resources
docker stats
```

---

**Last updated:** 2025-10-29
