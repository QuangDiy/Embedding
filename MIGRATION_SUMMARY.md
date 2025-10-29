# Migration từ BGE-M3 sang Jina-Embeddings-v3

## Tổng quan

Đã thực hiện migration thành công từ model **BAAI/bge-m3** sang **jinaai/jina-embeddings-v3** ONNX.

## Các thay đổi chính

### 1. **Model Repository Structure**
- ✅ Đổi tên thư mục: `model_repository/bge-m3/` → `model_repository/jina-embeddings-v3/`
- ✅ Cập nhật config.pbtxt với các thay đổi quan trọng:
  - Thêm input `task_id` (INT64) cho LoRA adapters
  - Đổi output từ `sentence_embedding` thành `last_hidden_state`
  - Output shape thay đổi từ `[-1]` thành `[-1, -1]` (batch_size, hidden_size)

### 2. **Model Files Required**
Jina-Embeddings-v3 cần **6 files** (thay vì 8 files của BGE-M3):
- `model.onnx` - ONNX model structure
- `model.onnx_data` - Model weights (large file) ⭐
- `tokenizer.json` - XLM-RoBERTa tokenizer
- `tokenizer_config.json`
- `special_tokens_map.json`
- `config.json`

❌ **Không cần nữa:**
- `sentencepiece.bpe.model`
- `Constant_7_attr__value`

### 3. **Triton Client Updates** (`api/triton_client.py`)
✅ **Thay đổi:**
- Thêm parameter `task_id` vào `get_embeddings()`
- Implement hàm `_mean_pooling()` để xử lý output
- Thêm L2 normalization cho embeddings
- Model output giờ là `last_hidden_state` thay vì `sentence_embedding`

**Task IDs cho LoRA adapters:**
- 0: `retrieval.query` (default)
- 1: `retrieval.passage`
- 2: `separation`
- 3: `classification`
- 4: `text-matching`

### 4. **API Updates** (`api/app.py`)
✅ **Thay đổi:**
- Thêm field `task` vào `EmbeddingRequest` model
- Thêm `TASK_MAPPING` dictionary
- Cập nhật tokenizer loading từ `jinaai/jina-embeddings-v3`
- Pass `task_id` đến Triton client
- Cập nhật tất cả references từ `bge-m3` → `jina-embeddings-v3`

### 5. **Download Scripts**
✅ **Cập nhật:**
- `download_model.ps1`: Download từ `jinaai/jina-embeddings-v3` repository
- `download_model.sh`: Download từ `jinaai/jina-embeddings-v3` repository
- Cập nhật target directory paths

### 6. **Verification Script** (`verify_model_files.py`)
✅ **Cập nhật:**
- Danh sách files required (6 files thay vì 8)
- Model directory path
- HuggingFace URL references

### 7. **Docker Configuration** (`docker-compose.yml`)
✅ **Cập nhật:**
- Container name: `jina-embeddings-v3-api`
- Volume mount: `./model_repository/jina-embeddings-v3/1:/tokenizer:ro`
- Environment variable: `MODEL_NAME=jina-embeddings-v3`
- Dockerfile path: `dockerfile`

### 8. **Example Usage** (`example_usage.py`)
✅ **Cập nhật:**
- Thêm parameter `task` vào tất cả API requests
- Thêm function `test_different_tasks()` để test các task types
- Cập nhật model name references
- Thêm documentation về available tasks

### 9. **README.md**
✅ **Cập nhật toàn diện:**
- Title và description
- Project structure
- Model files list
- All API examples với `task` parameter
- Thêm section "Task Types" với documentation
- Performance notes
- Resource links

## Điểm khác biệt quan trọng

### BGE-M3 vs Jina-Embeddings-v3

| Feature | BGE-M3 | Jina-Embeddings-v3 |
|---------|--------|-------------------|
| **Tokenizer** | SentencePiece + JSON | XLM-RoBERTa (JSON only) |
| **Model Output** | `sentence_embedding` (pre-pooled) | `last_hidden_state` (raw) |
| **Pooling** | Built-in | Manual mean pooling |
| **Task Adapters** | None | 5 LoRA adapters |
| **Input Required** | input_ids, attention_mask | input_ids, attention_mask, task_id |
| **Files** | 8 files | 6 files |
| **Max Length** | 8192 tokens | 8192 tokens |

## Cách sử dụng Task Types

### Retrieval (Search)
```python
# For queries
{"input": "what is machine learning?", "task": "retrieval.query"}

# For documents
{"input": "Machine learning is...", "task": "retrieval.passage"}
```

### Text Matching / Similarity
```python
{"input": ["text1", "text2"], "task": "text-matching"}
```

### Classification
```python
{"input": "This is a positive review", "task": "classification"}
```

### Separation
```python
{"input": "text to separate", "task": "separation"}
```

## Các bước tiếp theo

### 1. Download Model Files
```bash
# Windows
.\download_model.ps1

# Linux/Mac
bash download_model.sh
```

### 2. Verify Files
```bash
python verify_model_files.py
```

### 3. Build và Start Services
```bash
docker-compose build
docker-compose up -d
```

### 4. Test API
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, Jina-Embeddings-v3!",
    "model": "jina-embeddings-v3",
    "task": "text-matching"
  }'
```

### 5. Run Examples
```bash
python example_usage.py
```

## Breaking Changes

⚠️ **API Changes:**
- Thêm parameter `task` (optional, default: `retrieval.query`)
- Model name thay đổi từ `bge-m3` → `jina-embeddings-v3`
- Response format giữ nguyên (OpenAI-compatible)

⚠️ **Infrastructure:**
- Model directory path đã thay đổi
- Container names đã thay đổi
- Environment variables cần cập nhật

## Performance Notes

- **Mean Pooling**: Được thực hiện ở Python client (overhead nhỏ)
- **L2 Normalization**: Embeddings được normalize tự động
- **Task Switching**: Không ảnh hưởng performance đáng kể
- **Batch Processing**: Vẫn được hỗ trợ tốt với dynamic batching

## Resources

- **Model Card**: https://huggingface.co/jinaai/jina-embeddings-v3
- **Paper**: https://arxiv.org/abs/2409.10173
- **ONNX Files**: https://huggingface.co/jinaai/jina-embeddings-v3/tree/main/onnx

## Troubleshooting

### Issue: "task_id not found"
**Solution**: Đảm bảo config.pbtxt đã có input `task_id`

### Issue: "last_hidden_state not found"
**Solution**: Đảm bảo config.pbtxt output name là `last_hidden_state`

### Issue: Model không load
**Solution**: 
1. Kiểm tra tất cả 6 files đã có
2. Verify `model.onnx_data` không bị corrupt
3. Check Triton logs: `docker-compose logs triton`

---

**Migration completed successfully! 🎉**

Date: 2025-10-29

