from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import numpy as np
import logging
import time
from triton_client import TritonEmbeddingClient, TritonRerankerClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Jina AI API",
    description="OpenAI-compatible embedding and reranking API powered by Triton Inference Server",
    version="1.0.0"
)

triton_client = TritonEmbeddingClient(
    triton_url="triton:8000",
    model_name="jina-embeddings-v3"
)

triton_reranker_client = TritonRerankerClient(
    triton_url="triton:8000",
    model_name="jina-reranker-v2"
)


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    input: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    model: str = Field(default="jina-embeddings-v3", description="Model name")
    encoding_format: Optional[str] = Field(default="float", description="Encoding format (float or base64)")
    task: Optional[str] = Field(default="retrieval.query", description="Task type: retrieval.query, retrieval.passage, separation, classification, text-matching")
    user: Optional[str] = Field(default=None, description="User identifier")


class EmbeddingData(BaseModel):
    """Embedding data object"""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class RerankRequest(BaseModel):
    """OpenAI-compatible rerank request"""
    query: str = Field(..., description="The search query")
    documents: Union[List[str], List[dict]] = Field(..., description="List of documents to rerank")
    model: str = Field(default="jina-reranker-v2", description="Model name")
    top_n: Optional[int] = Field(default=None, description="Number of most relevant documents to return")
    return_documents: Optional[bool] = Field(default=True, description="Whether to return document text")


class RerankResult(BaseModel):
    """Single rerank result"""
    index: int
    relevance_score: float
    document: Optional[Union[str, dict]] = None


class RerankResponse(BaseModel):
    """OpenAI-compatible rerank response"""
    object: str = "list"
    data: List[RerankResult]
    model: str
    usage: dict


import os
from transformers import AutoTokenizer

TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "/tokenizer")
RERANKER_TOKENIZER_PATH = os.getenv("RERANKER_TOKENIZER_PATH", "/reranker_tokenizer")
tokenizer = None
reranker_tokenizer = None

def load_tokenizer():
    """Load tokenizer from local files"""
    global tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        logger.info(f"Tokenizer loaded from {TOKENIZER_PATH}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {TOKENIZER_PATH}: {e}")
        logger.info("Falling back to download from HuggingFace Hub...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
            logger.info("Tokenizer loaded from HuggingFace Hub")
        except Exception as e2:
            logger.error(f"Failed to load tokenizer from HuggingFace: {e2}")
            raise


def load_reranker_tokenizer():
    """Load reranker tokenizer from local files"""
    global reranker_tokenizer
    try:
        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_TOKENIZER_PATH)
        logger.info(f"Reranker tokenizer loaded from {RERANKER_TOKENIZER_PATH}")
    except Exception as e:
        logger.error(f"Failed to load reranker tokenizer from {RERANKER_TOKENIZER_PATH}: {e}")
        logger.info("Falling back to download from HuggingFace Hub...")
        try:
            reranker_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-reranker-v2-base-multilingual")
            logger.info("Reranker tokenizer loaded from HuggingFace Hub")
        except Exception as e2:
            logger.error(f"Failed to load reranker tokenizer from HuggingFace: {e2}")
            raise

TASK_MAPPING = {
    "retrieval.query": 0,
    "retrieval.passage": 1,
    "separation": 2,
    "classification": 3,
    "text-matching": 4
}

def prepare_inputs_for_triton(texts: List[str]) -> tuple:
    """
    Prepare inputs for Triton Server.
    
    Args:
        texts: List of texts
        
    Returns:
        tuple of (input_ids, attention_mask) as numpy arrays
    """
    global tokenizer
    
    if tokenizer is None:
        load_tokenizer()
    
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192, 
        return_tensors="np"
    )
    
    input_ids = encoded['input_ids'].astype(np.int64)
    attention_mask = encoded['attention_mask'].astype(np.int64)
    
    return input_ids, attention_mask


def prepare_rerank_inputs_for_triton(query: str, documents: List[str]) -> tuple:
    """
    Prepare query-document pairs for reranking.
    
    Args:
        query: Search query
        documents: List of documents to rerank
        
    Returns:
        tuple of (input_ids, attention_mask) as numpy arrays
    """
    global reranker_tokenizer
    
    if reranker_tokenizer is None:
        load_reranker_tokenizer()
    
    pairs = [[query, doc] for doc in documents]
    
    encoded = reranker_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )
    
    input_ids = encoded['input_ids'].astype(np.int64)
    attention_mask = encoded['attention_mask'].astype(np.int64)
    
    return input_ids, attention_mask


@app.on_event("startup")
async def startup_event():
    """Startup and connect to Triton Server"""
    try:
        load_tokenizer()
        logger.info("Tokenizer loaded successfully")
        
        load_reranker_tokenizer()
        logger.info("Reranker tokenizer loaded successfully")
        
        triton_client.connect()
        logger.info("Successfully connected to Triton Server for embeddings")
        
        triton_reranker_client.connect()
        logger.info("Successfully connected to Triton Server for reranking")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close connection on shutdown"""
    triton_client.close()
    triton_reranker_client.close()
    logger.info("Triton client connections closed")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Jina AI API (Embeddings + Reranking)",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    embedding_ready = triton_client.is_ready()
    reranker_ready = triton_reranker_client.is_ready()
    
    if embedding_ready and reranker_ready:
        return {
            "status": "healthy",
            "triton_server": "connected",
            "models": {
                "jina-embeddings-v3": "ready",
                "jina-reranker-v2": "ready"
            }
        }
    elif embedding_ready or reranker_ready:
        return {
            "status": "partial",
            "triton_server": "connected",
            "models": {
                "jina-embeddings-v3": "ready" if embedding_ready else "not_ready",
                "jina-reranker-v2": "ready" if reranker_ready else "not_ready"
            }
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton server is not ready"
        )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for text input (OpenAI-compatible endpoint)
    
    Args:
        request: EmbeddingRequest containing text input
        
    Returns:
        EmbeddingResponse with embeddings
    """
    try:
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        if not texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input cannot be empty"
            )
        
        input_ids, attention_mask = prepare_inputs_for_triton(texts)
        
        task_name = getattr(request, 'task', 'retrieval.query')
        task_id = TASK_MAPPING.get(task_name, 0)
        
        embeddings = triton_client.get_embeddings(input_ids, attention_mask, task_id)
        
        embedding_data = []
        for idx, emb in enumerate(embeddings):
            embedding_data.append(
                EmbeddingData(
                    embedding=emb.tolist(),
                    index=idx
                )
            )
        
        total_tokens = sum(len(text.split()) for text in texts)
        
        response = EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing embedding request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}"
        )


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "jina-embeddings-v3",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "jinaai",
                "permission": [],
                "root": "jina-embeddings-v3",
                "parent": None
            },
            {
                "id": "jina-reranker-v2",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "jinaai",
                "permission": [],
                "root": "jina-reranker-v2",
                "parent": None
            }
        ]
    }


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """
    Rerank documents based on relevance to a query (OpenAI-compatible endpoint)
    
    Args:
        request: RerankRequest containing query and documents
        
    Returns:
        RerankResponse with reranked documents and relevance scores
    """
    try:
        if not request.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Documents list cannot be empty"
            )
        
        doc_texts = []
        for doc in request.documents:
            if isinstance(doc, str):
                doc_texts.append(doc)
            elif isinstance(doc, dict):
                doc_texts.append(doc.get('text', doc.get('content', str(doc))))
            else:
                doc_texts.append(str(doc))
        
        input_ids, attention_mask = prepare_rerank_inputs_for_triton(request.query, doc_texts)
        
        scores = triton_reranker_client.get_rerank_scores(input_ids, attention_mask)
        
        results = []
        for idx, score in enumerate(scores):
            result = RerankResult(
                index=idx,
                relevance_score=float(score)
            )
            
            if request.return_documents:
                result.document = request.documents[idx]
            
            results.append(result)
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        if request.top_n is not None and request.top_n > 0:
            results = results[:request.top_n]
        
        response = RerankResponse(
            data=results,
            model=request.model,
            usage={
                "total_tokens": sum(len(doc.split()) for doc in doc_texts)
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing rerank request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rerank documents: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
