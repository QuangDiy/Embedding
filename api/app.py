from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import numpy as np
import logging
import time
from triton_client import TritonEmbeddingClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BGE-M3 Embedding API",
    description="OpenAI-compatible embedding API powered by Triton Inference Server",
    version="1.0.0"
)

# Initialize Triton client
triton_client = TritonEmbeddingClient(
    triton_url="triton:8000",
    model_name="bge-m3"
)


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    input: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    model: str = Field(default="bge-m3", description="Model name")
    encoding_format: Optional[str] = Field(default="float", description="Encoding format (float or base64)")
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


# Load tokenizer from local files (mounted from model repository)
import os
from transformers import AutoTokenizer

TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "/tokenizer")
tokenizer = None

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
            tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
            logger.info("Tokenizer loaded from HuggingFace Hub")
        except Exception as e2:
            logger.error(f"Failed to load tokenizer from HuggingFace: {e2}")
            raise

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
    
    # Tokenize texts
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,  # BGE-M3 supports up to 8192 tokens
        return_tensors="np"
    )
    
    input_ids = encoded['input_ids'].astype(np.int64)
    attention_mask = encoded['attention_mask'].astype(np.int64)
    
    return input_ids, attention_mask


@app.on_event("startup")
async def startup_event():
    """Startup and connect to Triton Server"""
    try:
        # Load tokenizer
        load_tokenizer()
        logger.info("Tokenizer loaded successfully")
        
        # Connect to Triton
        triton_client.connect()
        logger.info("Successfully connected to Triton Server")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close connection on shutdown"""
    triton_client.close()
    logger.info("Triton client connection closed")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "BGE-M3 Embedding API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    triton_ready = triton_client.is_ready()
    
    if triton_ready:
        return {
            "status": "healthy",
            "triton_server": "connected",
            "model": "bge-m3"
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
        # Normalize input to list
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        if not texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input cannot be empty"
            )
        
        # Prepare inputs for Triton
        input_ids, attention_mask = prepare_inputs_for_triton(texts)
        
        # Call Triton to get embeddings
        embeddings = triton_client.get_embeddings(input_ids, attention_mask)
        
        # Prepare response in OpenAI format
        embedding_data = []
        for idx, emb in enumerate(embeddings):
            embedding_data.append(
                EmbeddingData(
                    embedding=emb.tolist(),
                    index=idx
                )
            )
        
        # Estimate token count (can be improved)
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
                "id": "bge-m3",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization",
                "permission": [],
                "root": "bge-m3",
                "parent": None
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

