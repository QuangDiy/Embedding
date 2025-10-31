"""Health check and info endpoints."""

import logging
import time
from fastapi import APIRouter, HTTPException, status, Depends

from services.embedding_service import EmbeddingService
from services.reranking_service import RerankingService
from api.dependencies import get_embedding_service, get_reranking_service
from core.config import get_settings
from models.domain import ModelInfo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/")
async def root():
    """Root endpoint with service information."""
    settings = get_settings()
    return {
        "status": "running",
        "service": settings.api_title,
        "version": settings.api_version
    }


@router.get("/health")
async def health_check(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    reranking_service: RerankingService = Depends(get_reranking_service)
):
    """
    Detailed health check endpoint.
    
    Checks the status of Triton server and models.
    """
    settings = get_settings()
    embedding_ready = embedding_service.is_ready()
    reranker_ready = reranking_service.is_ready()
    
    if embedding_ready and reranker_ready:
        return {
            "status": "healthy",
            "triton_server": "connected",
            "models": {
                settings.embedding_model_name: "ready",
                settings.reranker_model_name: "ready"
            }
        }
    elif embedding_ready or reranker_ready:
        return {
            "status": "partial",
            "triton_server": "connected",
            "models": {
                settings.embedding_model_name: "ready" if embedding_ready else "not_ready",
                settings.reranker_model_name: "ready" if reranker_ready else "not_ready"
            }
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton server is not ready"
        )


@router.get("/v1/models")
async def list_models():
    """
    List available models (OpenAI-compatible endpoint).
    
    Returns information about available embedding and reranking models.
    """
    settings = get_settings()
    
    embedding_model = ModelInfo(
        id=settings.embedding_model_name,
        owned_by="jinaai"
    )
    
    reranker_model = ModelInfo(
        id=settings.reranker_model_name,
        owned_by="jinaai"
    )
    
    created_time = int(time.time())
    
    return {
        "object": "list",
        "data": [
            embedding_model.to_dict(created_time),
            reranker_model.to_dict(created_time)
        ]
    }

