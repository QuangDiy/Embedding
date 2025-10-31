"""Embedding API endpoints."""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, status, Depends

from models.schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingData, EmbeddingUsage
from services.embedding_service import EmbeddingService
from api.dependencies import get_embedding_service
from core.exceptions import ApplicationError, ValidationError, InferenceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Create embeddings for text input (OpenAI-compatible endpoint).
    
    Args:
        request: EmbeddingRequest containing text input
        service: Injected EmbeddingService
        
    Returns:
        EmbeddingResponse with embeddings
    """
    try:
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        embedding_models = service.create_embeddings(
            texts=texts,
            task=request.task or "retrieval.query"
        )
        
        embedding_data = []
        for model in embedding_models:
            embedding_data.append(
                EmbeddingData(
                    embedding=model.to_list(),
                    index=model.index
                )
            )
        
        response = EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=0,
                total_tokens=0
            )
        )
        
        logger.info(f"Successfully processed embedding request for {len(texts)} texts")
        return response
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except InferenceError as e:
        logger.error(f"Inference error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {e.message}"
        )
    except ApplicationError as e:
        logger.error(f"Application error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
