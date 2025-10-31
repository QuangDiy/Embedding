"""Reranking API endpoints."""

import logging
from fastapi import APIRouter, HTTPException, status, Depends

from models.schemas import RerankRequest, RerankResponse, RerankResult
from services.reranking_service import RerankingService
from api.dependencies import get_reranking_service
from core.exceptions import ApplicationError, ValidationError, InferenceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["reranking"])


@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(
    request: RerankRequest,
    service: RerankingService = Depends(get_reranking_service)
):
    """
    Rerank documents based on relevance to a query (OpenAI-compatible endpoint).
    
    Args:
        request: RerankRequest containing query and documents
        service: Injected RerankingService
        
    Returns:
        RerankResponse with reranked documents and relevance scores
    """
    try:
        # Rerank documents
        result_models = service.rerank_documents(
            query=request.query,
            documents=request.documents,
            top_n=request.top_n,
            return_documents=request.return_documents
        )
        
        # Convert to API response format
        results = []
        for model in result_models:
            result = RerankResult(
                index=model.index,
                relevance_score=model.relevance_score,
                document=model.document
            )
            results.append(result)
        
        # Extract document texts for token counting
        doc_texts = service._extract_document_texts(request.documents)
        total_tokens = service.calculate_token_count(doc_texts)
        
        response = RerankResponse(
            data=results,
            model=request.model,
            usage={"total_tokens": total_tokens}
        )
        
        logger.info(f"Successfully reranked {len(request.documents)} documents")
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
            detail=f"Failed to rerank documents: {e.message}"
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

