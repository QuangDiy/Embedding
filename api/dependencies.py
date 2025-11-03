import logging
from functools import lru_cache
from typing import Optional

from fastapi import Header, HTTPException, status

from core.config import get_settings
from services.tokenizer_service import TokenizerService
from services.embedding_service import EmbeddingService
from services.reranking_service import RerankingService
from repositories.triton_embedding_repository import TritonEmbeddingRepository
from repositories.triton_reranking_repository import TritonRerankingRepository

logger = logging.getLogger(__name__)


@lru_cache()
def get_tokenizer_service() -> TokenizerService:
    """
    Get singleton instance of TokenizerService.
    
    Returns:
        TokenizerService instance
    """
    settings = get_settings()
    service = TokenizerService(
        tokenizer_path=settings.tokenizer_path,
        reranker_tokenizer_path=settings.reranker_tokenizer_path,
        max_length=settings.max_sequence_length,
        reranker_max_length=settings.reranker_max_sequence_length
    )
    logger.info("TokenizerService dependency created")
    return service


@lru_cache()
def get_embedding_repository() -> TritonEmbeddingRepository:
    """
    Get singleton instance of embedding repository.
    
    Returns:
        TritonEmbeddingRepository instance
    """
    settings = get_settings()
    repository = TritonEmbeddingRepository(
        triton_url=settings.triton_url,
        model_name=settings.embedding_model_name
    )
    logger.info("TritonEmbeddingRepository dependency created")
    return repository


@lru_cache()
def get_reranking_repository() -> TritonRerankingRepository:
    """
    Get singleton instance of reranking repository.
    
    Returns:
        TritonRerankingRepository instance
    """
    settings = get_settings()
    repository = TritonRerankingRepository(
        triton_url=settings.triton_url,
        model_name=settings.reranker_model_name
    )
    logger.info("TritonRerankingRepository dependency created")
    return repository


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """
    Get singleton instance of EmbeddingService.
    
    Returns:
        EmbeddingService instance with all dependencies
    """
    service = EmbeddingService(
        repository=get_embedding_repository(),
        tokenizer_service=get_tokenizer_service()
    )
    logger.info("EmbeddingService dependency created")
    return service


@lru_cache()
def get_reranking_service() -> RerankingService:
    """
    Get singleton instance of RerankingService.
    
    Returns:
        RerankingService instance with all dependencies
    """
    service = RerankingService(
        repository=get_reranking_repository(),
        tokenizer_service=get_tokenizer_service()
    )
    logger.info("RerankingService dependency created")
    return service


async def verify_api_key(
    authorization: Optional[str] = Header(None)
) -> None:
    """
    Verify API key from Authorization header (Bearer token).
    
    Expected format: Authorization: Bearer YOUR_API_KEY
    
    Args:
        authorization: Authorization header value in "Bearer TOKEN" format
        
    Raises:
        HTTPException: If API key is required but missing or invalid
    """
    settings = get_settings()
    
    if not settings.require_api_key or not settings.api_key:
        return
    
    provided_key = None
    
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            provided_key = parts[1]
        else:
            logger.warning("Invalid Authorization header format")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format. Expected: 'Authorization: Bearer YOUR_API_KEY'",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    if not provided_key:
        logger.warning("API key missing in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide via 'Authorization: Bearer YOUR_API_KEY' header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if provided_key != settings.api_key:
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
