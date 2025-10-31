import logging
from functools import lru_cache

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
