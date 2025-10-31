import logging
from typing import List, Union

from repositories.interfaces import IRerankingRepository
from services.tokenizer_service import TokenizerService
from models.domain import RerankResultModel
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class RerankingService:
    """Service for document reranking."""
    
    def __init__(
        self,
        repository: IRerankingRepository,
        tokenizer_service: TokenizerService
    ):
        """
        Initialize reranking service.
        
        Args:
            repository: Repository for reranking inference
            tokenizer_service: Service for text tokenization
        """
        self._repository = repository
        self._tokenizer_service = tokenizer_service
        logger.info("RerankingService initialized")
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Union[str, dict]],
        top_n: int = None,
        return_documents: bool = True
    ) -> List[RerankResultModel]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of documents (strings or dicts)
            top_n: Number of top results to return
            return_documents: Whether to include document content
            
        Returns:
            List of RerankResultModel objects sorted by relevance
            
        Raises:
            ValidationError: If inputs are invalid
            InferenceError: If reranking fails
        """
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        
        if not documents:
            raise ValidationError("Documents list cannot be empty")
        
        logger.info(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")
        
        doc_texts = self._extract_document_texts(documents)
        
        input_ids, attention_mask = self._tokenizer_service.tokenize_for_reranking(
            query=query,
            documents=doc_texts
        )
        
        scores = self._repository.compute_scores(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        results = []
        for idx, score in enumerate(scores):
            result = RerankResultModel(
                index=idx,
                relevance_score=float(score),
                document=documents[idx] if return_documents else None
            )
            results.append(result)
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        if top_n is not None and top_n > 0:
            results = results[:top_n]
        
        logger.info(f"Reranking complete, returning {len(results)} results")
        return results
    
    def _extract_document_texts(self, documents: List[Union[str, dict]]) -> List[str]:
        """
        Extract text content from documents.
        
        Args:
            documents: List of documents (strings or dicts)
            
        Returns:
            List of text strings
        """
        doc_texts = []
        for doc in documents:
            if isinstance(doc, str):
                doc_texts.append(doc)
            elif isinstance(doc, dict):
                text = doc.get('text', doc.get('content', str(doc)))
                doc_texts.append(text)
            else:
                doc_texts.append(str(doc))
        
        return doc_texts
    
    def calculate_token_count(self, documents: List[str]) -> int:
        """
        Calculate approximate token count for usage statistics.
        
        Args:
            documents: List of document texts
            
        Returns:
            Approximate token count
        """
        return sum(len(doc.split()) for doc in documents)
    
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self._repository.is_ready()

