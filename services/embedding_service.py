import logging
from typing import List

from repositories.interfaces import IEmbeddingRepository
from services.tokenizer_service import TokenizerService
from models.domain import EmbeddingModel
from core.constants import TASK_MAPPING, DEFAULT_TASK
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings."""
    
    def __init__(
        self,
        repository: IEmbeddingRepository,
        tokenizer_service: TokenizerService
    ):
        """
        Initialize embedding service.
        
        Args:
            repository: Repository for embedding inference
            tokenizer_service: Service for text tokenization
        """
        self._repository = repository
        self._tokenizer_service = tokenizer_service
        logger.info("EmbeddingService initialized")
    
    def create_embeddings(
        self,
        texts: List[str],
        task: str = DEFAULT_TASK
    ) -> List[EmbeddingModel]:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: List of texts to embed
            task: Task type (retrieval.query, text-matching, etc.)
            
        Returns:
            List of EmbeddingModel objects
            
        Raises:
            ValidationError: If inputs are invalid
            InferenceError: If embedding generation fails
        """
        if not texts:
            raise ValidationError("Text input cannot be empty")
        
        if not all(isinstance(text, str) for text in texts):
            raise ValidationError("All inputs must be strings")
        
        task_id = TASK_MAPPING.get(task, TASK_MAPPING[DEFAULT_TASK])
        logger.info(f"Generating embeddings for {len(texts)} texts with task '{task}'")
        
        input_ids, attention_mask = self._tokenizer_service.tokenize_for_embedding(texts)
        
        embeddings = self._repository.generate_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_id=task_id
        )
        
        embedding_models = []
        for idx, embedding_vector in enumerate(embeddings):
            embedding_models.append(
                EmbeddingModel(
                    vector=embedding_vector,
                    index=idx
                )
            )
        
        logger.info(f"Successfully generated {len(embedding_models)} embeddings")
        return embedding_models
    
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self._repository.is_ready()
