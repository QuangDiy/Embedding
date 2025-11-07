import logging
from typing import List
import numpy as np
from core.config import get_settings

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
        total_texts = len(texts)
        logger.info(f"Generating embeddings for {total_texts} texts with task '{task}'")
        
        settings = get_settings()
        max_batch = getattr(settings, "embedding_client_max_batch", 4)
        if not isinstance(max_batch, int) or max_batch <= 0:
            max_batch = 4
        
        if total_texts > max_batch:
            num_chunks = (total_texts + max_batch - 1) // max_batch
            logger.info(f"Chunking {total_texts} texts into {num_chunks} batches of up to {max_batch}")
        
        all_embeddings: List[np.ndarray] = []
        for start_idx in range(0, total_texts, max_batch):
            end_idx = min(start_idx + max_batch, total_texts)
            chunk_texts = texts[start_idx:end_idx]
            input_ids, attention_mask = self._tokenizer_service.tokenize_for_embedding(chunk_texts)
            chunk_embeddings = self._repository.generate_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task_id=task_id
            )
            all_embeddings.append(chunk_embeddings)
        
        embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 0), dtype=np.float32)
        
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
