from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class IEmbeddingRepository(ABC):
    """Interface for embedding inference operations."""
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to inference server."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if inference server and model are ready."""
        pass
    
    @abstractmethod
    def generate_embeddings(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        task_id: int
    ) -> np.ndarray:
        """
        Generate embeddings for tokenized input.
        
        Args:
            input_ids: Token IDs array, shape (batch_size, seq_length)
            attention_mask: Attention mask array, shape (batch_size, seq_length)
            task_id: Task identifier for LoRA adapter
            
        Returns:
            Embeddings array, shape (batch_size, embedding_dim)
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close connection to inference server."""
        pass


class IRerankingRepository(ABC):
    """Interface for reranking inference operations."""
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to inference server."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if inference server and model are ready."""
        pass
    
    @abstractmethod
    def compute_scores(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        Compute relevance scores for query-document pairs.
        
        Args:
            input_ids: Token IDs array, shape (batch_size, seq_length)
            attention_mask: Attention mask array, shape (batch_size, seq_length)
            
        Returns:
            Relevance scores array, shape (batch_size,)
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close connection to inference server."""
        pass
