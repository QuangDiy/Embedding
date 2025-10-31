import logging
import numpy as np
from typing import Optional

from repositories.interfaces import IRerankingRepository
from core.exceptions import InferenceError, RepositoryError, ModelNotReadyError
from api.triton_client import TritonRerankerClient

logger = logging.getLogger(__name__)


class TritonRerankingRepository(IRerankingRepository):
    """Repository for reranking inference using Triton Inference Server."""
    
    def __init__(self, triton_url: str, model_name: str):
        """
        Initialize repository with Triton client.
        
        Args:
            triton_url: URL of Triton inference server
            model_name: Name of the reranking model
        """
        self._client = TritonRerankerClient(
            triton_url=triton_url,
            model_name=model_name
        )
        self._triton_url = triton_url
        self._model_name = model_name
        logger.info(f"Initialized TritonRerankingRepository for model {model_name}")
    
    def connect(self) -> None:
        """Establish connection to Triton server."""
        try:
            self._client.connect()
            logger.info(f"Connected to Triton server at {self._triton_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise RepositoryError(
                f"Failed to connect to Triton server",
                details={"url": self._triton_url, "error": str(e)}
            )
    
    def is_ready(self) -> bool:
        """Check if Triton server and model are ready."""
        try:
            return self._client.is_ready()
        except Exception as e:
            logger.warning(f"Error checking readiness: {e}")
            return False
    
    def compute_scores(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        Compute relevance scores using Triton inference.
        
        Args:
            input_ids: Token IDs array
            attention_mask: Attention mask array
            
        Returns:
            Relevance scores
            
        Raises:
            InferenceError: If inference fails
        """
        if not self.is_ready():
            raise ModelNotReadyError(
                f"Model {self._model_name} is not ready",
                details={"model": self._model_name}
            )
        
        try:
            scores = self._client.get_rerank_scores(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logger.info(f"Computed {len(scores)} reranking scores")
            return scores
        except Exception as e:
            logger.error(f"Reranking inference failed: {e}")
            raise InferenceError(
                "Failed to compute relevance scores",
                details={"error": str(e), "model": self._model_name}
            )
    
    def close(self) -> None:
        """Close connection to Triton server."""
        try:
            self._client.close()
            logger.info("Closed Triton client connection")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
