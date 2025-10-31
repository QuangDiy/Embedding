from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class EmbeddingModel:
    """Internal representation of an embedding."""
    
    vector: np.ndarray
    index: int
    
    def to_list(self) -> List[float]:
        """Convert vector to list of floats."""
        return self.vector.tolist()


@dataclass
class RerankResultModel:
    """Internal representation of a reranking result."""
    
    index: int
    relevance_score: float
    document: Optional[any] = None
    
    def __lt__(self, other):
        """Support sorting by relevance score (descending)."""
        return self.relevance_score > other.relevance_score


@dataclass
class ModelInfo:
    """Model metadata information."""
    
    id: str
    object: str = "model"
    owned_by: str = "jinaai"
    
    def to_dict(self, created: int) -> dict:
        """Convert to API response format."""
        return {
            "id": self.id,
            "object": self.object,
            "created": created,
            "owned_by": self.owned_by,
            "permission": [],
            "root": self.id,
            "parent": None,
        }
