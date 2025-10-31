import logging
import numpy as np
from typing import List, Tuple
from transformers import AutoTokenizer

from core.exceptions import TokenizerError

logger = logging.getLogger(__name__)


class TokenizerService:
    """Service for tokenizing text inputs."""
    
    def __init__(
        self,
        tokenizer_path: str,
        reranker_tokenizer_path: str,
        max_length: int = 8192,
        reranker_max_length: int = 512
    ):
        """
        Initialize tokenizer service.
        
        Args:
            tokenizer_path: Path to embedding tokenizer
            reranker_tokenizer_path: Path to reranker tokenizer
            max_length: Max sequence length for embeddings
            reranker_max_length: Max sequence length for reranking
        """
        self._tokenizer_path = tokenizer_path
        self._reranker_tokenizer_path = reranker_tokenizer_path
        self._max_length = max_length
        self._reranker_max_length = reranker_max_length
        
        self._embedding_tokenizer = None
        self._reranker_tokenizer = None
        
        logger.info("TokenizerService initialized")
    
    def load_embedding_tokenizer(self) -> None:
        """Load embedding tokenizer from HuggingFace."""
        if self._embedding_tokenizer is not None:
            return
        
        try:
            logger.info(f"Loading embedding tokenizer from {self._tokenizer_path}...")
            self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path,
                trust_remote_code=False
            )
            logger.info(f"✓ Embedding tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding tokenizer: {e}")
            raise TokenizerError(
                "Failed to load embedding tokenizer",
                details={"path": self._tokenizer_path, "error": str(e)}
            )
    
    def load_reranker_tokenizer(self) -> None:
        """Load reranker tokenizer from HuggingFace."""
        if self._reranker_tokenizer is not None:
            return
        
        try:
            logger.info(f"Loading reranker tokenizer from {self._reranker_tokenizer_path}...")
            self._reranker_tokenizer = AutoTokenizer.from_pretrained(
                self._reranker_tokenizer_path,
                trust_remote_code=False
            )
            logger.info(f"✓ Reranker tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker tokenizer: {e}")
            raise TokenizerError(
                "Failed to load reranker tokenizer",
                details={"path": self._reranker_tokenizer_path, "error": str(e)}
            )
    
    def tokenize_for_embedding(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize texts for embedding generation.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tuple of (input_ids, attention_mask) as numpy arrays
            
        Raises:
            TokenizerError: If tokenization fails
        """
        if self._embedding_tokenizer is None:
            self.load_embedding_tokenizer()
        
        try:
            encoded = self._embedding_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="np"
            )
            
            input_ids = encoded['input_ids'].astype(np.int64)
            attention_mask = encoded['attention_mask'].astype(np.int64)
            
            logger.debug(f"Tokenized {len(texts)} texts, shape: {input_ids.shape}")
            return input_ids, attention_mask
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise TokenizerError(
                "Failed to tokenize texts for embedding",
                details={"num_texts": len(texts), "error": str(e)}
            )
    
    def tokenize_for_reranking(
        self,
        query: str,
        documents: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize query-document pairs for reranking.
        
        Args:
            query: Search query
            documents: List of documents
            
        Returns:
            Tuple of (input_ids, attention_mask) as numpy arrays
            
        Raises:
            TokenizerError: If tokenization fails
        """
        if self._reranker_tokenizer is None:
            self.load_reranker_tokenizer()
        
        try:
            pairs = [[query, doc] for doc in documents]
            
            encoded = self._reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self._reranker_max_length,
                return_tensors="np"
            )
            
            input_ids = encoded['input_ids'].astype(np.int64)
            attention_mask = encoded['attention_mask'].astype(np.int64)
            
            logger.debug(f"Tokenized {len(documents)} pairs, shape: {input_ids.shape}")
            return input_ids, attention_mask
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise TokenizerError(
                "Failed to tokenize query-document pairs",
                details={"num_documents": len(documents), "error": str(e)}
            )

