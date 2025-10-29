import numpy as np
import tritonclient.http as httpclient
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TritonEmbeddingClient:
    """Client to communicate with Triton Inference Server"""
    
    def __init__(self, triton_url: str = "triton:8000", model_name: str = "bge-m3"):
        self.triton_url = triton_url
        self.model_name = model_name
        self.client = None
        
    def connect(self):
        """Initialize connection with Triton Server"""
        try:
            self.client = httpclient.InferenceServerClient(url=self.triton_url)
            if not self.client.is_server_live():
                raise RuntimeError("Triton server is not live")
            if not self.client.is_model_ready(self.model_name):
                raise RuntimeError(f"Model {self.model_name} is not ready")
            logger.info(f"Connected to Triton server at {self.triton_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Triton: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if Triton server is ready"""
        try:
            if self.client is None:
                self.connect()
            return self.client.is_server_live() and self.client.is_model_ready(self.model_name)
        except:
            return False
    
    def get_embeddings(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Send request to Triton and receive embeddings
        
        Args:
            input_ids: Array of token IDs, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)
            
        Returns:
            embeddings: Array of embeddings, shape (batch_size, embedding_dim)
        """
        if self.client is None:
            self.connect()
        
        # Prepare input tensors
        inputs = []
        inputs.append(
            httpclient.InferInput("input_ids", input_ids.shape, "INT64")
        )
        inputs[0].set_data_from_numpy(input_ids)
        
        inputs.append(
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
        )
        inputs[1].set_data_from_numpy(attention_mask)
        
        # Prepare output
        outputs = []
        outputs.append(
            httpclient.InferRequestedOutput("sentence_embedding")
        )
        
        # Send inference request
        try:
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # Get embeddings result
            embeddings = response.as_numpy("sentence_embedding")
            return embeddings
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def close(self):
        """Close connection"""
        if self.client:
            self.client.close()

