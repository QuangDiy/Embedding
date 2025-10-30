import numpy as np
import tritonclient.http as httpclient
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TritonEmbeddingClient:
    """Client to communicate with Triton Inference Server"""
    
    def __init__(self, triton_url: str = "triton:8000", model_name: str = "jina-embeddings-v3"):
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
    
    def get_embeddings(self, input_ids: np.ndarray, attention_mask: np.ndarray, task_id: int = 0) -> np.ndarray:
        """
        Send request to Triton and receive embeddings
        
        Args:
            input_ids: Array of token IDs, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)
            task_id: Task ID for LoRA adapter (0=retrieval.query, 1=retrieval.passage, 
                    2=separation, 3=classification, 4=text-matching)
            
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
        
        # Add task_id input - one task_id per batch item
        batch_size = input_ids.shape[0]
        task_id_array = np.full((batch_size, 1), task_id, dtype=np.int64)
        inputs.append(
            httpclient.InferInput("task_id", task_id_array.shape, "INT64")
        )
        inputs[2].set_data_from_numpy(task_id_array)
        
        # Prepare output
        outputs = []
        outputs.append(
            httpclient.InferRequestedOutput("last_hidden_state")
        )
        
        # Send inference request
        try:
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # Get last hidden state
            last_hidden_state = response.as_numpy("last_hidden_state")
            
            # Apply mean pooling
            embeddings = self._mean_pooling(last_hidden_state, attention_mask)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def _mean_pooling(self, model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Apply mean pooling on model output
        
        Args:
            model_output: Last hidden state from model, shape (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask, shape (batch_size, seq_length)
            
        Returns:
            pooled_embeddings: Mean pooled embeddings, shape (batch_size, hidden_size)
        """
        # Expand attention mask to match hidden state dimensions
        attention_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        
        # Apply mask and sum
        sum_embeddings = np.sum(model_output * attention_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(attention_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask
        
        # Normalize embeddings (L2 normalization)
        norms = np.linalg.norm(mean_embeddings, ord=2, axis=1, keepdims=True)
        normalized_embeddings = mean_embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        
        return normalized_embeddings
    
    def close(self):
        """Close connection"""
        if self.client:
            self.client.close()

