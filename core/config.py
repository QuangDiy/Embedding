"""Configuration settings for the application."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    triton_url: str = "triton:8000"
    
    embedding_model_name: str = "jina-embeddings-v3"
    reranker_model_name: str = "jina-reranker-v2"
    
    tokenizer_path: str = "jinaai/jina-embeddings-v3"
    reranker_tokenizer_path: str = "jinaai/jina-reranker-v2-base-multilingual"
    
    max_sequence_length: int = 8192
    reranker_max_sequence_length: int = 1024 
    
    api_title: str = "Jina AI API"
    api_description: str = "OpenAI-compatible embedding and reranking API powered by Triton Inference Server"
    api_version: str = "1.0.0"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
