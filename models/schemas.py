from pydantic import BaseModel, Field
from typing import List, Union, Optional


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    
    input: Union[str, List[str]] = Field(
        ...,
        description="Text or list of texts to embed"
    )
    model: str = Field(
        default="jina-embeddings-v3",
        description="Model name"
    )
    encoding_format: Optional[str] = Field(
        default="float",
        description="Encoding format (float or base64)"
    )
    task: Optional[str] = Field(
        default="retrieval.query",
        description="Task type: retrieval.query, retrieval.passage, separation, classification, text-matching"
    )
    user: Optional[str] = Field(
        default=None,
        description="User identifier"
    )


class EmbeddingData(BaseModel):
    """Embedding data object."""
    
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage statistics."""
    
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""
    
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class RerankRequest(BaseModel):
    """OpenAI-compatible rerank request."""
    
    query: str = Field(
        ...,
        description="The search query"
    )
    documents: Union[List[str], List[dict]] = Field(
        ...,
        description="List of documents to rerank"
    )
    model: str = Field(
        default="jina-reranker-v2",
        description="Model name"
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Number of most relevant documents to return"
    )
    return_documents: Optional[bool] = Field(
        default=True,
        description="Whether to return document text"
    )


class RerankResult(BaseModel):
    """Single rerank result."""
    
    index: int
    relevance_score: float
    document: Optional[Union[str, dict]] = None


class RerankResponse(BaseModel):
    """OpenAI-compatible rerank response."""
    
    object: str = "list"
    data: List[RerankResult]
    model: str
    usage: dict
