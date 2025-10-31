"""API v1 routers."""

from api.v1.embeddings import router as embeddings_router
from api.v1.reranking import router as reranking_router
from api.v1.health import router as health_router

__all__ = [
    "embeddings_router",
    "reranking_router",
    "health_router",
]

