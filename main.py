import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from core.config import get_settings
from api.v1 import embeddings_router, reranking_router, health_router
from api.dependencies import (
    get_embedding_service,
    get_reranking_service,
    get_tokenizer_service,
    get_embedding_repository,
    get_reranking_repository
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version
    )
    
    app.include_router(health_router)
    app.include_router(embeddings_router)
    app.include_router(reranking_router)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        try:
            logger.info("Starting application...")
            
            tokenizer_service = get_tokenizer_service()
            tokenizer_service.load_embedding_tokenizer()
            logger.info("Embedding tokenizer loaded")
            
            tokenizer_service.load_reranker_tokenizer()
            logger.info("Reranker tokenizer loaded")
            
            embedding_repo = get_embedding_repository()
            embedding_repo.connect()
            logger.info("Connected to Triton server for embeddings")
            
            reranking_repo = get_reranking_repository()
            reranking_repo.connect()
            logger.info("Connected to Triton server for reranking")
            
            logger.info("Application started successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown."""
        try:
            logger.info("Shutting down application...")
            
            embedding_repo = get_embedding_repository()
            embedding_repo.close()
            
            reranking_repo = get_reranking_repository()
            reranking_repo.close()
            
            logger.info("Application shutdown complete")
            
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
