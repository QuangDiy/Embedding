class ApplicationError(Exception):
    """Base exception for all application errors."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(ApplicationError):
    """Raised when input validation fails."""
    pass


class InferenceError(ApplicationError):
    """Raised when inference operation fails."""
    pass


class TokenizerError(ApplicationError):
    """Raised when tokenizer operations fail."""
    pass


class RepositoryError(ApplicationError):
    """Raised when repository operations fail."""
    pass


class ModelNotReadyError(RepositoryError):
    """Raised when model is not ready for inference."""
    pass


class ServerNotAvailableError(RepositoryError):
    """Raised when inference server is not available."""
    pass

