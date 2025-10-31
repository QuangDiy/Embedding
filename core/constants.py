from enum import IntEnum


class TaskType(IntEnum):
    """Task types for Jina embeddings with LoRA adapters."""
    RETRIEVAL_QUERY = 0
    RETRIEVAL_PASSAGE = 1
    SEPARATION = 2
    CLASSIFICATION = 3
    TEXT_MATCHING = 4


TASK_MAPPING = {
    "retrieval.query": TaskType.RETRIEVAL_QUERY,
    "retrieval.passage": TaskType.RETRIEVAL_PASSAGE,
    "separation": TaskType.SEPARATION,
    "classification": TaskType.CLASSIFICATION,
    "text-matching": TaskType.TEXT_MATCHING,
}


DEFAULT_TASK = "retrieval.query"
