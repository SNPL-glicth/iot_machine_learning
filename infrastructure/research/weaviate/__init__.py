"""Weaviate cognitive memory adapter package.

Modular implementation of CognitiveMemoryPort using Weaviate vector database.
All modules use stdlib urllib.request — no SDK dependency.

Modules:
    - http_client: HTTP primitives for REST API
    - object_operations: Object creation
    - batch_operations: Batch object creation (reduces HTTP overhead)
    - query_operations: GraphQL semantic search
    - filter_builders: Query filter utilities
    - memory_writers: Write operations (remember_*)
    - memory_readers: Read operations (recall_*)
    - result_mapper: Result conversion utilities
"""

from __future__ import annotations

__all__ = [
    "WeaviateCognitiveAdapter",
    "WeaviateBatch",
    "BatchResult",
]

from .weaviate_cognitive import WeaviateCognitiveAdapter
from .batch_operations import WeaviateBatch, BatchResult
