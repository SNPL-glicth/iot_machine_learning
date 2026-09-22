"""VectorMemoryPort protocol for semantic similarity search."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class VectorMemoryPort(Protocol):
    """Protocol for vector memory similarity queries."""

    def search_similar_explanations(
        self,
        concept: str,
        min_certainty: float,
        limit: int = 5,
    ) -> List[str]:
        """Search for explanation texts semantically similar to concept.

        Args:
            concept: Text concept to search for.
            min_certainty: Minimum semantic certainty (0.0 to 1.0).
            limit: Maximum number of results to return.

        Returns:
            List of explanation texts from semantically similar documents.
        """
        ...
