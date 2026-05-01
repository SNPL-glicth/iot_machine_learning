"""EmbeddingNarrativeGenerator — no f-strings, no if/elif, no external LLM.

Input: 18-dim situation vector.
Pipeline:
    1. Run through NarrativeEmbeddingNetwork → 8-dim embedding.
    2. Cosine similarity against phrase bank target vectors.
    3. Select top-k phrases with similarity > threshold.
    4. Concatenate with " | " separator.
    5. If no phrase passes threshold → generic safety fallback.
"""

from __future__ import annotations

import math
from typing import List, Optional

from .embedding_network import NarrativeEmbeddingNetwork
from .phrase_bank import get_phrase_bank, PhraseEntry


# Threshold for a phrase to be included in output.
_DEFAULT_SIMILARITY_THRESHOLD = 0.70

# Max phrases to concatenate.
_MAX_PHRASES = 3

# Generic safety fallback when no phrase matches.
_GENERIC_FALLBACK = (
    "Evaluación completada. Resultados disponibles para revisión humana."
)


class EmbeddingNarrativeGenerator:
    """Generates narrative explanations using deterministic semantic mapping
    or a trained embedding network (when available)."""

    def __init__(
        self,
        network: Optional[NarrativeEmbeddingNetwork] = None,
        neural_enabled: bool = False,
        phrase_bank: Optional[List[PhraseEntry]] = None,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
        max_phrases: int = _MAX_PHRASES,
        fallback: str = _GENERIC_FALLBACK,
    ) -> None:
        self.neural_enabled = neural_enabled
        self.network = network if (network or neural_enabled) else None
        self.phrase_bank = phrase_bank or get_phrase_bank()
        self.threshold = similarity_threshold
        self.max_phrases = max_phrases
        self.fallback = fallback

    def generate(
        self,
        situation_vector: List[float],
        fallback_text: str = "",
        domain: Optional[str] = None,
    ) -> str:
        """Select the best-matching phrase from the phrase bank.

        When ``neural_enabled=True``: uses the neural-network projection of
        ``situation_vector`` → 8-dim embedding.

        When ``neural_enabled=False`` (default): maps ``situation_vector``
        deterministically to 8-dim semantic space without training.

        Args:
            situation_vector: 18-dim raw situation vector
            fallback_text: Text to return if no phrase matches above threshold
            domain: Document domain hint (e.g. "security", "infrastructure").
                When provided, only domain-matching phrases and domain-neutral
                phrases (``PhraseEntry.domain is None``) are scored. This
                prevents cross-domain phrase contamination.
        Returns:
            A narrative string (single best-matching phrase or fallback)
        """
        if self.neural_enabled and self.network is not None:
            embedding = self.network.forward(situation_vector)
        else:
            embedding = self._build_semantic_embedding(situation_vector)

        # Pre-filter by domain: keep neutral phrases + domain-matching ones
        eligible = [
            phrase for phrase in self.phrase_bank
            if phrase.domain is None or phrase.domain == domain
        ] if domain else self.phrase_bank

        scored: List[tuple[float, PhraseEntry]] = []
        for phrase in eligible:
            sim = _cosine_similarity(embedding, phrase.target)
            scored.append((sim, phrase))

        # Sort descending
        scored.sort(key=lambda t: t[0], reverse=True)

        # Filter by threshold
        selected = [
            phrase.text for sim, phrase in scored
            if sim >= self.threshold
        ][: self.max_phrases]

        if not selected:
            return self.fallback

        return " | ".join(selected)

    def _build_semantic_embedding(self, vector: List[float]) -> List[float]:
        """Map 18-dim situation vector to 8-dim semantic space deterministically.

        Uses known semantic dimensions from the situation vector:
            dim 0: criticality / urgency      ← vector[6]  composite_severity
            dim 1: warning / concern level    ← vector[1]  regime_curvature
            dim 2: stability / calm           ← vector[2]  regime_stability
            dim 3: positive trend strength  ← 0.0 (no direct mapping)
            dim 4: negative trend strength  ← 0.0 (no direct mapping)
            dim 5: anomaly presence           ← vector[9]  drift_score
            dim 6: high model confidence    ← vector[7]  urgency_score
            dim 7: uncertainty / low conf   ← 0.0 (padding)

        Padding with zeros for dimensions without explicit mapping keeps the
        comparison coherent with the phrase-bank target vectors.
        """
        if len(vector) < 18:
            # Pad short vectors so indices don't raise IndexError
            vector = list(vector) + [0.0] * (18 - len(vector))

        return [
            float(vector[6]),   # dim 0: criticality / urgency
            float(vector[1]),   # dim 1: warning / concern (volatility)
            float(vector[2]),   # dim 2: stability / calm
            0.0,                # dim 3: positive trend (no info)
            0.0,                # dim 4: negative trend (no info)
            float(vector[9]),   # dim 5: anomaly presence
            float(vector[7]),   # dim 6: high model confidence
            0.0,                # dim 7: uncertainty / low confidence (padding)
        ]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return min(1.0, max(-1.0, dot / (norm_a * norm_b)))
