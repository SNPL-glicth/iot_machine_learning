"""Structural analysis on text signals (sentence lengths as time series).

Uses the existing ``compute_structural_analysis`` engine from the
domain layer to detect regime, trend, stability, and noise in the
narrative structure of text documents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MIN_POINTS_FOR_STRUCTURAL = 3

_ml_available = True
try:
    from .....domain.validators.structural_analysis import compute_structural_analysis
except Exception:
    _ml_available = False


@dataclass(frozen=True)
class TextStructuralResult:
    """Structural profile of the text's narrative signal.

    Attributes:
        regime: Detected regime (``"stable"``, ``"trending"``, etc.).
        trend: Trend strength of sentence-length signal.
        stability: Stability metric.
        noise: Noise ratio.
        available: Whether structural analysis could be computed.
        raw: Full dict from ``StructuralAnalysis.to_dict()``.
    """

    regime: str = "unknown"
    trend: float = 0.0
    stability: float = 0.0
    noise: float = 0.0
    available: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


def compute_text_structure(sentences: List[str]) -> TextStructuralResult:
    """Compute structural analysis on sentence-length signal.

    Treats sentence lengths (word count per sentence) as a numeric
    time series and runs the domain's ``compute_structural_analysis``.

    Args:
        sentences: List of sentence strings.

    Returns:
        ``TextStructuralResult`` with regime, trend, stability, noise.
    """
    if not _ml_available or len(sentences) < _MIN_POINTS_FOR_STRUCTURAL:
        return TextStructuralResult()

    try:
        sentence_lengths = [float(len(s.split())) for s in sentences]
        timestamps = [float(i) for i in range(len(sentence_lengths))]
        sa = compute_structural_analysis(sentence_lengths, timestamps)

        return TextStructuralResult(
            regime=sa.regime.value,
            trend=round(sa.trend_strength, 4),
            stability=round(sa.stability, 4),
            noise=round(sa.noise_ratio, 4),
            available=True,
            raw=sa.to_dict(),
        )
    except Exception as exc:
        logger.warning("[TEXT_STRUCTURAL] Analysis failed: %s", exc)
        return TextStructuralResult()
