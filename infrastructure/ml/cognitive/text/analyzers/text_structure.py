"""Text structural regime classification.

Classifies the structural pattern of a text based on sentence-level metrics:
NARRATIVE, TECHNICAL, MIXED, or UNKNOWN.

Pure function: no I/O, no global state, no external dependencies.
Guarantees: never raises, never returns None.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List

_NUMERIC_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?(?:%|°[CF])?\b")
_TECHNICAL_KEYWORD_PATTERN = re.compile(
    r"\b(valor|parámetro|parametro|medición|medicion|lectura|umbral|límite|limite|"
    r"rango|promedio|desviación|desviacion|tolerancia|especificación|especificacion|"
    r"configuración|configuracion|parámetro|parametro|sensor|dispositivo|equipo|"
    r"value|parameter|measurement|reading|threshold|limit|range|average|deviation|"
    r"tolerance|specification|configuration|sensor|device|equipment)\b",
    re.IGNORECASE,
)

_STABLE_VARIANCE_THRESHOLD: float = 0.15
_TRENDING_VARIANCE_THRESHOLD: float = 0.35
_STABLE_NOISE_THRESHOLD: float = 0.10
_TRENDING_NOISE_THRESHOLD: float = 0.25
_LONG_SENTENCE_THRESHOLD: float = 30.0
_SHORT_SENTENCE_THRESHOLD: float = 8.0
_TECHNICAL_SENTENCE_RATIO: float = 0.3
_NARRATIVE_SENTENCE_RATIO: float = 0.5


@dataclass(frozen=True)
class StructureResult:
    """Immutable result of structural regime analysis.

    Attributes:
        regime: Structural regime label:
            "narrative", "technical", "mixed", or "unknown".
        trend: Length trend across sentences:
            "increasing", "decreasing", "stable", or "flat".
        stability: Normalised measure of sentence-length consistency
            [0.0, 1.0]. Higher = more uniform structure.
        noise: Ratio of numeric tokens to total tokens
            [0.0, 1.0]. Higher = more data-dense text.
        available: Whether enough data was available for classification.
    """

    regime: str
    trend: str
    stability: float
    noise: float
    available: bool


def _sentence_lengths(sentences: List[str]) -> List[int]:
    """Compute word count per sentence."""
    return [len(s.split()) for s in sentences]


def _classify_trend(lengths: List[int]) -> str:
    """Classify length trend across sentences.

    Uses simple slope estimation: compare first-half mean vs second-half mean.
    """
    n = len(lengths)
    if n < 3:
        return "flat"
    mid = n // 2
    first_half = lengths[:mid]
    second_half = lengths[mid:]
    diff = (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half))
    threshold = (sum(lengths) / n) * 0.15
    if diff > threshold:
        return "increasing"
    if diff < -threshold:
        return "decreasing"
    return "stable"


def _estimate_stability(lengths: List[int]) -> float:
    """Estimate structural stability from coefficient of variation.

    Returns 1.0 for perfect stability, approaching 0.0 for high variance.
    """
    if not lengths:
        return 0.0
    mean = sum(lengths) / len(lengths)
    if mean < 1.0:
        return 1.0
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    cv = math.sqrt(variance) / mean
    return max(0.0, min(1.0, 1.0 - cv))


def _compute_noise_ratio(text: str, sentences: List[str]) -> float:
    """Compute ratio of numeric tokens to total word tokens."""
    total_words = sum(len(s.split()) for s in sentences)
    if total_words == 0:
        return 0.0
    numeric_count = len(_NUMERIC_PATTERN.findall(text))
    return min(1.0, numeric_count / total_words)


def _classify_regime(
    lengths: List[int],
    stability: float,
    noise: float,
    technical_ratio: float,
) -> str:
    """Classify structural regime from derived metrics."""
    if len(lengths) < 3:
        return "unknown"

    if technical_ratio >= _TECHNICAL_SENTENCE_RATIO:
        if stability >= _STABLE_VARIANCE_THRESHOLD:
            return "technical"
        return "mixed"

    if noise >= _TRENDING_NOISE_THRESHOLD:
        return "mixed"

    if stability >= _STABLE_VARIANCE_THRESHOLD:
        if noise <= _STABLE_NOISE_THRESHOLD:
            return "narrative"
        return "mixed"

    return "unknown"


def _compute_technical_ratio(sentences: List[str]) -> float:
    """Compute ratio of sentences containing technical keywords."""
    if not sentences:
        return 0.0
    technical_count = sum(
        1 for s in sentences if _TECHNICAL_KEYWORD_PATTERN.search(s)
    )
    return technical_count / len(sentences)


def compute_text_structure(sentences: List[str]) -> StructureResult:
    """Analyse the structural regime of a text from its sentences.

    Args:
        sentences: List of sentence strings. May be empty.

    Returns:
        StructureResult — never None, never raises.
        Empty input returns available=False with neutral values.
    """
    if not sentences:
        return StructureResult(
            regime="unknown",
            trend="flat",
            stability=0.0,
            noise=0.0,
            available=False,
        )

    lengths = _sentence_lengths(sentences)
    full_text = " ".join(sentences)
    noise = _compute_noise_ratio(full_text, sentences)
    stability = _estimate_stability(lengths)
    trend = _classify_trend(lengths)
    technical_ratio = _compute_technical_ratio(sentences)
    regime = _classify_regime(lengths, stability, noise, technical_ratio)

    return StructureResult(
        regime=regime,
        trend=trend,
        stability=round(stability, 4),
        noise=round(noise, 4),
        available=True,
    )
