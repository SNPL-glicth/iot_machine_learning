"""Text pattern analyzer — detects urgency shifts and spikes across sentences.

Pure function, no I/O except config JSON read at import time.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception,
)


def _load_config() -> Dict:
    cfg_path = Path(__file__).parent / "data" / "pattern_config.json"
    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_CONFIG = _load_config()


def _urgency_score(sentence: str, keywords: Dict[str, List[str]]) -> float:
    lowered = sentence.lower()
    hits = sum(1 for kw in keywords["es"] if kw in lowered)
    hits += sum(1 for kw in keywords["en"] if kw in lowered)
    words = len(lowered.split())
    return min(1.0, hits / max(1, words)) if words else 0.0


def _detect_changes(
    scores: List[float], threshold: float
) -> Tuple[List[int], List[int]]:
    change_points: List[int] = []
    spikes: List[int] = []
    n = len(scores)
    for i in range(1, n):
        delta = abs(scores[i] - scores[i - 1])
        if delta > threshold:
            change_points.append(i)
        # local spike: delta > mean + k*std of local window
        window = scores[max(0, i - 3) : i]
        if len(window) >= 2:
            mean = statistics.mean(window)
            std = statistics.stdev(window) if len(window) > 2 else 0.0
            if delta > mean + _CONFIG["spike_sigma_multiplier"] * std:
                spikes.append(i)
    return change_points, list(set(spikes))


def compute_text_patterns(sentences: List[str]) -> EnginePerception:
    """Analyze sentence sequence for urgency changes and spikes.

    Args:
        sentences: List of sentences from the document.

    Returns:
        EnginePerception with pattern density, confidence and trend.
    """
    if len(sentences) < _CONFIG["min_sentences"]:
        return EnginePerception(
            engine_name="text_pattern",
            predicted_value=0.0,
            confidence=0.3,
            trend="stable",
            stability=0.0,
            local_fit_error=0.5,
            metadata={"available": False, "reason": "insufficient_sentences"},
        )

    scores = [_urgency_score(s, _CONFIG["urgency_keywords"]) for s in sentences]
    change_points, spikes = _detect_changes(scores, _CONFIG["change_threshold"])

    n_patterns = len(change_points) + len(spikes)
    density = min(1.0, n_patterns / _CONFIG["max_patterns_cap"])

    # Trend from first half vs second half
    mid = len(scores) // 2
    first_avg = sum(scores[:mid]) / max(1, mid)
    second_avg = sum(scores[mid:]) / max(1, len(scores) - mid)
    if second_avg > first_avg + 0.1:
        trend = "escalating"
    elif second_avg < first_avg - 0.1:
        trend = "declining"
    else:
        trend = "stable"

    # Confidence: more sentences = more reliable, capped by pattern clarity
    base_conf = min(1.0, 0.5 + len(sentences) / 40.0)
    clarity = 1.0 - density
    confidence = round(min(1.0, base_conf * (0.7 + clarity * 0.3)), 4)

    # Pattern summary
    if n_patterns == 0:
        summary = "stable_narrative"
    elif len(change_points) > len(spikes):
        summary = "tone_shift_dominant"
    else:
        summary = "urgency_spike_dominant"

    return EnginePerception(
        engine_name="text_pattern",
        predicted_value=round(density, 4),
        confidence=confidence,
        trend=trend,
        stability=round(1.0 - density, 4),
        local_fit_error=round(density * 0.3, 4),
        metadata={
            "n_patterns": n_patterns,
            "change_points": change_points,
            "spikes": spikes,
            "pattern_summary": summary,
            "available": True,
        },
    )
