"""Universal signal profiler — dispatcher to type-specific profilers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from iot_machine_learning.domain.entities.explainability.signal_snapshot import SignalSnapshot

from .types import InputType


class UniversalSignalProfiler:
    """Maps any input type to SignalSnapshot.

    Dispatches to appropriate profiler based on InputType.
    Reuses TextSignalProfiler for TEXT.
    Creates profilers for NUMERIC, TABULAR, MIXED.

    All profilers produce SignalSnapshot with consistent schema:
        n_points, mean, std, noise_ratio, slope, curvature, regime, dt, extra
    """

    def profile(
        self,
        raw_data: Any,
        input_type: InputType,
        metadata: Dict[str, Any],
        domain: str,
    ) -> SignalSnapshot:
        """Build SignalSnapshot from any input type.

        Args:
            raw_data: Original input
            input_type: Detected InputType
            metadata: Pre-computed metadata from input_detector
            domain: Classified domain

        Returns:
            SignalSnapshot with universal encoding
        """
        if input_type == InputType.TEXT:
            return self._profile_text(str(raw_data), metadata, domain)
        
        if input_type == InputType.NUMERIC:
            return self._profile_numeric(raw_data, metadata, domain)
        
        if input_type == InputType.TABULAR:
            return self._profile_tabular(raw_data, metadata, domain)
        
        if input_type == InputType.MIXED:
            return self._profile_mixed(raw_data, metadata, domain)
        
        return SignalSnapshot.empty()

    def _profile_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        domain: str,
    ) -> SignalSnapshot:
        """Delegate to TextSignalProfiler logic (inline to avoid circular import)."""
        word_count = metadata.get("word_count", 0)
        paragraph_count = metadata.get("paragraph_count", 0)
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentence_lengths = [float(len(s.split())) for s in sentences]
        
        avg_sentence_length = (
            sum(sentence_lengths) / len(sentence_lengths)
            if sentence_lengths else 0.0
        )
        std = _std(sentence_lengths)
        
        unique_words = len(set(text.lower().split()))
        total_words = word_count
        vocabulary_richness = unique_words / max(total_words, 1)
        
        extra: Dict[str, Any] = {
            "source": "universal_signal_profiler",
            "input_type": "text",
            "paragraph_count": paragraph_count,
            "vocabulary_richness": round(vocabulary_richness, 4),
        }
        
        return SignalSnapshot(
            n_points=word_count,
            mean=round(avg_sentence_length, 4),
            std=round(std, 4),
            noise_ratio=round(vocabulary_richness, 4),
            slope=0.0,
            curvature=0.0,
            regime=domain,
            dt=1.0,
            extra=extra,
        )

    def _profile_numeric(
        self,
        values: list,
        metadata: Dict[str, Any],
        domain: str,
    ) -> SignalSnapshot:
        """Map numeric series to SignalSnapshot."""
        n = len(values)
        if n == 0:
            return SignalSnapshot.empty()
        
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
        std = math.sqrt(variance)
        noise_ratio = std / abs(mean) if abs(mean) > 1e-9 else 0.0
        
        slope = 0.0
        if n >= 2:
            x_mean = (n - 1) / 2.0
            y_mean = mean
            num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
            den = sum((i - x_mean) ** 2 for i in range(n))
            slope = num / den if abs(den) > 1e-9 else 0.0
        
        curvature = 0.0
        if n >= 3:
            diffs = [values[i+1] - values[i] for i in range(n-1)]
            if len(diffs) >= 2:
                curvature = diffs[-1] - diffs[-2]
        
        regime = _classify_regime_simple(slope, noise_ratio)
        
        extra: Dict[str, Any] = {
            "source": "universal_signal_profiler",
            "input_type": "numeric",
            "has_timestamps": metadata.get("has_timestamps", False),
        }
        
        return SignalSnapshot(
            n_points=n,
            mean=round(mean, 6),
            std=round(std, 6),
            noise_ratio=round(noise_ratio, 6),
            slope=round(slope, 6),
            curvature=round(curvature, 6),
            regime=regime,
            dt=1.0,
            extra=extra,
        )

    def _profile_tabular(
        self,
        data: dict,
        metadata: Dict[str, Any],
        domain: str,
    ) -> SignalSnapshot:
        """Map tabular data to SignalSnapshot.

        Strategy: Profile the first numeric column, aggregate metadata
        """
        numeric_columns = metadata.get("numeric_columns", [])
        
        if not numeric_columns:
            n_rows = metadata.get("n_rows", 0)
            n_cols = metadata.get("n_columns", 0)
            return SignalSnapshot(
                n_points=n_rows,
                mean=0.0,
                std=0.0,
                noise_ratio=0.0,
                slope=0.0,
                curvature=0.0,
                regime=domain,
                dt=1.0,
                extra={
                    "source": "universal_signal_profiler",
                    "input_type": "tabular",
                    "n_columns": n_cols,
                    "numeric_columns": numeric_columns,
                },
            )
        
        first_col = numeric_columns[0]
        values = data.get(first_col, [])
        
        return self._profile_numeric(values, metadata, domain)

    def _profile_mixed(
        self,
        data: Any,
        metadata: Dict[str, Any],
        domain: str,
    ) -> SignalSnapshot:
        """Hybrid profiling for mixed-type data."""
        numeric_count = metadata.get("numeric_count", 0)
        total = metadata.get("n_points", 1)
        
        return SignalSnapshot(
            n_points=total,
            mean=0.0,
            std=0.0,
            noise_ratio=1.0 - (numeric_count / max(total, 1)),
            slope=0.0,
            curvature=0.0,
            regime=domain,
            dt=1.0,
            extra={
                "source": "universal_signal_profiler",
                "input_type": "mixed",
                "numeric_ratio": round(numeric_count / max(total, 1), 3),
            },
        )


def _std(values: List[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _classify_regime_simple(slope: float, noise_ratio: float) -> str:
    """Simple regime classification from slope and noise."""
    if noise_ratio > 0.5:
        return "volatile"
    if abs(slope) > 0.1:
        return "trending"
    if noise_ratio < 0.1:
        return "stable"
    return "transitional"
