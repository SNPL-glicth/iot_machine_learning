#!/usr/bin/env python3
"""Quick debug script for SanitizePhase."""

import sys
import os

# Add parent directory to path so iot_machine_learning is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from iot_machine_learning.infrastructure.ml.cognitive.sanitize.phase import (
    SanitizePhase,
    SanitizeConfig,
    LocalWindowStatisticsProvider,
    SeriesStatisticsProvider,
)


@dataclass
class FakeCtx:
    orchestrator: Any = None
    values: List[float] = field(default_factory=list)
    timestamps: Optional[List[float]] = None
    series_id: str = ""
    flags: Any = None
    timer: Any = None
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sanitized_values: Optional[List[float]] = None
    sanitization_flags: List[str] = field(default_factory=list)
    degradation_reasons: List[str] = field(default_factory=list)
    is_fallback: bool = False
    fallback_reason: Optional[str] = None

    def with_field(self, **kwargs):
        current = {k: getattr(self, k) for k in self.__dataclass_fields__}
        current.update(kwargs)
        return FakeCtx(**current)


def test_nan():
    phase = SanitizePhase()
    ctx = FakeCtx(values=[1.0, float("nan"), 3.0], series_id="s-1")
    result = phase.execute(ctx)
    print("NAN test:")
    print(f"  is_fallback: {result.is_fallback}")
    print(f"  fallback_reason: {result.fallback_reason}")
    print(f"  sanitization_flags: {result.sanitization_flags}")
    print(f"  sanitized_values: {result.sanitized_values}")
    assert result.is_fallback is True
    assert result.fallback_reason == "nan_or_inf_rejected"
    assert "nan_or_inf_rejected" in result.sanitization_flags
    assert result.sanitized_values == []
    print("  PASSED")


def test_inf():
    phase = SanitizePhase()
    ctx = FakeCtx(values=[1.0, float("inf"), 3.0], series_id="s-1")
    result = phase.execute(ctx)
    print("INF test:")
    print(f"  is_fallback: {result.is_fallback}")
    print(f"  fallback_reason: {result.fallback_reason}")
    print(f"  sanitization_flags: {result.sanitization_flags}")
    assert result.is_fallback is True
    assert result.fallback_reason == "nan_or_inf_rejected"
    assert "nan_or_inf_rejected" in result.sanitization_flags
    print("  PASSED")


def test_neg_inf():
    phase = SanitizePhase()
    ctx = FakeCtx(values=[1.0, float("-inf"), 3.0], series_id="s-1")
    result = phase.execute(ctx)
    print("NEG_INF test:")
    print(f"  is_fallback: {result.is_fallback}")
    print(f"  fallback_reason: {result.fallback_reason}")
    print(f"  sanitization_flags: {result.sanitization_flags}")
    assert result.is_fallback is True
    assert result.fallback_reason == "nan_or_inf_rejected"
    print("  PASSED")


def test_clamp():
    class Provider(SeriesStatisticsProvider):
        def get_statistics(self, series_id: str):
            return (10.0, 1.0)

    phase = SanitizePhase(statistics_provider=Provider())
    ctx = FakeCtx(values=[10.0, 10.0, 10.0, 17.0], series_id="s-1")
    result = phase.execute(ctx)
    print("CLAMP test:")
    print(f"  values: {result.values}")
    print(f"  sanitization_flags: {result.sanitization_flags}")
    assert result.values == [10.0, 10.0, 10.0, 16.0]
    assert "value_clamped:1" in result.sanitization_flags
    print("  PASSED")


def test_clean():
    phase = SanitizePhase()
    ctx = FakeCtx(values=[10.0, 11.0, 12.0, 10.5, 11.5], series_id="s-1")
    result = phase.execute(ctx)
    print("CLEAN test:")
    print(f"  values: {result.values}")
    print(f"  sanitization_flags: {result.sanitization_flags}")
    assert result.values == [10.0, 11.0, 12.0, 10.5, 11.5]
    assert result.sanitization_flags == []
    print("  PASSED")


if __name__ == "__main__":
    test_nan()
    test_inf()
    test_neg_inf()
    test_clamp()
    test_clean()
    print("\nAll debug tests passed!")
