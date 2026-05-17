"""Tests cierre final — Fases 0-4 gaps críticos."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from iot_machine_learning.domain.entities.series.structural_analysis import (
    RegimeType,
    _classify_regime,
)
from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception,
)
from iot_machine_learning.infrastructure.ml.cognitive.fusion.hampel_filter import (
    hampel_filter_with_profile,
)
from iot_machine_learning.infrastructure.ml.moe.events.prediction_drift_detector import (
    PredictionDriftDetector,
)
from iot_machine_learning.infrastructure.ml.moe.feature_context import FeatureContext
from iot_machine_learning.infrastructure.ml.moe.gating.contextual_regime import (
    ContextualRegimeGating,
)


# --- Hampel filter ---

@dataclass(frozen=True)
class _MockProfile:
    hampel_k: float = 3.0
    hampel_window: int = 10


def _perceptions(values: List[float]) -> List[EnginePerception]:
    return [EnginePerception(engine_name=f"e{i}", predicted_value=v, confidence=0.5) for i, v in enumerate(values)]


def test_hampel_uses_profile_window_llenadora() -> None:
    profile = _MockProfile(hampel_k=4.0, hampel_window=20)
    p = _perceptions([1.0] * 5)
    r = hampel_filter_with_profile(p, sensor_profile=profile)
    assert len(r.kept) == 5  # window=20 > len=5, MAD=0 → no filter
    p25 = _perceptions([1.0] * 12 + [2.0] * 12 + [100.0])
    r25 = hampel_filter_with_profile(p25, sensor_profile=profile)
    assert len(r25.rejected) == 1


def test_hampel_uses_default_window_without_profile() -> None:
    p = _perceptions([1.0] * 3)
    r = hampel_filter_with_profile(p, sensor_profile=None)
    assert len(r.kept) == 3  # min_perceptions=3 default


# --- ContextualRegimeGating ---

@pytest.fixture
def gating() -> ContextualRegimeGating:
    return ContextualRegimeGating(expert_ids=["baseline", "statistical", "taylor", "kalman"])


def _fc(regime: str, equipment_class: str = "GENERIC") -> FeatureContext:
    return FeatureContext(regime=regime, mean=10.0, std=0.1, slope=0.0, curvature=0.0, noise_ratio=0.01, stability=0.0, hampel_outlier_mask=[], spatial_correlation_score=0.0, equipment_class=equipment_class)


def test_gating_pasteurizer_stable_taylor_dominates(gating: ContextualRegimeGating) -> None:
    probs = gating.route(_fc("stable", "PASTEURIZER"))
    assert probs.top_expert == "taylor"
    assert probs.probabilities["taylor"] > 0.5


def test_gating_filler_stable_statistical_dominates(gating: ContextualRegimeGating) -> None:
    probs = gating.route(_fc("stable", "FILLER"))
    assert probs.top_expert == "statistical"
    assert probs.probabilities["statistical"] > 0.6
    assert probs.probabilities["taylor"] == 0.0


def test_gating_filler_volatile_taylor_zero(gating: ContextualRegimeGating) -> None:
    probs = gating.route(_fc("volatile", "FILLER"))
    assert probs.probabilities["taylor"] == 0.0


def test_gating_generic_uses_global_weights(gating: ContextualRegimeGating) -> None:
    probs = gating.route(_fc("stable", "GENERIC"))
    assert probs.top_expert == "baseline"
    assert pytest.approx(probs.probabilities["baseline"], 0.01) == 0.80


def test_gating_metadata_includes_equipment_class(gating: ContextualRegimeGating) -> None:
    probs = gating.route(_fc("stable", "PASTEURIZER"))
    assert probs.metadata["equipment_class"] == "PASTEURIZER"
    assert probs.metadata["used_equipment_override"] is True


# --- _classify_regime ---

def test_classify_regime_night_escalation_high_noise() -> None:
    assert _classify_regime(0.3, 0.0, 1.0, 10.0, hour_of_day=23) == RegimeType.NOISY


def test_classify_regime_night_no_escalation_low_noise() -> None:
    assert _classify_regime(0.17, 0.0, 1.0, 10.0, hour_of_day=23) == RegimeType.VOLATILE


def test_classify_regime_day_volatile_stays_volatile() -> None:
    assert _classify_regime(0.3, 0.0, 1.0, 10.0, hour_of_day=14) == RegimeType.VOLATILE


# --- PredictionDriftDetector ---

def test_drift_alert_carries_equipment_class() -> None:
    detector = PredictionDriftDetector()
    for i in range(40):
        detector.record_error("s1", 1.0 + (i % 3) * 0.1, equipment_class="PASTEURIZER")
    for _ in range(20):
        alert = detector.record_error("s1", 10.0, equipment_class="PASTEURIZER")
    assert alert is not None
    assert alert.equipment_class == "PASTEURIZER"
