"""Tests for CausalPhase (Redis temporal correlation), ExplainPhase (actionable
summaries), and NarrativeUnificationPhase (3 sources, agreement/contradiction)."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import (
    PipelineContext,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.causal_phase import (
    ANOMALY_HISTORY_PREFIX,
    ANOMALY_TTL_S,
    CausalEvent,
    CausalPhase,
    _check_recent_anomalies,
    _discover_related_params,
    _equipment_id,
    _is_anomaly,
    _save_current_anomaly,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.explain_phase import (
    ExplainPhase,
    _action_text,
    _normal_range,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.narrative_unification_phase import (
    NarrativeUnificationPhase,
)


# ── Fixtures ───────────────────────────────────────────────────────


def _ctx(**overrides) -> PipelineContext:
    class FakeTimer:
        total_ms = 10.0
        budget_ms = 500.0
        is_over_budget = False
        def to_dict(self):
            return {"total_ms": self.total_ms, "budget_ms": self.budget_ms}

    redis = MagicMock()
    store = MagicMock()
    store._redis = redis
    store.is_active = True

    orch = MagicMock()
    orch._series_values_store = store
    orch._state_lock = MagicMock()
    orch._fusion = MagicMock()
    orch._fusion.fuse.return_value = (100.0, 0.85, "stable", {"a": 0.5}, "fusion", "consensus")
    orch._error_history = MagicMock()
    orch._error_history.get_error_dict_for_inhibition.return_value = {}
    orch._correlation_port = None

    profile = MagicMock()
    profile.std = 5.0
    profile.mean = 100.0
    profile.z_score = 3.0
    profile.noise_ratio = 0.1
    profile.stability = 0.9
    profile.unit = "°C"
    profile.regime = "STABLE"
    profile.series_state = "UNKNOWN"
    profile.operational_range = (80.0, 120.0)

    flags = MagicMock()
    flags.get.return_value = True

    ctx = PipelineContext(
        orchestrator=orch,
        values=[101.0, 102.0, 103.0],
        timestamps=[0.0, 1.0, 2.0],
        series_id="evaporator_temperature",
        flags=flags,
        timer=FakeTimer(),
        fused_value=105.0,
        fused_confidence=0.85,
        regime="STABLE",
        data_quality_score=1.0,
        profile=profile,
        causal_events=[],
        max_action="PREDICT",
        consecutive_anomalies=0,
    )
    ctx.feature_context = None
    ctx.neighbors = []
    ctx.neighbor_values = {}
    ctx.explanation = None
    ctx.inhibition_states = []
    ctx.perceptions = []
    ctx.metrics_collector = None
    ctx.selected_engine = "fusion"
    ctx.selection_reason = "consensus"
    ctx.fusion_method = "weighted_average"
    ctx.final_weights = {"a": 1.0}

    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


# ── CausalPhase ────────────────────────────────────────────────────


class TestCausalPhase:
    def test_anomaly_saved_to_redis(self):
        """Anomaly in param A is saved to Redis with correct key."""
        redis = MagicMock()
        ctx = _ctx()
        ctx.orchestrator._series_values_store._redis = redis
        ctx.profile.z_score = 3.5  # Triggers anomaly

        phase = CausalPhase()
        phase.execute(ctx)

        expected_key = f"{ANOMALY_HISTORY_PREFIX}:evaporator:evaporator_temperature"
        assert redis.setex.called
        args = redis.setex.call_args
        assert args[0][0] == expected_key
        assert args[0][1] == ANOMALY_TTL_S

    def test_causal_chain_detected(self):
        """Anomaly in param B after param A → causal_chain detected."""
        redis = MagicMock()
        now = time.time()
        preceding_key = f"{ANOMALY_HISTORY_PREFIX}:evaporator:pressure_evaporator"
        preceding_data = json.dumps({
            "param": "pressure_evaporator",
            "timestamp": now - 7200,  # 2 hours ago
            "z_score": 3.2,
        })
        redis.get.return_value = preceding_data
        redis.scan.return_value = (0, [preceding_key.encode()])

        ctx = _ctx()
        ctx.orchestrator._series_values_store._redis = redis
        ctx.profile.z_score = 3.5
        ctx.series_id = "evaporator_temperature"
        ctx.metadata = {}

        phase = CausalPhase()
        result = phase.execute(ctx)

        events = result.causal_events
        assert len(events) >= 1
        ev = events[0]
        assert ev["preceding_param"] == "pressure_evaporator"
        assert ev["current_param"] == "evaporator_temperature"
        assert ev["time_delta_minutes"] == pytest.approx(120.0, abs=5)

    def test_no_anomaly_no_causal_chain(self):
        """No anomaly → no events saved and no causal chains."""
        redis = MagicMock()
        ctx = _ctx()
        ctx.orchestrator._series_values_store._redis = redis
        ctx.profile.z_score = 0.5  # Normal

        phase = CausalPhase()
        result = phase.execute(ctx)
        assert len(result.causal_events) == 0

    def test_no_redis_returns_early(self):
        """No Redis store → phase returns early without error."""
        ctx = _ctx()
        ctx.orchestrator._series_values_store = None
        phase = CausalPhase()
        result = phase.execute(ctx)
        assert result is not None


# ── ExplainPhase ───────────────────────────────────────────────────


class TestExplainPhase:
    def test_explanation_summary_format(self):
        """explanation_summary has the actionable format."""
        phase = ExplainPhase()
        ctx = _ctx()
        ctx.fused_value = 105.0
        ctx.regime = "STABLE"
        ctx.max_action = "PREDICT"
        ctx.causal_events = []
        ctx.timer.is_over_budget = False

        result = phase.execute(ctx)
        summary = result.explanation_summary
        assert summary is not None
        assert "Equipo" in summary
        assert "evaporator_temperature" in summary
        assert "105" in summary  # fused_value
        assert "°C" in summary
        assert "normal: 80.0-120.0" in summary
        assert "Régimen: STABLE" in summary
        assert "Acción recomendada: Sin acción requerida" in summary

    def test_action_text_escale(self):
        """ESCALATE → 'Revisar equipo inmediatamente'."""
        assert _action_text("ESCALATE") == "Revisar equipo inmediatamente"
        assert _action_text("INVESTIGATE") == "Programar inspección en próximas 24h"
        assert _action_text("MONITOR") == "Monitorear cada hora"
        assert _action_text("LOG_ONLY") == "Sin acción requerida"

    def test_causal_chain_in_summary(self):
        """Causal events appear in the explanation_summary."""
        phase = ExplainPhase()
        ctx = _ctx()
        ctx.causal_events = [{
            "preceding_param": "pressure_evaporator",
            "time_delta_minutes": 120.0,
            "current_param": "evaporator_temperature",
            "preceding_timestamp": 0.0,
            "current_timestamp": 7200.0,
            "correlation_strength": 0.7,
        }]
        ctx.timer.is_over_budget = False

        result = phase.execute(ctx)
        summary = result.explanation_summary
        assert "Precedido por anomalía en pressure_evaporator" in summary
        assert "120.0 minutos" in summary

    def test_consecutive_anomalies_trend_in_explanation(self):
        """Explanation dict contains consecutive_anomalies_trend."""
        phase = ExplainPhase()
        ctx = _ctx()
        ctx.profile.z_score = 3.0
        ctx.timer.is_over_budget = False

        result = phase.execute(ctx)
        assert "consecutive_anomalies_trend" in result.explanation

    def test_no_rul_heuristic(self):
        """RUL code should no longer be called."""
        phase = ExplainPhase()
        ctx = _ctx()
        ctx.timer.is_over_budget = False
        result = phase.execute(ctx)
        assert result is not None


# ── NarrativeUnificationPhase ──────────────────────────────────────


class TestNarrativeUnificationPhase:
    def test_three_sources_used_when_all_available(self):
        """With prediction, anomaly, and causal → all 3 sources used."""
        phase = NarrativeUnificationPhase()
        ctx = _ctx()
        ctx.explanation = {
            "narratives": ["predicción normal"],
            "selected_engine": "fusion",
        }
        ctx.causal_events = [{
            "preceding_param": "p1",
            "time_delta_minutes": 30.0,
            "current_param": "p2",
            "preceding_timestamp": 0.0,
            "current_timestamp": 1800.0,
            "correlation_strength": 0.7,
        }]
        ctx.max_action = "PREDICT"
        ctx.fused_confidence = 0.85
        ctx.profile.z_score = 1.0

        result = phase.execute(ctx)
        assert result.unified_narrative is not None
        sources = result.unified_narrative.sources_used
        assert len(sources) >= 2

    def test_agreement_boosts_confidence(self):
        """Prediction and anomaly agree → confidence boost."""
        phase = NarrativeUnificationPhase()
        ctx = _ctx()
        ctx.explanation = {
            "narratives": ["alerta crítica"],
            "selected_engine": "fusion",
        }
        ctx.causal_events = []
        ctx.max_action = "ESCALATE"
        ctx.fused_confidence = 0.7
        ctx.profile.z_score = 5.0
        ctx.consecutive_anomalies = 3

        result = phase.execute(ctx)
        # Both should be CRITICAL → boost
        assert result.unified_narrative.confidence >= 0.7

    def test_contradiction_detected(self):
        """Prediction says NORMAL, anomaly says CRITICAL → contradiction."""
        phase = NarrativeUnificationPhase()
        ctx = _ctx()
        ctx.explanation = {
            "narratives": ["todo normal"],
            "selected_engine": "fusion",
        }
        ctx.causal_events = []
        ctx.max_action = "PREDICT"
        ctx.fused_confidence = 0.9
        ctx.profile.z_score = 5.0
        ctx.consecutive_anomalies = 3

        result = phase.execute(ctx)
        if result.unified_narrative.contradictions:
            assert any("prediction" in c and "anomaly" in c for c in result.unified_narrative.contradictions)

    def test_confidence_penalty_on_contradiction(self):
        """Contradiction → confidence reduced by 0.1."""
        phase = NarrativeUnificationPhase()
        ctx = _ctx()
        ctx.explanation = {
            "narratives": ["todo normal"],
            "selected_engine": "fusion",
        }
        ctx.causal_events = []
        ctx.max_action = "PREDICT"
        ctx.fused_confidence = 0.9
        ctx.profile.z_score = 5.0
        ctx.consecutive_anomalies = 3

        result = phase.execute(ctx)
        if result.unified_narrative.contradictions:
            assert result.unified_narrative.confidence <= 0.8
