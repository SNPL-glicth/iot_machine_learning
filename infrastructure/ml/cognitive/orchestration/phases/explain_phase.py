"""Explain Phase — narratives that an operator (Erick) can read and act on.

Uses causal_events from CausalPhase to enrich the explanation with
temporal anomaly chains.  Generates an actionable explanation_summary.

Replaces the old RUL heuristic with a ``consecutive_anomalies_trend``
derived from the last 5 anomaly timestamps stored in Redis.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .context import PipelineContext
from iot_machine_learning.application.explainability.explanation_renderer import ExplanationRenderer

from ...analysis.types import MetaDiagnostic

try:
    from ...observability import ExplainabilityValidator
except (ImportError, ModuleNotFoundError):
    ExplainabilityValidator = None  # type: ignore[assignment,misc]

try:
    from ...explainability import ContextualExplainabilityEngine
except (ImportError, ModuleNotFoundError):
    ContextualExplainabilityEngine = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_ANOMALY_TREND_KEY_PREFIX = "zenin:anomaly_trend"
_ANOMALY_TREND_TTL_S = 86400 * 7  # 7 days
_ANOMALY_TREND_MAX = 5


def _action_text(max_action: str) -> str:
    return {
        "ESCALATE": "Revisar equipo inmediatamente",
        "INVESTIGATE": "Programar inspección en próximas 24h",
        "MONITOR": "Monitorear cada hora",
        "LOG_ONLY": "Sin acción requerida",
    }.get(max_action, "Sin acción requerida")


def _normal_range(profile: Any) -> Tuple[float, float]:
    """Return (low, high) normal range from profile."""
    if profile is not None and hasattr(profile, "operational_range"):
        return profile.operational_range
    mean = getattr(profile, "mean", 50.0)
    std = getattr(profile, "std", 10.0)
    return (round(mean - 2 * std, 2), round(mean + 2 * std, 2))


def _consecutive_anomalies_trend(ctx: PipelineContext) -> str:
    """Determine trend based on last 5 anomaly timestamps in Redis."""
    store = getattr(getattr(ctx, "orchestrator", None), "_series_values_store", None)
    redis = store._redis if store and hasattr(store, "_redis") else None
    if redis is None:
        return "estable"
    key = f"{_ANOMALY_TREND_KEY_PREFIX}:{ctx.series_id}"
    try:
        raw = redis.lrange(key, 0, _ANOMALY_TREND_MAX - 1)
        if not raw:
            return "estable"
        scores = []
        for item in raw:
            try:
                data = json.loads(item)
                scores.append(float(data.get("z_score", 0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        if len(scores) < 2:
            return "estable"
        # Compute trend: average of last 3 differences
        diffs = [scores[i] - scores[i + 1] for i in range(min(3, len(scores) - 1))]
        avg_diff = sum(diffs) / len(diffs)
        if avg_diff > 0.5:
            return "aumentando"
        if avg_diff < -0.5:
            return "disminuyendo"
        return "estable"
    except Exception:
        return "estable"


def _save_consecutive_anomaly(ctx: PipelineContext, z_score: float) -> None:
    """Append current anomaly to the trend list in Redis."""
    store = getattr(getattr(ctx, "orchestrator", None), "_series_values_store", None)
    redis = store._redis if store and hasattr(store, "_redis") else None
    if redis is None:
        return
    key = f"{_ANOMALY_TREND_KEY_PREFIX}:{ctx.series_id}"
    payload = json.dumps({"z_score": round(z_score, 4), "ts": time.time()})
    try:
        redis.lpush(key, payload)
        redis.ltrim(key, 0, _ANOMALY_TREND_MAX - 1)
        redis.expire(key, _ANOMALY_TREND_TTL_S)
    except Exception:
        pass


class CausalNarrativeBuilder:
    @staticmethod
    def from_signal_profile(profile) -> List[str]:
        narratives = []
        if profile is None:
            return narratives
        z_score = getattr(profile, "z_score", None)
        if z_score is not None and z_score > 2.5:
            narratives.append("Detección de anomalía por cambio súbito de magnitud")
        stability = getattr(profile, "stability", None)
        if stability is not None and stability < 0.3:
            narratives.append("Predicción conservadora debido a alta inestabilidad en la señal")
        regime = getattr(profile, "regime", None)
        if regime == "VOLATILE":
            narratives.append("Alta volatilidad detectada: adaptando pesos dinámicamente")
        elif regime == "TRENDING":
            trend_dir = getattr(profile, "trend_direction", "up")
            narratives.append(f"Tendencia {trend_dir} establecida: extrapolando momentum")
        noise_ratio = getattr(profile, "noise_ratio", None)
        if noise_ratio is not None and noise_ratio > 0.3:
            narratives.append("Señal con ruido significativo: aplicando filtrado adaptativo")
        return narratives

    @staticmethod
    def from_perceptions(perceptions) -> List[str]:
        narratives = []
        if not perceptions:
            return narratives
        inhibited = [p for p in perceptions if getattr(p, "inhibited", False)]
        active = [p for p in perceptions if not getattr(p, "inhibited", False)]
        if len(inhibited) > len(active):
            narratives.append(
                f"Mayoría de engines inhibidos ({len(inhibited)}/{len(perceptions)}): "
                "usando fallback conservador"
            )
        if len(active) == 1:
            narratives.append(f"Engine único activo: {active[0].engine_name}")
        return narratives

    @staticmethod
    def from_causal_events(causal_events: List[Dict]) -> List[str]:
        narratives = []
        for ev in causal_events:
            narratives.append(
                f"Precedido por anomalía en {ev['preceding_param']} "
                f"hace {ev['time_delta_minutes']} minutos"
            )
        return narratives


class ExplainPhase:
    """Phase 9: Explanation and diagnostic generation with actionable summaries."""

    def __init__(
        self,
        explainability_validator: Optional[Any] = None,
        contextual_explainability_engine: Optional[Any] = None,
    ) -> None:
        self._explainability_validator = explainability_validator
        if ExplainabilityValidator is not None and self._explainability_validator is None:
            try:
                self._explainability_validator = ExplainabilityValidator()
            except Exception:
                pass
        self._contextual_explainability_engine = contextual_explainability_engine
        if ContextualExplainabilityEngine is not None and self._contextual_explainability_engine is None:
            try:
                self._contextual_explainability_engine = ContextualExplainabilityEngine()
            except Exception:
                pass

    @property
    def name(self) -> str:
        return "explain"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        orchestrator = ctx.orchestrator

        # ── Narratives ──────────────────────────────────────────
        signal_narratives = CausalNarrativeBuilder.from_signal_profile(ctx.profile)
        perception_narratives = CausalNarrativeBuilder.from_perceptions(ctx.perceptions)
        causal_events = getattr(ctx, "causal_events", []) or []
        causal_narratives = CausalNarrativeBuilder.from_causal_events(causal_events)
        all_narratives = signal_narratives + perception_narratives + causal_narratives

        # ── Meta diagnostic ─────────────────────────────────────
        diag = MetaDiagnostic(
            signal_profile=ctx.profile,
            perceptions=ctx.perceptions,
            inhibition_states=ctx.inhibition_states,
            final_weights=ctx.final_weights,
            selected_engine=ctx.selected_engine,
            selection_reason=ctx.selection_reason,
            fusion_method=ctx.fusion_method,
        )

        explanation_dict: Dict[str, Any] = {
            "selected_engine": ctx.selected_engine,
            "selection_reason": ctx.selection_reason,
            "regime": ctx.regime,
            "narratives": all_narratives,
            "n_engines_active": len([p for p in (ctx.perceptions or [])
                                    if not getattr(p, "inhibited", False)]),
            "n_engines_inhibited": len([p for p in (ctx.perceptions or [])
                                       if getattr(p, "inhibited", False)]),
            "causal_events": causal_events,
        }

        # ── Actionable explanation_summary ──────────────────────
        low, high = _normal_range(ctx.profile)
        unit = getattr(ctx.profile, "unit", "") if ctx.profile else ""
        action = _action_text(ctx.max_action)
        eq = ctx.series_id.split("_")[0] if "_" in ctx.series_id else ctx.series_id

        parts = [
            f"Equipo {eq} — {ctx.series_id}: "
            f"valor {ctx.fused_value} {unit} "
            f"(normal: {low}-{high}). "
            f"Régimen: {ctx.regime}."
        ]
        for ev in causal_events:
            parts.append(
                f"Precedido por anomalía en {ev['preceding_param']} "
                f"hace {ev['time_delta_minutes']} minutos."
            )
        parts.append(f"Acción recomendada: {action}")

        # ── Consecutive anomalies trend ─────────────────────────
        if ctx.profile:
            z = abs(getattr(ctx.profile, "z_score", 0.0))
            if z > 2.5:
                _save_consecutive_anomaly(ctx, z)
        trend = _consecutive_anomalies_trend(ctx)
        explanation_dict["consecutive_anomalies_trend"] = trend

        explanation_summary = " ".join(parts)

        # ── State update & storage ──────────────────────────────
        if hasattr(orchestrator, "update_series_state"):
            orchestrator.update_series_state(
                series_id=ctx.series_id,
                regime=ctx.regime,
                perceptions=list(ctx.perceptions) if ctx.perceptions else [],
            )

        with orchestrator._state_lock:
            orchestrator._last_diagnostic = diag
            orchestrator._last_explanation = explanation_dict
            orchestrator._last_timer = ctx.timer

        logger.debug("cognitive_prediction", extra={
            "n_engines": len(ctx.perceptions) if ctx.perceptions else 0,
            "selected": ctx.selected_engine,
            "regime": ctx.regime,
            "fused_value": round(ctx.fused_value, 4) if ctx.fused_value else None,
            "pipeline_ms": round(ctx.timer.total_ms, 2),
            "narratives": all_narratives,
        })

        if ctx.timer and getattr(ctx.timer, "is_over_budget", False):
            logger.warning("pipeline_over_budget", extra=ctx.timer.to_dict())

        # ── Explainability validation ───────────────────────────
        validation_result = None
        if self._explainability_validator is not None:
            try:
                from domain.entities.explainability import ContextualExplanation
                simple_explanation = ContextualExplanation(
                    sensor_id=ctx.series_id,
                    sensor_type="generic",
                    current_regime=ctx.regime or "unknown",
                    anomaly_score=getattr(ctx.profile, "z_score", 0.0) if ctx.profile else 0.0,
                    operational_confidence=ctx.fused_confidence or 0.0,
                    primary_drivers=[ctx.selected_engine] if ctx.selected_engine else [],
                    suggested_actions=all_narratives[:3],
                    timestamp=time.time(),
                )
                validation_result = self._explainability_validator.validate_explanation(
                    explanation=simple_explanation,
                    retrieval_relevance=ctx.fused_confidence or 0.0,
                )
            except Exception as e:
                logger.debug(f"explainability_validation_failed: {e}")

        # ── Contextual explanation ──────────────────────────────
        contextual_explanation = None
        if self._contextual_explainability_engine is not None and ctx.memory_registry is not None:
            try:
                contextual_explanation = self._contextual_explainability_engine.generate_explanation(
                    sensor_id=ctx.series_id,
                    regime=ctx.regime or "unknown",
                    anomaly_score=getattr(ctx.profile, "z_score", 0.0) if ctx.profile else 0.0,
                    confidence=ctx.fused_confidence or 0.0,
                    memory_registry=ctx.memory_registry,
                )
                if contextual_explanation:
                    all_narratives.extend(contextual_explanation.get("narratives", []))
            except Exception as e:
                logger.debug(f"contextual_explanation_failed: {e}")

        return ctx.with_field(
            diagnostic=diag,
            explanation=explanation_dict,
            explanation_summary=explanation_summary,
            validation_result=validation_result,
            contextual_explanation=contextual_explanation,
        )
