"""SituationVectorBuilder — ensambla vector numérico 18-dim del pipeline.

Responsabilidad única: convertir el estado cognitivo completo en un vector
numérico normalizado que fluye por el pipeline sin convertirse a string.

Diseño aditivo: no elimina ni modifica texto legacy.  Si datos faltan,
rellena con 0.0 para mantener dimensionalidad fija.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..entities.explainability.explanation import Explanation


def _clamp01(value: float) -> float:
    """Clamp a [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def _circuit_to_numeric(status: str) -> float:
    """Convert circuit breaker status to continuous value."""
    return {"closed": 0.0, "half_open": 0.5, "open": 1.0}.get(str(status).lower(), 0.0)


_SITUATION_VECTOR_DIM = 18  # 13 activas + 5 reservadas MoE (dims 12-16)


def build_situation_vector(
    explanation: Optional[Explanation],
    metadata: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Construye un vector de situación de 18 dimensiones."""
    vec: List[float] = [0.0] * _SITUATION_VECTOR_DIM

    if explanation is None:
        return vec

    # --- 0-5: Regime vector ---
    signal = explanation.signal
    rv = signal.regime_vector if signal else []
    for i, val in enumerate((rv + [0.0] * 6)[:6]):
        # regime_vector values are raw floats; soft-clamp to [-10, 10]
        # then scale to [0, 1] via tanh-like mapping: (x/(1+|x|) + 1) / 2
        mapped = (val / (1.0 + abs(val)) + 1.0) / 2.0
        vec[i] = round(mapped, 6)

    # --- 6-8: Outcome scores ---
    outcome = explanation.outcome
    vec[6] = _clamp01(outcome.confidence if outcome else 0.0)
    vec[7] = _clamp01(
        outcome.anomaly_score if outcome and outcome.anomaly_score is not None else 0.0
    )
    extra = dict(outcome.extra) if outcome else {}
    vec[8] = _clamp01(extra.get("composite_score", 0.0))

    # --- 9-11: Cognitive trace ---
    trace = explanation.trace
    vec[17] = 0.0
    if trace:
        available = max(1, trace.n_engines_available)
        vec[17] = round(min(1.0, trace.n_engines_active / available), 6)

    # DIMS 12-16: Reservadas para MoE (Mixture of Experts)
    # Infraestructura disponible en infrastructure/ml/moe/
    # pero no cableada al pipeline de inferencia.
    # Cuando se active MoE, AssemblyPhase debe inyectar
    # metadata["moe_vector"] = [top1_prob, top2_prob,
    #   top3_prob, entropy, sparsity_k/10]
    # Hasta entonces estas dimensiones retornan 0.0 (no ruido).

    # cognitive_trace viene en metadata (AssemblyPhase lo pone allí)
    meta = metadata or {}
    cognitive_trace = meta.get("cognitive_trace") or {}
    if isinstance(cognitive_trace, dict):
        drift = float(cognitive_trace.get("drift_score", 0.0))
        vec[9] = _clamp01(drift / 5.0)  # clamp drift to [0,5] then /5
        vec[10] = _circuit_to_numeric(cognitive_trace.get("circuit_breaker_status", "closed"))
        vec[11] = 1.0 if cognitive_trace.get("amnesic_mode", False) else 0.0

    # --- 12-16: MoE vector ---
    moe_vector = meta.get("moe_vector") if meta else None
    if isinstance(moe_vector, list) and len(moe_vector) >= 5:
        vec[12] = _clamp01(moe_vector[0])
        vec[13] = _clamp01(moe_vector[1])
        vec[14] = _clamp01(moe_vector[2])
        vec[15] = _clamp01(moe_vector[3])
        vec[16] = _clamp01(moe_vector[4])

    return vec
