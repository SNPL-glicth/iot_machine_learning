"""MoE Configuration — pesos externalizados y parámetros operativos.

Los pesos de gating se configuran aquí, NO hardcodeados en ContextualRegimeGating.
Para ajustar según resultados de A/B testing, modificar este archivo o cargar
sobreescrituras desde variables de entorno.

Estructura:
    REGIME_WEIGHTS: Dict[str, Dict[str, float]]
        {regime: {expert_id: weight}}
    DISPATCH_TIMEOUT_MS: int — timeout por experto
    DISCREPANCY_THRESHOLD: float — umbral para penalizar confianza
    SPARSITY_K: int — número de expertos top-k
    SHADOW_GATING_ENABLED: bool — activa TreeGatingNetwork en shadow mode

Ejemplo de ajuste por A/B:
    Si el A/B muestra que kalman supera a baseline en "stable",
    incrementar el peso de kalman y reducir baseline en stable.
"""

from __future__ import annotations

from typing import Dict
import os


# ------------------------------------------------------------------
# Regime weights
# ------------------------------------------------------------------
# Suma de pesos por régimen puede ser != 1.0; se normalizan internamente.
DEFAULT_REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "stable": {
        "baseline": 0.80,
        "statistical": 0.15,
        "taylor": 0.05,
        "kalman": 0.00,
    },
    "trending": {
        "baseline": 0.05,
        "statistical": 0.55,
        "taylor": 0.35,
        "kalman": 0.05,
    },
    "volatile": {
        "baseline": 0.05,
        "statistical": 0.25,
        "taylor": 0.50,
        "kalman": 0.20,
    },
    "noisy": {
        "baseline": 0.10,
        "statistical": 0.20,
        "taylor": 0.20,
        "kalman": 0.50,
    },
}

# Override por variable de entorno (JSON string, parseado si existe)
def _load_override_weights() -> Dict[str, Dict[str, float]]:
    import json
    env = os.getenv("MOE_REGIME_WEIGHTS")
    if env:
        try:
            return json.loads(env)
        except Exception:
            pass
    return {}

REGIME_WEIGHTS = _load_override_weights() or DEFAULT_REGIME_WEIGHTS

# ------------------------------------------------------------------
# Operational params
# ------------------------------------------------------------------
DISPATCH_TIMEOUT_MS = int(os.getenv("MOE_DISPATCH_TIMEOUT_MS", "200"))
DISCREPANCY_THRESHOLD = float(os.getenv("MOE_DISCREPANCY_THRESHOLD", "2.0"))
SPARSITY_K = int(os.getenv("MOE_SPARSITY_K", "2"))

# ------------------------------------------------------------------
# Shadow mode
# ------------------------------------------------------------------
SHADOW_GATING_ENABLED = os.getenv("MOE_SHADOW_GATING", "false").lower() == "true"

# ------------------------------------------------------------------
# A/B test logging
# ------------------------------------------------------------------
AB_LOG_ENABLED = os.getenv("MOE_AB_LOG_ENABLED", "false").lower() == "true"
