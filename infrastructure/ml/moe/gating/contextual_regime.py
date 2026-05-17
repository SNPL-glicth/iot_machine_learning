"""ContextualRegimeGating — routing basado en features numéricos del pipeline.

Reemplaza RegimeBasedGating:
- Usa FeatureContext completo (std, slope, noise_ratio), no solo regime string.
- Pesos externalizados a dict/configurable, NO hardcoded en código.
- Determinista y explicable (explain() retorna string legible).
- Diseño Strategy Pattern: implementa GatingStrategy Protocol.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .base import GatingProbs
from ..feature_context import FeatureContext


def _load_regime_weights() -> Dict[str, Dict[str, float]]:
    """Carga pesos desde config externa; fallback a defaults si no existe."""
    try:
        from ..config.moe_config import REGIME_WEIGHTS
        return REGIME_WEIGHTS
    except Exception:
        pass
    return {
        "stable": {"baseline": 0.80, "statistical": 0.15, "taylor": 0.05, "kalman": 0.00},
        "trending": {"baseline": 0.05, "statistical": 0.55, "taylor": 0.35, "kalman": 0.05},
        "volatile": {"baseline": 0.05, "statistical": 0.25, "taylor": 0.50, "kalman": 0.20},
        "noisy": {"baseline": 0.10, "statistical": 0.20, "taylor": 0.20, "kalman": 0.50},
    }


EQUIPMENT_REGIME_WEIGHTS: Dict[str, Dict[str, Dict[str, float]]] = {
    "PASTEURIZER": {
        "stable": {"baseline": 0.10, "statistical": 0.10, "taylor": 0.65, "kalman": 0.15},
        "trending": {"baseline": 0.05, "statistical": 0.10, "taylor": 0.75, "kalman": 0.10},
        "volatile": {"baseline": 0.05, "statistical": 0.10, "taylor": 0.55, "kalman": 0.30},
        "noisy": {"baseline": 0.05, "statistical": 0.10, "taylor": 0.30, "kalman": 0.55},
    },
    "CIP": {
        "stable": {"baseline": 0.60, "statistical": 0.20, "taylor": 0.10, "kalman": 0.10},
        "trending": {"baseline": 0.50, "statistical": 0.20, "taylor": 0.10, "kalman": 0.20},
        "volatile": {"baseline": 0.55, "statistical": 0.25, "taylor": 0.05, "kalman": 0.15},
        "noisy": {"baseline": 0.50, "statistical": 0.20, "taylor": 0.05, "kalman": 0.25},
    },
    "FILLER": {
        "stable": {"baseline": 0.10, "statistical": 0.70, "taylor": 0.00, "kalman": 0.20},
        "trending": {"baseline": 0.05, "statistical": 0.65, "taylor": 0.00, "kalman": 0.30},
        "volatile": {"baseline": 0.05, "statistical": 0.70, "taylor": 0.00, "kalman": 0.25},
        "noisy": {"baseline": 0.05, "statistical": 0.55, "taylor": 0.00, "kalman": 0.40},
    },
    "CONVEYOR": {
        "stable": {"baseline": 0.15, "statistical": 0.20, "taylor": 0.10, "kalman": 0.55},
        "trending": {"baseline": 0.10, "statistical": 0.20, "taylor": 0.20, "kalman": 0.50},
        "volatile": {"baseline": 0.10, "statistical": 0.20, "taylor": 0.10, "kalman": 0.60},
        "noisy": {"baseline": 0.05, "statistical": 0.15, "taylor": 0.05, "kalman": 0.75},
    },
    "SILO": {
        "stable": {"baseline": 0.10, "statistical": 0.15, "taylor": 0.65, "kalman": 0.10},
        "trending": {"baseline": 0.05, "statistical": 0.10, "taylor": 0.75, "kalman": 0.10},
        "volatile": {"baseline": 0.10, "statistical": 0.20, "taylor": 0.55, "kalman": 0.15},
        "noisy": {"baseline": 0.10, "statistical": 0.20, "taylor": 0.40, "kalman": 0.30},
    },
    "PET_BLOWER": {
        "stable": {"baseline": 0.10, "statistical": 0.20, "taylor": 0.20, "kalman": 0.50},
        "trending": {"baseline": 0.05, "statistical": 0.15, "taylor": 0.25, "kalman": 0.55},
        "volatile": {"baseline": 0.05, "statistical": 0.15, "taylor": 0.15, "kalman": 0.65},
        "noisy": {"baseline": 0.05, "statistical": 0.10, "taylor": 0.10, "kalman": 0.75},
    },
}


class ContextualRegimeGating:
    """Gating que utiliza features numéricos del pipeline para routing.

    Attributes:
        _regime_weights: Dict {regime: {expert_id: weight}} externalizado.
        _expert_ids: Lista de expertos conocidos.
        _noise_boost_factor: Multiplicador de peso cuando noise_ratio es alto.
        _slope_threshold: Umbral para considerar tendencia significativa.
    """

    # Pesos por defecto — cargados desde config externa
    DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = _load_regime_weights()

    def __init__(
        self,
        regime_weights: Optional[Dict[str, Dict[str, float]]] = None,
        expert_ids: Optional[List[str]] = None,
        noise_boost_factor: float = 1.5,
        slope_threshold: float = 0.01,
    ) -> None:
        """Inicializa gating con pesos externalizados.

        Args:
            regime_weights: Dict {regime: {expert_id: weight}}.
                Si None, usa DEFAULT_WEIGHTS.
            expert_ids: Lista de expertos registrados.
            noise_boost_factor: Factor multiplicador para expertos de noisy.
            slope_threshold: Umbral mínimo de slope para trending.
        """
        self._regime_weights = regime_weights or dict(self.DEFAULT_WEIGHTS)
        self._expert_ids = expert_ids or []
        self._noise_boost_factor = noise_boost_factor
        self._slope_threshold = slope_threshold

    def route(self, feature_context: FeatureContext) -> GatingProbs:
        """Decide distribución de probabilidades usando features numéricos.

        Lookup por (equipment_class, regime) con fallback a pesos globales.
        """
        regime = feature_context.regime
        equipment_class = getattr(feature_context, "equipment_class", "GENERIC")

        equipment_override = EQUIPMENT_REGIME_WEIGHTS.get(equipment_class, {})
        base_weights = equipment_override.get(regime) or self._regime_weights.get(regime, {})

        # Inicializar pesos para todos los expertos conocidos
        weights: Dict[str, float] = {
            eid: base_weights.get(eid, 0.0) for eid in self._expert_ids
        }

        # Ajuste por std (volatilidad)
        weights = self._adjust_by_std(weights, feature_context.std)

        # Ajuste por slope (tendencia)
        weights = self._adjust_by_slope(weights, feature_context.slope)

        # Ajuste por noise_ratio (ruido)
        weights = self._adjust_by_noise(weights, feature_context.noise_ratio)

        # Normalizar a probabilidades
        probabilities = self._normalize(weights)

        # Entropía como medida de incertidumbre
        entropy = self._compute_entropy(probabilities)

        top_expert = max(probabilities.items(), key=lambda x: x[1])[0] if probabilities else ""

        return GatingProbs(
            probabilities=probabilities,
            entropy=entropy,
            top_expert=top_expert,
            metadata={
                "regime": regime,
                "std": feature_context.std,
                "slope": feature_context.slope,
                "noise_ratio": feature_context.noise_ratio,
                "raw_weights": weights,
                "equipment_class": equipment_class,
                "used_equipment_override": equipment_class in EQUIPMENT_REGIME_WEIGHTS,
            },
        )

    def explain(
        self, feature_context: FeatureContext, probs: GatingProbs
    ) -> str:
        """Explica la decisión de routing en lenguaje humano.

        Args:
            feature_context: Contexto usado.
            probs: Probabilidades resultantes.

        Returns:
            String legible explicando la decisión.
        """
        regime = feature_context.regime
        top = probs.top_expert
        top_prob = probs.max_probability

        reasons = []

        # Régimen base
        reasons.append(f"régimen={regime}")

        # Std
        if feature_context.std > 2.0:
            reasons.append(f"alta_volatilidad(std={feature_context.std:.2f})")
        elif feature_context.std < 0.5:
            reasons.append(f"baja_volatilidad(std={feature_context.std:.2f})")

        # Slope
        if abs(feature_context.slope) > self._slope_threshold:
            reasons.append(f"tendencia(slope={feature_context.slope:.4f})")

        # Noise
        if feature_context.noise_ratio > 0.3:
            reasons.append(f"ruido_alto(noise={feature_context.noise_ratio:.2f})")

        return (
            f"ContextualRegimeGating: top_expert={top}({top_prob:.2f}) "
            f"basado en {', '.join(reasons)}. "
            f"Entropía={probs.entropy:.3f}"
        )

    def get_expert_ids(self) -> List[str]:
        """Retorna expertos conocidos."""
        return list(self._expert_ids)

    def _adjust_by_std(
        self, weights: Dict[str, float], std: float
    ) -> Dict[str, float]:
        """Boost a expertos de alta volatilidad si std es elevado."""
        if std < 1.0:
            return weights
        adjusted = dict(weights)
        for eid in ("taylor", "kalman"):
            if eid in adjusted:
                adjusted[eid] = adjusted[eid] * (1.0 + 0.2 * std)
        return adjusted

    def _adjust_by_slope(
        self, weights: Dict[str, float], slope: float
    ) -> Dict[str, float]:
        """Boost a statistical si hay tendencia significativa."""
        if abs(slope) <= self._slope_threshold:
            return weights
        adjusted = dict(weights)
        if "statistical" in adjusted:
            adjusted["statistical"] = adjusted["statistical"] * (1.0 + abs(slope))
        return adjusted

    def _adjust_by_noise(
        self, weights: Dict[str, float], noise_ratio: float
    ) -> Dict[str, float]:
        """Boost a kalman si hay mucho ruido."""
        if noise_ratio < 0.2:
            return weights
        adjusted = dict(weights)
        if "kalman" in adjusted:
            adjusted["kalman"] = adjusted["kalman"] * self._noise_boost_factor
        return adjusted

    @staticmethod
    def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
        """Normaliza pesos a probabilidades que suman 1.0."""
        total = sum(weights.values())
        if total < 1e-9:
            n = len(weights)
            return {k: 1.0 / n for k in weights} if n > 0 else {}
        return {k: v / total for k, v in weights.items()}

    @staticmethod
    def _compute_entropy(probabilities: Dict[str, float]) -> float:
        """Calcula entropía de Shannon."""
        entropy = 0.0
        for p in probabilities.values():
            if p > 1e-9:
                entropy -= p * math.log2(p)
        return entropy
