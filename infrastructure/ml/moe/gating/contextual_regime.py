"""ContextualRegimeGating — routing basado en features numéricos del pipeline.

Mejoras:
- Ajustes acotados vía sigmoide (nunca distortionan pesos más allá de [0.5x, 3x])
- Historial de performance por experto para ajuste fino del routing
- Factor de estabilidad temporal: evita cambios bruscos de top_expert
- Utiliza curvature y relative_deviation del FeatureContext
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .base import GatingProbs
from ..feature_context import FeatureContext


def _sigmoid_boost(x: float, midpoint: float = 1.0, steepness: float = 2.0) -> float:
    """Factor de boost acotado [1.0, 3.0) vía sigmoide."""
    return 1.0 + 2.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


def _sigmoid_reduce(x: float, midpoint: float = 1.0, steepness: float = 2.0) -> float:
    """Factor de reducción acotado (0.5, 1.0] vía sigmoide inversa."""
    return 1.0 - 0.5 / (1.0 + math.exp(-steepness * (x - midpoint)))


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


class ExpertPerformanceTracker:
    """Track historial de performance por experto para ajustar routing."""

    def __init__(self, decay: float = 0.3, window: int = 20):
        self._decay = decay
        self._window = window
        self._scores: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def record(self, expert_id: str, error: float) -> None:
        prev = self._scores.get(expert_id, 0.0)
        count = self._counts.get(expert_id, 0)
        if count >= self._window:
            self._scores[expert_id] = prev * (1 - self._decay) + error * self._decay
        else:
            self._scores[expert_id] = (prev * count + error) / (count + 1)
        self._counts[expert_id] = min(count + 1, self._window)

    def get_reliability(self, expert_id: str) -> float:
        """Retorna confiabilidad [0, 1] basada en error promedio histórico."""
        score = self._scores.get(expert_id)
        if score is None:
            return 0.5
        return max(0.1, 1.0 - abs(score))


class ContextualRegimeGating:
    """Gating que utiliza features numéricos del pipeline para routing.

    Attributes:
        _regime_weights: Dict {regime: {expert_id: weight}} externalizado.
        _expert_ids: Lista de expertos conocidos.
        _noise_boost_factor: Factor multiplicador para expertos de noisy.
        _slope_threshold: Umbral para considerar tendencia significativa.
        _performance_tracker: Historial de error por experto.
        _last_top_expert: Último top_expert seleccionado (estabilidad temporal).
    """

    # Pesos por defecto — cargados desde config externa
    DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = _load_regime_weights()

    def __init__(
        self,
        regime_weights: Optional[Dict[str, Dict[str, float]]] = None,
        expert_ids: Optional[List[str]] = None,
        noise_boost_factor: float = 1.5,
        slope_threshold: float = 0.01,
        stability_bonus: float = 0.05,
    ) -> None:
        self._regime_weights = regime_weights or dict(self.DEFAULT_WEIGHTS)
        self._expert_ids = expert_ids or []
        self._noise_boost_factor = noise_boost_factor
        self._slope_threshold = slope_threshold
        self._stability_bonus = stability_bonus
        self._performance_tracker = ExpertPerformanceTracker()
        self._last_top_expert: Optional[str] = None

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

        # Ajuste por std (volatilidad) — sigmoide acotado
        weights = self._adjust_by_std(weights, feature_context.std)

        # Ajuste por slope (tendencia) — sigmoide acotado
        weights = self._adjust_by_slope(weights, feature_context.slope)

        # Ajuste por noise_ratio (ruido) — sigmoide acotado
        weights = self._adjust_by_noise(weights, feature_context.noise_ratio)

        # Ajuste por curvature (aceleración) — nuevo
        weights = self._adjust_by_curvature(weights, feature_context.curvature)

        # Ajuste por reliability histórico — nuevo
        weights = self._adjust_by_performance(weights)

        # Estabilidad temporal: leve bonus al último top_expert
        if self._last_top_expert is not None and self._last_top_expert in weights:
            weights[self._last_top_expert] *= (1.0 + self._stability_bonus)

        # Normalizar a probabilidades
        probabilities = self._normalize(weights)

        # Entropía como medida de incertidumbre
        entropy = self._compute_entropy(probabilities)

        top_expert = max(probabilities.items(), key=lambda x: x[1])[0] if probabilities else ""
        self._last_top_expert = top_expert

        return GatingProbs(
            probabilities=probabilities,
            entropy=entropy,
            top_expert=top_expert,
            metadata={
                "regime": regime,
                "std": feature_context.std,
                "slope": feature_context.slope,
                "noise_ratio": feature_context.noise_ratio,
                "curvature": feature_context.curvature,
                "raw_weights": weights,
                "equipment_class": equipment_class,
                "used_equipment_override": equipment_class in EQUIPMENT_REGIME_WEIGHTS,
                "performance_scores": {
                    eid: round(self._performance_tracker.get_reliability(eid), 3)
                    for eid in self._expert_ids
                },
            },
        )

    def explain(
        self, feature_context: FeatureContext, probs: GatingProbs
    ) -> str:
        """Explica la decisión de routing en lenguaje humano."""
        regime = feature_context.regime
        top = probs.top_expert
        top_prob = probs.max_probability

        reasons = [f"régimen={regime}"]

        if feature_context.std > 1.0:
            reasons.append(f"std={feature_context.std:.2f}")
        if abs(feature_context.slope) > self._slope_threshold:
            reasons.append(f"slope={feature_context.slope:.4f}")
        if feature_context.noise_ratio > 0.3:
            reasons.append(f"noise={feature_context.noise_ratio:.2f}")
        if abs(feature_context.curvature) > 0.001:
            reasons.append(f"curvature={feature_context.curvature:.4f}")

        perf_scores = probs.metadata.get("performance_scores", {})
        if perf_scores:
            top_perf = max(perf_scores.items(), key=lambda x: x[1])
            reasons.append(f"best_history={top_perf[0]}({top_perf[1]:.2f})")

        return (
            f"ContextualRegimeGating: top_expert={top}({top_prob:.2f}) "
            f"entropy={probs.entropy:.3f} | {' | '.join(reasons)}"
        )

    def get_expert_ids(self) -> List[str]:
        """Retorna expertos conocidos."""
        return list(self._expert_ids)

    def _adjust_by_std(
        self, weights: Dict[str, float], std: float
    ) -> Dict[str, float]:
        """Boost acotado [1x, 3x) a expertos de alta volatilidad vía sigmoide."""
        if std < 0.5:
            return weights
        boost = _sigmoid_boost(std, midpoint=1.5, steepness=1.5)
        adjusted = dict(weights)
        for eid in ("taylor", "kalman"):
            if eid in adjusted:
                adjusted[eid] = adjusted[eid] * boost
        # Penalizar baseline en alta volatilidad
        if std > 1.5 and "baseline" in adjusted:
            adjusted["baseline"] *= _sigmoid_reduce(std, midpoint=2.0, steepness=1.0)
        return adjusted

    def _adjust_by_slope(
        self, weights: Dict[str, float], slope: float
    ) -> Dict[str, float]:
        """Boost acotado [1x, 3x) a statistical si hay tendencia, vía sigmoide."""
        if abs(slope) <= self._slope_threshold:
            return weights
        boost = _sigmoid_boost(abs(slope), midpoint=0.05, steepness=30.0)
        adjusted = dict(weights)
        if "statistical" in adjusted:
            adjusted["statistical"] = adjusted["statistical"] * boost
        if "taylor" in adjusted:
            adjusted["taylor"] = adjusted["taylor"] * (1.0 + (boost - 1.0) * 0.5)
        return adjusted

    def _adjust_by_noise(
        self, weights: Dict[str, float], noise_ratio: float
    ) -> Dict[str, float]:
        """Boost acotado [1x, 3x) a kalman si hay ruido, vía sigmoide."""
        if noise_ratio < 0.15:
            return weights
        boost = _sigmoid_boost(noise_ratio, midpoint=0.3, steepness=5.0)
        adjusted = dict(weights)
        if "kalman" in adjusted:
            adjusted["kalman"] = adjusted["kalman"] * boost
        # Penalizar baseline con ruido alto
        if noise_ratio > 0.4 and "baseline" in adjusted:
            adjusted["baseline"] *= _sigmoid_reduce(noise_ratio, midpoint=0.5, steepness=3.0)
        return adjusted

    def _adjust_by_curvature(
        self, weights: Dict[str, float], curvature: float
    ) -> Dict[str, float]:
        """Ajuste por curvatura (aceleración). Alta curvatura -> Taylor."""
        if abs(curvature) <= 0.001:
            return weights
        boost = _sigmoid_boost(abs(curvature), midpoint=0.01, steepness=50.0)
        adjusted = dict(weights)
        if "taylor" in adjusted:
            adjusted["taylor"] = adjusted["taylor"] * boost
        return adjusted

    def _adjust_by_performance(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Ajusta pesos según reliability histórico de cada experto."""
        adjusted = dict(weights)
        for eid in adjusted:
            rel = self._performance_tracker.get_reliability(eid)
            # Si reliability es baja (<0.3), penalizar; si alta (>0.7), boost suave
            if rel < 0.3:
                adjusted[eid] *= 0.5
            elif rel > 0.7:
                adjusted[eid] *= (1.0 + (rel - 0.7) * 0.3)
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
