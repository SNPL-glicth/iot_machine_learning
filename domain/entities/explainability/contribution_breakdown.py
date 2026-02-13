"""Desglose de contribución por engine al resultado final.

Cada engine que participa en la fusión produce un
``EngineContribution`` que documenta su predicción individual,
su peso, su estado de inhibición y por qué fue seleccionado
o suprimido.

``ContributionBreakdown`` agrega todas las contribuciones y
expone métricas de consenso.

Domain-pure.  Sin dependencias de infraestructura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class EngineContribution:
    """Contribución de un engine individual a la predicción fusionada.

    Attributes:
        engine_name: Identificador del engine.
        predicted_value: Predicción cruda del engine.
        confidence: Confianza auto-reportada [0, 1].
        trend: Tendencia detectada.
        base_weight: Peso antes de inhibición.
        final_weight: Peso después de inhibición (normalizado).
        inhibited: True si el engine fue suprimido.
        inhibition_reason: Razón de supresión (``"none"`` si no).
        local_fit_error: Error de ajuste local del modelo.
        stability: Indicador de estabilidad (0 = estable).
        metadata: Datos adicionales del engine.
    """

    engine_name: str
    predicted_value: float
    confidence: float = 0.0
    trend: str = "stable"
    base_weight: float = 0.0
    final_weight: float = 0.0
    inhibited: bool = False
    inhibition_reason: str = "none"
    local_fit_error: float = 0.0
    stability: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def weighted_contribution(self) -> float:
        """Contribución ponderada al valor fusionado."""
        return self.predicted_value * self.final_weight

    def to_dict(self) -> dict:
        return {
            "engine_name": self.engine_name,
            "predicted_value": round(self.predicted_value, 6),
            "confidence": round(self.confidence, 4),
            "trend": self.trend,
            "base_weight": round(self.base_weight, 4),
            "final_weight": round(self.final_weight, 4),
            "inhibited": self.inhibited,
            "inhibition_reason": self.inhibition_reason,
            "local_fit_error": round(self.local_fit_error, 6),
            "stability": round(self.stability, 4),
        }


@dataclass(frozen=True)
class ContributionBreakdown:
    """Desglose completo de contribuciones de todos los engines.

    Attributes:
        contributions: Lista de contribuciones individuales.
        fusion_method: Método de fusión usado.
        selected_engine: Engine con mayor peso final.
        selection_reason: Razón estructurada de la selección.
        fallback_used: True si se usó fallback.
        fallback_reason: Razón del fallback (si aplica).
    """

    contributions: List[EngineContribution] = field(default_factory=list)
    fusion_method: str = "weighted_average"
    selected_engine: str = "none"
    selection_reason: str = ""
    fallback_used: bool = False
    fallback_reason: Optional[str] = None

    @property
    def n_engines(self) -> int:
        """Número de engines que participaron."""
        return len(self.contributions)

    @property
    def n_inhibited(self) -> int:
        """Número de engines inhibidos."""
        return sum(1 for c in self.contributions if c.inhibited)

    @property
    def consensus_spread(self) -> float:
        """Dispersión de predicciones entre engines.

        0.0 = consenso perfecto.  Valor alto = desacuerdo.
        Calculado como max - min de predicted_value.
        """
        if len(self.contributions) < 2:
            return 0.0
        vals = [c.predicted_value for c in self.contributions]
        return max(vals) - min(vals)

    @property
    def dominant_weight_ratio(self) -> float:
        """Ratio del peso del engine dominante vs el total.

        1.0 = un solo engine domina.  1/N = distribución uniforme.
        """
        if not self.contributions:
            return 0.0
        max_w = max(c.final_weight for c in self.contributions)
        total = sum(c.final_weight for c in self.contributions)
        if total < 1e-12:
            return 0.0
        return max_w / total

    def to_dict(self) -> dict:
        return {
            "contributions": [c.to_dict() for c in self.contributions],
            "fusion_method": self.fusion_method,
            "selected_engine": self.selected_engine,
            "selection_reason": self.selection_reason,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "n_engines": self.n_engines,
            "n_inhibited": self.n_inhibited,
            "consensus_spread": round(self.consensus_spread, 6),
            "dominant_weight_ratio": round(self.dominant_weight_ratio, 4),
        }

    @classmethod
    def empty(cls) -> ContributionBreakdown:
        return cls()
