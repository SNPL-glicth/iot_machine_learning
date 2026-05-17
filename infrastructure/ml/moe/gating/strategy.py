"""GatingStrategy — Protocol para estrategias de routing MoE.

Patrón Strategy: permite intercambiar algoritmos de gating sin modificar
el motor de predicción MoE.
"""

from __future__ import annotations

from typing import Protocol, Dict, List, runtime_checkable

from .base import GatingProbs
from ..feature_context import FeatureContext


@runtime_checkable
class GatingStrategy(Protocol):
    """Contrato para estrategias de routing de expertos.

    Todo gating strategy debe:
    1. Recibir FeatureContext completo (no solo regime string).
    2. Retornar GatingProbs con distribución normalizada.
    3. Ser determinista y explicable vía explain().
    """

    def route(self, feature_context: FeatureContext) -> GatingProbs:
        """Decide distribución de probabilidades sobre expertos.

        Args:
            feature_context: Contexto enriquecido del pipeline.

        Returns:
            GatingProbs con distribución sobre expertos.
        """
        ...

    def explain(
        self, feature_context: FeatureContext, probs: GatingProbs
    ) -> str:
        """Explica la decisión de routing en lenguaje humano.

        Args:
            feature_context: Contexto usado para routing.
            probs: Probabilidades resultantes.

        Returns:
            Explicación textual de la decisión.
        """
        ...

    def get_expert_ids(self) -> List[str]:
        """Retorna los expertos conocidos por este gating."""
        ...
