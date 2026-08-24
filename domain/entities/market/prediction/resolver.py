"""OutcomeResolver (FASE 7) — espera el horizonte y resuelve el desenlace.

Fase intermedia del loop live persistido:

    Prediction ──> [espera horizonte] ──> Outcome ──> Evaluation ──> Reward

El resolver recibe predicciones pendientes (PENDING/ACTIVE/WAITING_OUTCOME)
y una fuente de precios (``PriceLookup``); cuando el reloj de los precios
alcanza el vencimiento del horizonte, construye el ``Outcome`` contra el
último cierre conocido y recorre el contrato del ciclo de vida
(``activate -> to_waiting_outcome -> evaluate -> issue_reward``).

Regla de FASE 7: este componente NO aprende. Solo materializa el
desenlace para que el store lo registre; la adaptación (calibración,
expertos, MoE) queda fuera de esta fase.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from .lifecycle import PredictionStatus
from .outcome import Outcome
from .prediction import Prediction
from .reward import RewardConfig

__all__ = ["PriceLookup", "ResolvedBatch", "OutcomeResolver"]


class PriceLookup(Protocol):
    """Último cierre disponible a lo sumo en un instante (sin mirar futuro)."""

    def last_close(self, at_or_before: float) -> float | None:
        """Retorna el último cierre con timestamp <= ``at_or_before``.

        ``None`` si no hay datos en o antes del plazo (horizonte aún no
        cubierto por el feed): la predicción sigue en espera.
        """
        ...


@dataclass(frozen=True, slots=True)
class ResolvedBatch:
    """Resultado de una pasada del resolver (inmutable)."""

    resolved: tuple[Prediction, ...]
    still_waiting: tuple[Prediction, ...]
    unchanged: tuple[Prediction, ...]

    @property
    def resolved_count(self) -> int:
        return len(self.resolved)

    @property
    def waiting_count(self) -> int:
        return len(self.still_waiting)


class OutcomeResolver:
    """Resuelve predicciones vencidas contra el último cierre conocido."""

    def __init__(self, reward_config: RewardConfig | None = None) -> None:
        self.reward_config = reward_config or RewardConfig()

    def resolve(
        self,
        predictions: Iterable[Prediction],
        prices: PriceLookup,
    ) -> ResolvedBatch:
        """Resuelve cada predicción cuyo horizonte ya venció.

        Predicciones ya terminales (REWARDED/INVALIDATED/ARCHIVED) se
        devuelven intactas en ``unchanged``; las que aún no tienen datos
        para su vencimiento quedan en ``still_waiting``.
        """
        resolved: list[Prediction] = []
        still_waiting: list[Prediction] = []
        unchanged: list[Prediction] = []

        for pred in predictions:
            if not isinstance(pred, Prediction):
                raise TypeError(
                    f"se espera Prediction, obtenido {type(pred).__name__}"
                )
            if pred.status not in (
                PredictionStatus.PENDING,
                PredictionStatus.ACTIVE,
                PredictionStatus.WAITING_OUTCOME,
            ):
                unchanged.append(pred)
                continue

            deadline = pred.observation.timestamp + pred.horizon_seconds
            final_price = prices.last_close(at_or_before=deadline)
            if final_price is None:
                still_waiting.append(pred)
                continue

            outcome = Outcome.from_prices(
                symbol=pred.observation.symbol,
                ref_timestamp=pred.observation.timestamp,
                ref_price=pred.entry_price,
                horizon_seconds=pred.horizon_seconds,
                final_price=final_price,
                measured_at=deadline,
            )
            resolved.append(self._materialize(pred, outcome))

        return ResolvedBatch(
            resolved=tuple(resolved),
            still_waiting=tuple(still_waiting),
            unchanged=tuple(unchanged),
        )

    def _materialize(self, pred: Prediction, outcome: Outcome) -> Prediction:
        """Recorre el contrato del ciclo de vida hasta REWARDED."""
        if pred.status is PredictionStatus.PENDING:
            pred = pred.activate()
        if pred.status is PredictionStatus.ACTIVE:
            pred = pred.to_waiting_outcome(outcome)
        if pred.status is PredictionStatus.WAITING_OUTCOME:
            pred = pred.evaluate(outcome)
        if pred.status is PredictionStatus.EVALUATED:
            pred = pred.issue_reward(self.reward_config)
        return pred
