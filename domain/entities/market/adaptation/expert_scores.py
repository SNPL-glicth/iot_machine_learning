"""Performance Analyzer (FASE 8) — qué tan confiable es cada experto.

Regla de FASE 8 (piedra): ZENIN solo aprende de su propia predicción
DESPUÉS de observar el outcome externo. Este módulo consume EXCLUSIVAMENTE
filas evaluadas del store (status=rewarded, sin INVALIDATED ni STALE) y
produce el score por contexto (experto, régimen, horizonte) que alimenta
el WeightProposer.

El score es deliberadamente simple y defendible:

    reward_adjusted = mean_reward * (1 - calibración_penalizada)

- ``mean_reward``: recompensa promedio observada (outcome real, no el
  "creo que acerté");
- la calibración (|P(up) declarada - acierto|) penaliza el reward: un
  experto con la misma recompensa pero confianza rota vale menos.

La precisión (accuracy) NO entra al score: entra al guardrail estadístico
(Wilson lower bound, en ``guard.py``). No optimizamos accuracy; optimizamos
recompensa ajustada por calibración, auditable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = ["ExpertScore", "PerformanceAnalyzer"]


@dataclass(frozen=True, slots=True)
class ExpertScore:
    """Score de un experto en un contexto (experto, régimen, horizonte)."""

    expert: str
    regime: str | None
    horizon_seconds: int
    n: int
    accuracy: float
    mean_reward: float
    reward_total: float
    calibration_error: float
    reward_adjusted: float
    history_days: int = 1
    # FASE 9.2: edge después de costos (agregados del store).
    expected_return: float = 0.0
    realized_return: float = 0.0
    execution_costs: float = 0.0
    # FASE 9.4: desviación del PnL direccional (±|move|) por contexto.
    risk_std: float = 0.0

    @property
    def context(self) -> tuple[str, str | None, int]:
        return (self.expert, self.regime, self.horizon_seconds)

    @property
    def context_label(self) -> str:
        return f"{self.expert}|{self.regime or '-'}|{self.horizon_seconds}s"


class PerformanceAnalyzer:
    """Convierte filas agregadas del store en ExpertScores (puro)."""

    def __init__(self, calibration_penalty: float = 0.5) -> None:
        if not 0.0 <= calibration_penalty <= 1.0:
            raise ValueError(f"calibration_penalty fuera de [0, 1]: {calibration_penalty}")
        self.calibration_penalty = calibration_penalty

    def analyze(self, rows: Iterable[dict[str, Any]]) -> tuple[ExpertScore, ...]:
        """Rows: agregados por (strategy, regime, horizon_seconds) del store.

        Cada fila requiere: strategy, regime, horizon_seconds, evaluated,
        hits, reward, calibration (mean |declared - realized|).
        """
        scores: list[ExpertScore] = []
        for row in rows:
            n = int(row.get("evaluated") or 0)
            hits = int(row.get("hits") or 0)
            if n < 0 or hits < 0 or hits > n:
                raise ValueError(
                    f"fila inválida: evaluated={n} hits={hits} para {row.get('strategy')!r}"
                )
            reward_total = float(row.get("reward") or 0.0)
            calibration = float(row.get("calibration") or 0.0)
            if not 0.0 <= calibration <= 1.0:
                raise ValueError(
                    f"calibration fuera de [0, 1]: {calibration} para {row.get('strategy')!r}"
                )
            mean_reward = reward_total / n if n else 0.0
            accuracy = hits / n if n else 0.0
            scores.append(
                ExpertScore(
                    expert=str(row["strategy"]),
                    regime=(str(row["regime"]) if row.get("regime") is not None else None),
                    horizon_seconds=int(row["horizon_seconds"]),
                    n=n,
                    accuracy=accuracy,
                    mean_reward=mean_reward,
                    reward_total=reward_total,
                    calibration_error=calibration,
                    reward_adjusted=mean_reward
                    * (1.0 - self.calibration_penalty * min(calibration, 1.0)),
                    history_days=max(1, int(row.get("days") or 1)),
                    expected_return=float(row.get("expected_return") or 0.0),
                    realized_return=float(row.get("realized_return") or 0.0),
                    execution_costs=float(row.get("execution_costs") or 0.0),
                    risk_std=float(row.get("risk_std") or 0.0),
                )
            )
        return tuple(scores)
