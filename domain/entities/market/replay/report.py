"""PerformanceReport del Market Replay (FASE 5).

Agregado **inmutable** de un solo pasada sobre las predicciones ya
materializadas de un run (``Prediction`` con outcome/evaluation/reward).
Responde el marcador del sistema con honestidad estadística:

    * predicciones emitidas / evaluadas / invalidadas (por horizon);
    * direction rate: fracción de aciertos de dirección;
    * calibration: |probabilidad promedio - tasa real| (0 = perfecta);
    * avg |return error|: error medio de magnitud;
    * reward acumulado (la señal que decidirá MoE más adelante).

``PerformanceReport`` no entrena ni pondera nada: es solo el tablero.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..prediction.prediction import Prediction


@dataclass(frozen=True, slots=True)
class HorizonStat:
    """Estadísticas agregadas para un (horizonte, estrategia)."""

    horizon_seconds: int
    strategy: str | None
    predictions: int = 0
    evaluated: int = 0
    invalidated: int = 0
    direction_rate: float = 0.0
    calibration: float = 0.0
    avg_return_error: float = 0.0
    reward: float = 0.0

    def merge(self, other: HorizonStat) -> HorizonStat:
        """Combina dos stats del mismo (horizonte, estrategia)."""
        if (
            self.horizon_seconds != other.horizon_seconds
            or self.strategy != other.strategy
        ):
            raise ValueError("no se pueden mergear stats de distinta clave")
        total = self.evaluated + other.evaluated
        if total == 0:
            return dataclasses.replace(self)
        w1 = self.evaluated / total
        w2 = other.evaluated / total
        return HorizonStat(
            horizon_seconds=self.horizon_seconds,
            strategy=self.strategy,
            predictions=self.predictions + other.predictions,
            evaluated=total,
            invalidated=self.invalidated + other.invalidated,
            direction_rate=self.direction_rate * w1 + other.direction_rate * w2,
            calibration=self.calibration * w1 + other.calibration * w2,
            avg_return_error=(
                self.avg_return_error * w1 + other.avg_return_error * w2
            ),
            reward=self.reward + other.reward,
        )


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """Tablero del run (por horizonte, con desglose por estrategia)."""

    symbol: str
    resolution_seconds: int
    stats: tuple[HorizonStat, ...] = ()
    total_predictions: int = 0
    total_evaluated: int = 0
    total_invalidated: int = 0

    @classmethod
    def from_run(cls, symbol: str, resolution_seconds: int,
                 predictions: tuple[Prediction, ...]) -> PerformanceReport:
        """Agrega un run completo (una sola pasada, orden determinista)."""
        stats: dict[tuple[int, str | None], HorizonStat] = {}
        for pred in predictions:
            key = (pred.horizon_seconds, pred.strategy)
            stat = stats.get(key)
            if stat is None:
                stat = HorizonStat(
                    horizon_seconds=pred.horizon_seconds,
                    strategy=pred.strategy,
                )
            stat = dataclasses.replace(
                stat, predictions=stat.predictions + 1
            )
            if pred.outcome is not None and pred.evaluation is not None:
                direction = 1.0 if pred.evaluation.direction_correct else 0.0
                n = stat.evaluated
                new_evaluated = n + 1
                stat = dataclasses.replace(
                    stat,
                    evaluated=new_evaluated,
                    direction_rate=(
                        (stat.direction_rate * n + direction) / new_evaluated
                    ),
                    calibration=(
                        (stat.calibration * n + pred.evaluation.calibration_error)
                        / new_evaluated
                    ),
                    avg_return_error=(
                        (stat.avg_return_error * n + pred.evaluation.magnitude_error)
                        / new_evaluated
                    ),
                    reward=stat.reward + (pred.reward.total if pred.reward else 0.0),
                )
            else:
                stat = dataclasses.replace(stat, invalidated=stat.invalidated + 1)
            stats[key] = stat

        rows = tuple(sorted(stats.values(), key=lambda s: (s.horizon_seconds, str(s.strategy))))
        total_pred = sum(s.predictions for s in rows)
        total_ev = sum(s.evaluated for s in rows)
        total_inv = sum(s.invalidated for s in rows)
        return cls(
            symbol=symbol,
            resolution_seconds=resolution_seconds,
            stats=rows,
            total_predictions=total_pred,
            total_evaluated=total_ev,
            total_invalidated=total_inv,
        )

    def by_horizon(self) -> list[tuple[str, HorizonStat]]:
        """Stats agregados por horizonte (mezclando estrategias)."""
        merged: dict[int, HorizonStat] = {}
        for stat in self.stats:
            key = stat.horizon_seconds
            if key in merged:
                merged[key] = merged[key].merge(stat)
            else:
                merged[key] = stat
        return [(self._fmt_horizon(h), merged[h]) for h in sorted(merged)]

    def _fmt_horizon(self, seconds: int) -> str:
        for label, span in (("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600), ("1d", 86400)):
            if seconds == span:
                return label
        if seconds % 86400 == 0:
            return f"{seconds // 86400}d"
        if seconds % 3600 == 0:
            return f"{seconds // 3600}h"
        if seconds % 60 == 0:
            return f"{seconds // 60}m"
        return f"{seconds}s"

    def render_ascii(self) -> str:
        """Renderiza el marcador ``ZENIN MARKET RUN`` (texto plano)."""
        lines: list[str] = []
        lines.append("ZENIN MARKET RUN")
        lines.append("──────────────────────────────")
        lines.append("")
        lines.append(f"Instrument: {self.symbol}")
        lines.append(f"Resolution: {self._fmt_horizon(self.resolution_seconds)}")
        lines.append("")
        lines.append(f"Predictions:      {self.total_predictions:,}")
        lines.append(f"Evaluated:        {self.total_evaluated:,}")
        lines.append(f"Invalidated:      {self.total_invalidated:,}")
        lines.append("")
        lines.append("──────────────────────────────")
        lines.append("")
        lines.append("HORIZON")
        lines.append("")
        for label, stat in self.by_horizon():
            lines.append(label)
            lines.append(
                f"  Direction:        {stat.direction_rate * 100:.1f}%"
            )
            lines.append(f"  Calibration:      {stat.calibration:.2f}")
            lines.append(
                f"  Avg return error: {stat.avg_return_error * 100:.2f}%"
            )
            lines.append(f"  Reward:           {stat.reward:+,.0f}")
            lines.append("")
        if any(s.strategy is not None for s in self.stats):
            lines.append("BY STRATEGY")
            lines.append("")
            current_horizon: int | None = None
            for stat in self.stats:
                if stat.horizon_seconds != current_horizon:
                    current_horizon = stat.horizon_seconds
                    lines.append(self._fmt_horizon(current_horizon))
                lines.append(
                    f"  {stat.strategy or '(default)':<12} "
                    f"reward {stat.reward:+.2f} "
                    f"dir {stat.direction_rate * 100:.1f}%"
                )
            lines.append("")
        return "\n".join(lines)
