"""CycleReport dataclass and status formatting for PaperBotRunner."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class CycleReport:
    """Resultado de un ciclo, para status line y tests."""

    cycle: int
    new_candles: int
    predictions_persisted: int
    evidence_persisted: int
    resolved: int
    waiting: int
    actions: Counter = field(default_factory=Counter)
    latest_prob_raw: float | None = None
    latest_prob_calibrated: float | None = None
    idle: bool = False

    def status_line(
        self,
        *,
        uptime_s: float,
        calibrator_version: str | None,
        connected: bool,
        gaps: int,
        errors: int,
    ) -> str:
        mm, ss = divmod(int(uptime_s), 60)
        hh, mm = divmod(mm, 60)
        total = sum(self.actions.values())
        parts = [
            f"up={hh:02d}:{mm:02d}:{ss:02d}",
            f"cyc={self.cycle}",
            f"velas_nuevas={self.new_candles}",
            f"pred={self.predictions_persisted}",
            f"ev={self.evidence_persisted}",
        ]
        if total:
            rates = " ".join(
                f"{action.value}={100 * n / total:.0f}%"
                for action, n in sorted(self.actions.items())
            )
            parts.append(rates)
        if self.latest_prob_calibrated is not None:
            parts.append(f"P={self.latest_prob_calibrated:.3f}")
        else:
            parts.append("P=-")
        parts.append(f"resueltas={self.resolved}")
        parts.append(f"esperando={self.waiting}")
        parts.append(f"cal={calibrator_version or 'UNCALIBRATED'}")
        parts.append("conn=" + ("OK" if connected else "DEGRADED"))
        parts.append(f"gaps={gaps}")
        parts.append(f"err={errors}")
        return " ".join(parts)
