"""FASE 5 — MetaLearner: tipos (outcome realizado + constantes)."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["META_STATE_VERSION", "GAIN_CLIP", "ExpertOutcome"]

META_STATE_VERSION: str = "meta-learner-v1"

#: Recorte simétrico de ganancia por update (fracción de retorno).
GAIN_CLIP: float = 0.05


@dataclass(frozen=True, slots=True, kw_only=True)
class ExpertOutcome:
    """Outcome realizado de un experto (solo datos externos)."""

    expert: str
    regime: str
    horizon_seconds: int
    net_return: float  # realizado − costos (fracción)

    def __post_init__(self) -> None:
        if not self.expert.strip():
            raise ValueError("expert no puede ser vacío")
        if not self.regime.strip():
            raise ValueError("regime no puede ser vacío")
        object.__setattr__(self, "expert", self.expert.strip())
        object.__setattr__(self, "regime", self.regime.strip())
        if self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds debe ser > 0")
        if not math.isfinite(self.net_return):
            raise ValueError(f"net_return no finito: {self.net_return!r}")
