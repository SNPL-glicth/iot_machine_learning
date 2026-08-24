"""Weight proposal dataclass (FASE 8)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeightProposal:
    """Propuesta de cambio de peso para un contexto (inmutable)."""

    expert: str
    regime: str | None
    horizon_seconds: int
    current_weight: float
    proposed_weight: float
    observed_reward: float
    calibration: float
    sample_size: int
    accuracy: float
    reason: str
    created_at: float
    parent_version: str | None = None

    @property
    def context_label(self) -> str:
        return f"{self.expert}|{self.regime or '-'}|{self.horizon_seconds}s"

    @property
    def weight_delta(self) -> float:
        return self.proposed_weight - self.current_weight