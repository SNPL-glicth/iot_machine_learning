"""Máquina de estados para control de histéresis y supresión de flapping.

Gobernanza de transiciones entre resoluciones temporales (10X, 2X, RAW).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from iot_machine_learning.domain.entities.representation_evidence import (
    PolicyDecision,
    RepresentationLevel,
    SystemOperationalState,
)


class PolicyResolutionState(str, Enum):
    COMPRESSED = "10x"
    BALANCED = "2x"
    HIGH_RESOLUTION = "raw"


class PolicyOperationalMode(str, Enum):
    COMPRESSED = "COMPRESSED"
    ESCALATING = "ESCALATING"
    HIGH_RESOLUTION = "HIGH_RESOLUTION"
    COOLDOWN = "COOLDOWN"
    DE_ESCALATING = "DE_ESCALATING"


@dataclass
class PolicyDecisionReport:
    resolution: PolicyResolutionState
    subsample_step: int
    operational_mode: PolicyOperationalMode
    action: str  # "ESCALATE", "HOLD", "DE-ESCALATE", "BACKFILL"
    shock_triggered: bool
    regime_triggered: bool
    shock_surprise: float
    regime_surprise: float
    requires_backfill: bool
    reason: str

    def to_domain_decision(self) -> PolicyDecision:
        """Convierte este reporte a la entidad inmutable del dominio."""
        level_map = {
            PolicyResolutionState.COMPRESSED: RepresentationLevel.TEN_X,
            PolicyResolutionState.BALANCED: RepresentationLevel.TWO_X,
            PolicyResolutionState.HIGH_RESOLUTION: RepresentationLevel.RAW,
        }
        if self.shock_triggered:
            state = SystemOperationalState.SHOCKED
        elif self.regime_triggered:
            state = SystemOperationalState.DRIFTING
        else:
            state = SystemOperationalState.RESTING

        return PolicyDecision(
            level=level_map[self.resolution],
            operational_state=state,
            reason=self.reason,
            backfill_count=1 if self.requires_backfill else 0,
            flapping_suppressed=(self.operational_mode == PolicyOperationalMode.COOLDOWN),
        )


class AgnosticPolicyStateMachine:
    """Máquina de estados para control de histéresis y supresión de flapping."""

    def __init__(
        self,
        enable_hysteresis: bool = True,
        enable_backfill: bool = True,
        enable_representation_safety: bool = True,
        cooldown_blocks: int = 3,
    ) -> None:
        self.enable_hysteresis = enable_hysteresis
        self.enable_backfill = enable_backfill
        self.enable_safety = enable_representation_safety
        self.cooldown_blocks = cooldown_blocks

        self.current_state = PolicyResolutionState.COMPRESSED
        self.current_mode = PolicyOperationalMode.COMPRESSED
        self.active_hold_until_block: int = -1

        # Métricas de telemetría de la máquina
        self.switch_count: int = 0
        self.escalation_count: int = 0
        self.de_escalation_count: int = 0
        self.hold_count: int = 0
        self.backfill_count: int = 0

    def evaluate_block(
        self,
        block_idx: int,
        shock_triggered: bool,
        regime_triggered: bool,
        shock_surprise: float,
        regime_surprise: float,
    ) -> PolicyDecisionReport:
        prev_res = self.current_state
        req_backfill = False

        if shock_triggered:
            # Choque violento: salto a RAW
            target_res = PolicyResolutionState.HIGH_RESOLUTION
            target_mode = PolicyOperationalMode.ESCALATING
            action = "ESCALATE"
            reason = "shock_quantile_exceeded"
            if self.enable_backfill and self.current_state == PolicyResolutionState.COMPRESSED:
                req_backfill = True
            if self.enable_hysteresis:
                self.active_hold_until_block = block_idx + self.cooldown_blocks

        elif regime_triggered:
            # Deriva de régimen: salto al menos a 2X (o RAW si safety lo exige)
            if self.enable_safety and regime_surprise > 1.5:
                target_res = PolicyResolutionState.HIGH_RESOLUTION
            else:
                target_res = PolicyResolutionState.BALANCED
            target_mode = PolicyOperationalMode.ESCALATING
            action = "ESCALATE"
            reason = "regime_envelope_exceeded"
            if self.enable_backfill and self.current_state == PolicyResolutionState.COMPRESSED:
                req_backfill = True
            if self.enable_hysteresis:
                self.active_hold_until_block = max(
                    self.active_hold_until_block, block_idx + self.cooldown_blocks
                )

        else:
            # Sin disparos activos: ¿sigue en cooldown/hold?
            if self.enable_hysteresis and block_idx <= self.active_hold_until_block:
                target_res = self.current_state
                target_mode = PolicyOperationalMode.COOLDOWN
                action = "HOLD"
                reason = "cooldown_hysteresis_active"
                self.hold_count += 1
            else:
                # Retorno a reposo comprimido
                target_res = PolicyResolutionState.COMPRESSED
                target_mode = PolicyOperationalMode.COMPRESSED
                action = (
                    "DE-ESCALATE"
                    if self.current_state != PolicyResolutionState.COMPRESSED
                    else "HOLD"
                )
                reason = "nominal_distribution_consistent"

        # Registrar transiciones
        if target_res != prev_res:
            self.switch_count += 1
            if (
                target_res == PolicyResolutionState.HIGH_RESOLUTION
                or (
                    target_res == PolicyResolutionState.BALANCED
                    and prev_res == PolicyResolutionState.COMPRESSED
                )
            ):
                self.escalation_count += 1
            else:
                self.de_escalation_count += 1

        if req_backfill:
            self.backfill_count += 1

        self.current_state = target_res
        self.current_mode = target_mode

        step = (
            1
            if target_res == PolicyResolutionState.HIGH_RESOLUTION
            else (2 if target_res == PolicyResolutionState.BALANCED else 10)
        )

        return PolicyDecisionReport(
            resolution=target_res,
            subsample_step=step,
            operational_mode=target_mode,
            action=action,
            shock_triggered=shock_triggered,
            regime_triggered=regime_triggered,
            shock_surprise=shock_surprise,
            regime_surprise=regime_surprise,
            requires_backfill=req_backfill,
            reason=reason,
        )
