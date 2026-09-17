"""Execution Plan Translator for ZENIN Master Equation.

Converts mathematical ExecutionPlan from MasterEquationOrchestrator into
canonical OrderDirective instances ready for risk verification and broker execution.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OrderDirective:
    """Canonical trading directive derived from the Master Equation."""

    action: str  # "EXECUTE", "HOLD", "EMERGENCY_FLUSH"
    trigger_code: int  # 1 (buy), -1 (sell), 0 (hold), -99 (flush)
    side: str  # "buy", "sell", "hold", "flush"
    confidence: float  # Phi_MoE or Certeza score
    magnitude: float  # Target magnitude / sizing factor
    plan: Any  # Source ExecutionPlan
    veto_reason: str = ""
    regime_alert: bool = False

    @property
    def is_executable(self) -> bool:
        return self.trigger_code in (1, -1) and self.action.upper() == "EXECUTE"

    @property
    def is_flush(self) -> bool:
        return self.trigger_code == -99 or self.action.upper() in ("EMERGENCY_FLUSH", "CLOSE")

    @property
    def is_hold(self) -> bool:
        return self.trigger_code == 0 or self.action.upper() == "HOLD"


def resolve_plan_action_trigger(plan: Any) -> tuple[int, str]:
    """Traduce el ExecutionPlan de la Ecuación Maestra al gatillo:

    >0 COMPRA, <0 VENTA, 0 HOLD, -99 FLUSH.
    """
    raw = getattr(plan, "action", 0)
    if raw in ("EMERGENCY_FLUSH", "CLOSE") or getattr(plan, "regime_alert", False):
        return -99, "flush"
    if isinstance(raw, (int, float)):
        return (1, "buy") if raw > 0 else ((-1, "sell") if raw < 0 else (0, "hold"))
    if str(raw).upper() == "EXECUTE":
        side = getattr(plan, "side", "")
        if not side and getattr(plan, "chosen_trajectory", None):
            traj = plan.chosen_trajectory
            side = getattr(traj, "side", "")
            if not side and getattr(traj, "terminal_state", None):
                vec = getattr(traj.terminal_state, "state_vector", None)
                if vec is not None and len(vec) > 0:
                    side = "buy" if float(vec[0]) >= 0 else "sell"
        return (1, "buy") if str(side).lower() in ("buy", "long") else (-1, "sell")
    return 0, "hold"


class PlanTranslator:
    """Institutional translator from ZENIN execution plans to trading directives."""

    @classmethod
    def translate(cls, plan: Any) -> OrderDirective:
        """Translates an ExecutionPlan into a structured OrderDirective."""
        trigger_code, trigger_side = resolve_plan_action_trigger(plan)
        confidence = float(getattr(plan, "global_confidence", 0.0))

        # Extract magnitude from envelope if available
        magnitude = 0.0
        envelope = getattr(plan, "envelope", None)
        if envelope is not None:
            magnitude = float(getattr(envelope, "magnitude", 0.0))

        # Extract veto reason if any
        veto_reason = ""
        veto_details = getattr(plan, "veto_details", None)
        if isinstance(veto_details, dict):
            veto_reason = str(veto_details.get("reason", ""))
        elif isinstance(veto_details, str):
            veto_reason = veto_details

        regime_alert = bool(getattr(plan, "regime_alert", False))

        # Normalize action string
        if trigger_code == -99:
            action_str = "EMERGENCY_FLUSH"
        elif trigger_code in (1, -1):
            action_str = "EXECUTE"
        else:
            action_str = "HOLD"

        return OrderDirective(
            action=action_str,
            trigger_code=trigger_code,
            side=trigger_side,
            confidence=confidence,
            magnitude=magnitude,
            plan=plan,
            veto_reason=veto_reason,
            regime_alert=regime_alert,
        )

    @classmethod
    def prepare_executable_plan(cls, plan: Any, trigger_side: str) -> Any:
        """Ensures the ExecutionPlan has action='EXECUTE' and side configured."""
        normalized_action = "EXECUTE"
        if dataclasses.is_dataclass(plan) and not isinstance(plan, type):
            try:
                return dataclasses.replace(plan, action=normalized_action, side=trigger_side)
            except Exception:
                pass
        try:
            plan.action = normalized_action
            plan.side = trigger_side
        except Exception:
            pass
        return plan
