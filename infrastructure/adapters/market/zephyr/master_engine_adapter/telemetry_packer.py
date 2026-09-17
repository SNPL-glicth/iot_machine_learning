"""Telemetry Packing and Decision Trace Extraction for ZENIN Master Engine.

Formats internal engine representations, stochastic risk bounds, and fractal
rhythm metrics for streaming to Weaviate and the React Dashboard.
"""

from __future__ import annotations

from typing import Any


def extract_decision_trace_metrics(plan: Any, state: Any) -> tuple[str, dict[str, Any]]:
    """Extracts unified metrics from ZENIN Master Equation v2.2.

    Args:
        plan: The ExecutionPlan produced by the Master Engine.
        state: The LiveBotState to update with extracted metrics.

    Returns:
        tuple[str, dict[str, Any]]: Diagnostic summary string and metrics score map.
    """
    dt: dict[str, Any] = {}
    if getattr(plan, "envelope", None) and plan.envelope.metadata and "decision_trace" in plan.envelope.metadata:
        dt = plan.envelope.metadata["decision_trace"]
    elif getattr(plan, "veto_details", None) and isinstance(plan.veto_details, dict) and "decision_trace" in plan.veto_details:
        dt = plan.veto_details["decision_trace"]

    scores: dict[str, Any] = {}
    shadow_str = ""
    if dt:
        if hasattr(state, "last_lambda_t"):
            state.last_lambda_t = float(dt.get("lambda_t", 0.0))
        if hasattr(state, "last_phi_ritmo"):
            state.last_phi_ritmo = float(dt.get("phi_ritmo", state.last_phi_ritmo))

        scores = dict(dt.get("expert_confidences", {}))
        if hasattr(state, "last_expert_votes"):
            state.last_expert_votes = [
                {
                    "name": n,
                    "vote": round(float(s) * 2.0 - 1.0, 2),
                    "weight": 1.0,
                    "confidence": round(float(s), 2),
                }
                for n, s in scores.items()
            ]

        r_sh = dt.get("risk_engine_shadow", {})
        t_sh = dt.get("temporal_engine_shadow", {})
        if r_sh or t_sh:
            r_veto = r_sh.get("veto_riesgo", 1)
            cvar = r_sh.get("cvar_t", 0.0)
            crono = t_sh.get("lambda_crono", 0.0)
            certeza = dt.get("certeza", dt.get("phi_redrose", 0.0))
            mom = dt.get("momentum_veto", 1.0)
            gov = dt.get("governing_component", "phi_moe_base")
            scores.update({
                "cvar_t": cvar,
                "certeza": certeza,
                "momentum_veto": mom,
                "lambda_crono": crono,
            })
            shadow_str = (
                f"[{gov}: r_veto={r_veto} cvar={cvar:.4f} "
                f"crono={crono:.3f} cert={certeza:.3f} mom={int(mom)}]"
            )
    return shadow_str, scores


class MasterTelemetryPacker:
    """Institutional telemetry and audit packer for Master Equation outputs."""

    @classmethod
    def pack(cls, plan: Any, state: Any) -> tuple[str, dict[str, Any]]:
        """Packs metrics and updates bot state from an ExecutionPlan.

        Args:
            plan: The ExecutionPlan produced by the engine.
            state: The LiveBotState to update.

        Returns:
            tuple[str, dict[str, Any]]: Diagnostic string and scores dictionary.
        """
        return extract_decision_trace_metrics(plan, state)

    @classmethod
    def create_weaviate_telemetry_record(
        cls,
        sym: str,
        mid: float,
        phi_moe: float,
        lambda_t: float,
        trigger_side: str,
        trigger_code: int,
        reason: str,
        scores: dict[str, Any],
    ) -> dict[str, Any]:
        """Builds a structured dictionary for Weaviate cognitive memory ingestion.

        Args:
            sym: Ticker symbol.
            mid: Current midpoint price.
            phi_moe: Epistemic MoE score or Certeza score.
            lambda_t: Temporal exploration / acceleration factor.
            trigger_side: Trigger direction ('buy', 'sell', 'hold', 'flush').
            trigger_code: Numerical trigger code.
            reason: Veto or execution rationale string.
            scores: Breakdown of expert and engine scores.

        Returns:
            dict[str, Any]: Formatted telemetry dictionary.
        """
        return {
            "symbol": sym,
            "mid_price": float(mid),
            "phi_moe": float(phi_moe),
            "cvar": float(scores.get("cvar_t", 0.0)),
            "lambda_t": float(lambda_t),
            "action": f"{trigger_side.upper()}:{trigger_code}",
            "reason": reason,
            "decision_rationale": scores,
        }

    @classmethod
    def create_weaviate_execution_record(
        cls,
        sym: str,
        side: str,
        qty: float,
        entry_price: float,
        pnl_usd: float = 0.0,
        status: str = "FILLED",
    ) -> dict[str, Any]:
        """Builds a structured dictionary for Weaviate execution logging.

        Args:
            sym: Ticker symbol.
            side: Execution side ('buy', 'sell').
            qty: Order quantity in units/shares.
            entry_price: Execution price.
            pnl_usd: Realized PnL if known.
            status: Order status.

        Returns:
            dict[str, Any]: Execution record dictionary.
        """
        return {
            "symbol": sym,
            "side": side,
            "qty": float(qty),
            "entry_price": float(entry_price),
            "pnl_usd": float(pnl_usd),
            "status": status,
        }
