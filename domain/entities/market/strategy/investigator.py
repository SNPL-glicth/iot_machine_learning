"""FASE 10.3 — Strategy Investigator."""

from __future__ import annotations

import math
from typing import Final

from ..adaptation.guard import wilson_lower_bound
from ..calibration import compute_brier, compute_ece
from .risk_metrics import compute_sharpe_ratio, compute_max_drawdown, compute_confidence_interval
from .types import StrategyCard


__all__ = ["StrategyInvestigator"]


class StrategyInvestigator:
    """Investigador de estrategias con análisis estadístico."""
    
    def __init__(self, risk_free_rate: float = 0.0) -> None:
        self.risk_free_rate = risk_free_rate
    
    def investigate_strategy(
        self,
        strategy: str,
        # Datos
        direction_correct: list[bool],
        returns: list[float],
        costs: list[float],
        # Calibración
        probabilities: list[float],
        # Metadata
        history_days: float,
        # Evidence (opcional)
        evidence_status: str = "unknown",
        evidence_reason: str = "",
    ) -> StrategyCard:
        """Investiga una estrategia y produce su ficha técnica."""
        
        if not direction_correct:
            raise ValueError("direction_correct no puede estar vacío")
        
        n = len(direction_correct)
        
        # Direction metrics
        accuracy = sum(1 for d in direction_correct if d) / n
        hits = sum(1 for d in direction_correct if d)
        wilson_lb_val = wilson_lower_bound(hits, n, z=1.96)
        # Wilson upper bound (simétrico aprox)
        wilson_ub_val = min(1.0, accuracy + (accuracy - wilson_lb_val))
        
        # Confidence interval normal
        acc_ci = compute_confidence_interval([float(d) for d in direction_correct])
        
        # Economic metrics
        gross_edge = sum(returns) / n if returns else 0.0
        avg_costs = sum(costs) / n if costs else 0.0
        net_edge = gross_edge - avg_costs
        total_pnl = sum(returns) - sum(costs)
        
        # Risk metrics
        sharpe = compute_sharpe_ratio(returns, self.risk_free_rate)
        max_dd = compute_max_drawdown(returns)
        volatility = math.sqrt(sum((r - gross_edge) ** 2 for r in returns) / (n - 1)) if n > 1 else 0.0
        
        # Calibration metrics
        brier = compute_brier(probabilities, direction_correct) if probabilities else 0.0
        ece = compute_ece(probabilities, direction_correct) if probabilities else 0.0
        
        # Statistical significance (prueba binomial vs azar 50%)
        # H0: p = 0.5, H1: p ≠ 0.5
        # Usar normal approximation (valido para n >= 30)
        if n >= 30:
            z_score = (accuracy - 0.5) / math.sqrt(0.5 * 0.5 / n)
            # Two-tailed test usando error function
            if z_score >= 0:
                p_val = 2 * (1 - (0.5 * (1 + math.erf(z_score / math.sqrt(2)))))
            else:
                p_val = 2 * (0.5 * (1 + math.erf(z_score / math.sqrt(2))))
        else:
            # Para n pequeño, usar Wilson lower bound como proxy
            # Si Wilson LB > 0.5, es significativo
            p_val = 0.10 if wilson_lb_val > 0.5 else 0.20  # Approximation
        
        is_significant = p_val < 0.05  # 95% confidence
        
        return StrategyCard(
            strategy=strategy,
            context=strategy,  # Para estrategia simple
            samples=n,
            history_days=history_days,
            accuracy=accuracy,
            wilson_lb=wilson_lb_val,
            wilson_ub=wilson_ub_val,
            accuracy_ci=acc_ci,
            gross_edge=gross_edge,
            costs=avg_costs,
            net_edge=net_edge,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            volatility=volatility,
            brier_score=brier,
            ece=ece,
            evidence_status=evidence_status,
            evidence_reason=evidence_reason,
            is_significant=is_significant,
            p_value=p_val,
        )
    
    def batch_investigate(
        self,
        strategies_data: dict[str, dict],  # strategy -> {direction_correct, returns, costs, probabilities, history_days, evidence_status, evidence_reason}
    ) -> dict[str, StrategyCard]:
        """Investiga múltiples estrategias."""
        results = {}
        for strategy, data in strategies_data.items():
            try:
                results[strategy] = self.investigate_strategy(
                    strategy=strategy,
                    direction_correct=data["direction_correct"],
                    returns=data.get("returns", []),
                    costs=data.get("costs", []),
                    probabilities=data.get("probabilities", []),
                    history_days=data.get("history_days", 0.0),
                    evidence_status=data.get("evidence_status", "unknown"),
                    evidence_reason=data.get("evidence_reason", ""),
                )
            except Exception as e:
                print(f"Error investigando {strategy}: {e}")
                continue
        return results