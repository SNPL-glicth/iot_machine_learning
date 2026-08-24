"""FASE 10.3 — Strategy Card Rendering."""

from __future__ import annotations

from .types import StrategyCard


def render_strategy_card(card: StrategyCard) -> str:
    """Renderiza ficha técnica ASCII."""
    lines = [
        f"EXPERT: {card.strategy.upper()}",
        "=" * (len(f"EXPERT: {card.strategy.upper()}")),
        "",
        f"Samples: {card.samples:,}",
        f"History: {card.history_days:.1f} days",
        "",
        "DIRECTION METRICS",
        "-" * 17,
        f"Accuracy: {card.accuracy:.2%}",
        f"95% CI (Wilson): [{card.wilson_lb:.2%}, {card.wilson_ub:.2%}]",
        f"95% CI (Normal): [{card.accuracy_ci[0]:.2%}, {card.accuracy_ci[1]:.2%}]",
        f"Significant vs chance: {'YES' if card.is_significant else 'NO'} (p={card.p_value:.4f})",
        "",
        "ECONOMIC METRICS",
        "-" * 16,
        f"Gross edge: {card.gross_edge:+.4f}",
        f"Costs: {card.costs:+.4f}",
        f"Net edge: {card.net_edge:+.4f}",
        f"Total PnL: {card.total_pnl:+.4f}",
        "",
        "RISK METRICS",
        "-" * 13,
        f"Sharpe ratio: {card.sharpe_ratio:.3f}",
        f"Max drawdown: {card.max_drawdown:.2%}",
        f"Volatility: {card.volatility:.4f}",
        "",
        "CALIBRATION",
        "-" * 11,
        f"Brier score: {card.brier_score:.4f}",
        f"ECE: {card.ece:.4f}",
        "",
        "EVIDENCE",
        "-" * 8,
        f"Status: {card.evidence_status}",
        f"Reason: {card.evidence_reason}",
        "",
    ]
    
    # Veredicto
    if card.is_profitable and card.is_significant:
        lines.append("VEREDICT: PROFITABLE & SIGNIFICANT 🎯")
    elif card.is_profitable:
        lines.append("VEREDICT: PROFITABLE but NOT SIGNIFICANT ⚠️")
    elif card.is_significant:
        lines.append("VEREDICT: SIGNIFICANT but NOT PROFITABLE ❌")
    else:
        lines.append("VEREDICT: NEITHER PROFITABLE NOR SIGNIFICANT ❌")
    
    return "\n".join(lines)


def render_strategy_comparison(cards: dict[str, StrategyCard]) -> str:
    """Renderiza tabla comparativa de estrategias."""
    lines = [
        "STRATEGY COMPARISON",
        "=" * 19,
        "",
        f"{'Strategy':<15} {'Samples':>8} {'Accuracy':>10} {'Wilson LB':>10} {'Net Edge':>10} {'Sharpe':>8} {'Max DD':>8} {'Signif':>8}",
        "-" * 87,
    ]
    
    for strategy, card in sorted(cards.items()):
        lines.append(
            f"{strategy:<15} {card.samples:>8} {card.accuracy:>10.2%} {card.wilson_lb:>10.2%} "
            f"{card.net_edge:>10.4f} {card.sharpe_ratio:>8.2f} {card.max_drawdown:>8.2%} "
            f"{'YES' if card.is_significant else 'NO':>8}"
        )
    
    return "\n".join(lines)