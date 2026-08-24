"""Walk-forward report rendering."""

from __future__ import annotations

from collections.abc import Sequence

from ..costs import EDGE_RISK_ADJUSTED_POSITIVE, classify_edge
from .eval import WfRow
from .sharpe import sharpe

__all__ = ["render_wf_report"]


def render_wf_report(
    rows: Sequence[WfRow],
    *,
    symbol: str,
    interval_label: str,
    box_width: int = 60,
) -> str:
    """Reporte ASCII: una caja por ventana + resumen agregado."""
    out: list[str] = []

    def row(line: str) -> str:
        return "║ " + line.ljust(box_width - 4) + " ║"

    def rule(char: str = "═") -> str:
        return "╠" + char * (box_width - 2) + "╣"

    def top() -> str:
        return "╔" + "═" * (box_width - 2) + "╗"

    def bottom() -> str:
        return "╚" + "═" * (box_width - 2) + "╝"

    out.append(top())
    out.append(row("WALK-FORWARD — ZENIN MARKET (FASE 9.2)"))
    out.append(row(f"instrumento: {symbol}   resolución: {interval_label}"))
    out.append(rule())

    if not rows:
        out.append(row("  (sin ventanas: dataset o parametros sin cobertura)"))
        out.append(bottom())
        return "\n".join(out)

    cost_bps = rows[0].cost_bps
    for w in rows:
        out.append(
            row(
                f"W{w.index:02d} {w.regime or 'ALL':<14} "
                f"test {w.model_accuracy:.1%} reward {w.model_reward:+.4f} "
                f"n={w.n_test}"
            )
        )
        out.append(
            row(
                f"    train {w.train_start:.0f}->{w.train_end:.0f} | "
                f"test {w.test_start:.0f}->{w.test_end:.0f}"
            )
        )
        if w.edge_class is not None and w.realized_gross is not None:
            gross = w.realized_gross
            net = w.realized_net if w.realized_net is not None else gross
            sh = f" sharpe {w.sharpe:.2f}" if w.sharpe is not None else ""
            out.append(row(f"    edge {gross:+.2%} -> {net:+.2%} [{w.edge_class}{sh}]"))
        for h in w.horizons:
            bits = ", ".join(
                f"{name}: {e['accuracy']:.1%}/{e['mean_reward']:+.4f}"
                for name, e in sorted(h.experts.items())
            )
            out.append(
                row(
                    f"    {h.horizon_seconds}s n={h.n:<4} "
                    f"modelo {h.model.model_accuracy:.1%} "
                    f"reward {h.model.model_reward:+.4f}"
                )
            )
            out.append(row(f"      {bits}"))
            if h.edge is not None:
                out.append(
                    row(
                        f"      exp {h.edge.expected_gross:+.2%} "
                        f"-> {h.edge.expected_net:+.2%} | "
                        f"real {h.edge.realized_gross:+.2%} "
                        f"-> {h.edge.realized_net:+.2%} "
                        f"(costos {h.edge.cost_bps}bps)"
                    )
                )
        if w.note:
            out.append(row(f"    nota: {w.note}"))
        out.append(rule())

    n = sum(w.n_test for w in rows)
    acc = sum(w.model_accuracy * w.n_test for w in rows) / n if n else 0.0
    reward = sum(w.model_reward * w.n_test for w in rows) / n if n else 0.0
    positive_windows = sum(1 for w in rows if w.model_reward > 0)
    by_regime: dict[str, list[float]] = {}
    for w in rows:
        by_regime.setdefault(w.regime or "ALL", []).append(w.model_reward)
    out.append(row(f"AGREGADO: {len(rows)} ventanas | n={n}"))
    out.append(
        row(
            f"  modelo: acc {acc:.1%}  reward {reward:+.4f}  "
            f"ventanas positivas {positive_windows}/{len(rows)}"
        )
    )
    for regime, rewards in sorted(by_regime.items()):
        mean = sum(rewards) / len(rewards)
        out.append(row(f"  {regime:<14} reward medio {mean:+.4f} (n={len(rewards)})"))

    # FASE 9.2: edge agregado (bruto -> neto) + clasificación.
    edge_rows: list[tuple[WfRow, float, float]] = []
    for w in rows:
        w_gross = w.realized_gross
        w_net = w.realized_net
        if w_gross is not None and w_net is not None:
            edge_rows.append((w, w_gross, w_net))
    if edge_rows and cost_bps:
        total_n = sum(w.n_test for w, _, _ in edge_rows) or 1
        gross = sum(gross * w.n_test for w, gross, _ in edge_rows) / total_n
        net = sum(net * w.n_test for w, _, net in edge_rows) / total_n
        nets = [net for _, _, net in edge_rows]
        sharpe_val = sharpe(nets) if len(nets) > 1 else None
        edge_class = classify_edge(gross, net, sharpe=sharpe_val)
        out.append(
            row(
                f"  EDGE ({cost_bps}bps): {gross:+.2%} -> {net:+.2%} "
                f"[{edge_class}" + (f" sharpe {sharpe_val:.2f}]" if sharpe_val is not None else "]")
            )
        )
        positives = [w for w, _, _ in edge_rows if w.edge_class == EDGE_RISK_ADJUSTED_POSITIVE]
        if positives:
            out.append(
                row(f"  ventanas con edge riesgo-ajustado: {len(positives)}/{len(edge_rows)}")
            )
    out.append(bottom())
    return "\n".join(out)