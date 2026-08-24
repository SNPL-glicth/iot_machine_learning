"""FASE 10.5 — Calibration Comparison Rendering."""

from __future__ import annotations

from .comparison import CalibrationComparison
from .verdicts import CalibrationVerdict


def render_calibration_comparison(comparison: CalibrationComparison) -> str:
    """Renderiza comparación raw vs calibrated."""
    lines = [
        f"CALIBRATION COMPARISON — {comparison.context}",
        "=" * (len(f"CALIBRATION COMPARISON — {comparison.context}")),
        "",
        f"Samples: Train={comparison.n_train}, Val={comparison.n_val}, Test={comparison.n_test}",
        "",
        f"VEREDICT: {comparison.verdict.value.upper()}",
        f"Reason: {comparison.rejection_reason or 'No rejection reason'}",
        "",
        "RAW vs CALIBRATED (TEST)",
        "-" * 30,
        f"{'Metric':<20} {'RAW':>12} {'CALIBRATED':>12} {'Δ':>12}",
        "-" * 60,
        f"{'Brier':<20} {comparison.raw_brier:>12.4f} {comparison.calibrated_brier:>12.4f} {comparison.brier_improvement:>12.4f}",
        f"{'ECE':<20} {comparison.raw_ece:>12.4f} {comparison.calibrated_ece:>12.4f} {comparison.ece_improvement:>12.4f}",
        f"{'Log Loss':<20} {comparison.raw_log_loss:>12.4f} {comparison.calibrated_log_loss:>12.4f} {comparison.log_loss_improvement:>12.4f}",
        f"{'Wilson LB':<20} {comparison.raw_wilson_lb:>12.4f} {comparison.calibrated_wilson_lb:>12.4f} {comparison.wilson_improvement:>12.4f}",
        f"{'Economic Edge':<20} {comparison.raw_economic_edge:>12.4f} {comparison.calibrated_economic_edge:>12.4f} {comparison.economic_impact:>12.4f}",
        "",
    ]
    
    if comparison.verdict == CalibrationVerdict.ACCEPTED:
        lines.append("✓ CALIBRATOR ACCEPTED — Mejora estadística sin degradación económica")
    else:
        lines.append("✗ CALIBRATOR REJECTED — No cumple criterios de aceptación")
    
    return "\n".join(lines)