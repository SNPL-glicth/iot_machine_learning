"""Action message builders for ThresholdPolicy.

Pure string-generation helpers; no I/O, no state.
"""

from __future__ import annotations

from typing import Optional


def build_action(
    severity_label: str,
    risk_level: str,
    label: str,
) -> str:
    """Build recommended action for numeric/physical domain."""
    if severity_label == "critical":
        return (
            f"Condición crítica{f' en {label}' if label else ''}. "
            "Requiere atención inmediata."
        )
    if severity_label == "warning":
        if risk_level == "HIGH":
            return (
                f"Riesgo elevado{f' en {label}' if label else ''}. "
                "Programar revisión prioritaria."
            )
        return (
            f"Comportamiento inusual{f' en {label}' if label else ''}. "
            "Supervisar de cerca."
        )
    return "Valores dentro del rango esperado. No se requiere acción."


def build_text_action(
    severity_label: str,
    domain: str,
) -> str:
    """Build recommended action for text/document domain."""
    domain_label = domain if domain != "general" else "document"
    if severity_label == "critical":
        return (
            f"Critical issues detected in {domain_label}. "
            "Immediate review and action required."
        )
    if severity_label == "warning":
        return (
            f"Concerns identified in {domain_label}. "
            "Schedule review and monitor closely."
        )
    return "No immediate action required. Continue standard monitoring."
