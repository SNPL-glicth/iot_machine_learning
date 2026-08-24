"""FASE 10.2 — Evidence Engine: evaluación multidimensional de evidencia.

Objetivo:
- Construir sistema que decide si hay evidencia suficiente para confiar en un contexto
- Considerar no solo dirección (accuracy) sino magnitud, costos y estabilidad
- Basarse en el descubrimiento de 9.5: accuracy ≠ rentabilidad
- Estados: INSUFFICIENT_EVIDENCE vs EVIDENCE_SUPPORTED

Dimensiones de evidencia:
1. Sample size: n mínimo por contexto
2. Direction accuracy: Wilson 95% lower bound ≥ umbral
3. Magnitude quality: error promedio de magnitud ≤ tolerancia
4. Economic edge: return neto después de costos ≥ mínimo
5. Stability: no degradación temporal (recency bands)
6. Calibration: calibración aceptable (ECE ≤ tolerancia)

El engine produce un veredicto auditable con razones específicas.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Final

from ..adaptation.guard import wilson_lower_bound
from ..calibration import compute_ece


__all__ = [
    "EvidenceStatus",
    "EvidenceConfig",
    "EvidenceDimension",
    "EvidenceVerdict",
    "EvidenceEngine",
]


class EvidenceStatus(Enum):
    """Estado de evidencia para un contexto."""
    
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EVIDENCE_SUPPORTED = "evidence_supported"
    EVIDENCE_DEGRADED = "evidence_degraded"  # Tenía evidencia pero degradó


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    """Configuración umbrales para evaluación de evidencia."""
    
    # Muestra mínima
    min_n: int = 100  # Mínimo 100 observaciones
    min_history_days: int = 7  # Mínimo 7 días de historial
    
    # Dirección
    min_accuracy: float = 0.52  # Accuracy mínimo
    wilson_confidence: float = 0.95  # Nivel de confianza Wilson
    
    # Magnitud
    max_magnitude_error: float = 0.02  # Error máximo de magnitud (2%)
    
    # Económico
    min_expected_net: float = 0.001  # Return neto mínimo (0.1%)
    max_cost_ratio: float = 0.5  # Costos no pueden ser >50% del return
    
    # Estabilidad
    max_recency_degradation: float = 0.05  # Degradación máxima por banda (5%)
    min_bands: int = 2  # Mínimo 2 bandas para evaluar estabilidad
    
    # Calibración
    max_ece: float = 0.10  # ECE máximo (10%)
    
    def __post_init__(self) -> None:
        if self.min_n < 1:
            raise ValueError(f"min_n debe ser >= 1: {self.min_n}")
        if not 0.0 < self.min_accuracy < 1.0:
            raise ValueError(f"min_accuracy inválido: {self.min_accuracy}")
        if self.max_magnitude_error < 0:
            raise ValueError(f"max_magnitude_error no puede ser negativo: {self.max_magnitude_error}")


@dataclass(frozen=True, slots=True)
class EvidenceDimension:
    """Resultado de evaluación de una dimensión de evidencia."""
    
    name: str
    passed: bool
    value: float
    threshold: float
    reason: str


@dataclass(frozen=True, slots=True)
class EvidenceVerdict:
    """Veredicto completo de evidencia para un contexto."""
    
    context: str  # "strategy·horizon·regime"
    status: EvidenceStatus
    dimensions: tuple[EvidenceDimension, ...]
    n: int
    history_days: int
    accuracy: float
    wilson_lb: float
    expected_net: float
    calibration_ece: float
    
    @property
    def failed_dimensions(self) -> tuple[EvidenceDimension, ...]:
        return tuple(d for d in self.dimensions if not d.passed)
    
    @property
    def summary(self) -> str:
        """Resumen humano del veredicto."""
        if self.status == EvidenceStatus.EVIDENCE_SUPPORTED:
            return f"EVIDENCE_SUPPORTED: {len(self.dimensions)} dimensiones pasaron"
        elif self.status == EvidenceStatus.EVIDENCE_DEGRADED:
            return f"EVIDENCE_DEGRADED: {len(self.failed_dimensions)} dimensiones fallaron"
        else:
            return f"INSUFFICIENT_EVIDENCE: {len(self.failed_dimensions)} dimensiones fallaron"


class EvidenceEngine:
    """Engine de evaluación de evidencia multidimensional."""
    
    def __init__(self, config: EvidenceConfig | None = None) -> None:
        self.config = config or EvidenceConfig()
    
    def evaluate(
        self,
        context: str,
        # Datos del contexto
        n: int,
        history_days: int,
        # Dirección
        accuracy: float,
        # Magnitud
        magnitude_errors: list[float],
        # Económico
        expected_returns: list[float],
        costs: list[float],
        # Estabilidad
        recency_accuracies: list[float],  # [antigua, ..., reciente]
        # Calibración
        calibration_errors: list[float],
    ) -> EvidenceVerdict:
        """Evalúa evidencia multidimensional para un contexto."""
        dimensions: list[EvidenceDimension] = []
        
        # 1. Sample size
        sample_passed = n >= self.config.min_n and history_days >= self.config.min_history_days
        dimensions.append(EvidenceDimension(
            name="sample_size",
            passed=sample_passed,
            value=float(n),
            threshold=float(self.config.min_n),
            reason=f"n={n}, days={history_days}" + (f" >= {self.config.min_n}" if sample_passed else f" < {self.config.min_n}")
        ))
        
        # 2. Direction accuracy (Wilson)
        hits = int(accuracy * n)
        wilson_lb_val = wilson_lower_bound(hits, n, z=1.96)  # 95% confidence
        accuracy_passed = wilson_lb_val >= self.config.min_accuracy
        dimensions.append(EvidenceDimension(
            name="direction_accuracy",
            passed=accuracy_passed,
            value=wilson_lb_val,
            threshold=self.config.min_accuracy,
            reason=f"Wilson 95% LB = {wilson_lb_val:.4f} vs {self.config.min_accuracy:.4f}"
        ))
        
        # 3. Magnitude quality
        avg_magnitude_error = sum(magnitude_errors) / len(magnitude_errors) if magnitude_errors else 0.0
        magnitude_passed = avg_magnitude_error <= self.config.max_magnitude_error
        dimensions.append(EvidenceDimension(
            name="magnitude_quality",
            passed=magnitude_passed,
            value=avg_magnitude_error,
            threshold=self.config.max_magnitude_error,
            reason=f"avg error = {avg_magnitude_error:.4f} vs {self.config.max_magnitude_error:.4f}"
        ))
        
        # 4. Economic edge
        avg_expected_return = sum(expected_returns) / len(expected_returns) if expected_returns else 0.0
        avg_cost = sum(costs) / len(costs) if costs else 0.0
        expected_net = avg_expected_return - avg_cost
        economic_passed = expected_net >= self.config.min_expected_net and avg_cost <= abs(avg_expected_return) * self.config.max_cost_ratio
        dimensions.append(EvidenceDimension(
            name="economic_edge",
            passed=economic_passed,
            value=expected_net,
            threshold=self.config.min_expected_net,
            reason=f"net = {expected_net:.4f} (return {avg_expected_return:.4f} - cost {avg_cost:.4f})"
        ))
        
        # 5. Stability (recency)
        if len(recency_accuracies) >= self.config.min_bands:
            oldest = recency_accuracies[0]
            newest = recency_accuracies[-1]
            degradation = oldest - newest
            stability_passed = degradation <= self.config.max_recency_degradation
            dimensions.append(EvidenceDimension(
                name="stability",
                passed=stability_passed,
                value=degradation,
                threshold=self.config.max_recency_degradation,
                reason=f"degradation = {degradation:.4f} (oldest {oldest:.4f} -> newest {newest:.4f})"
            ))
        else:
            dimensions.append(EvidenceDimension(
                name="stability",
                passed=False,
                value=0.0,
                threshold=self.config.max_recency_degradation,
                reason=f"insufficient bands: {len(recency_accuracies)} < {self.config.min_bands}"
            ))
        
        # 6. Calibration
        # Simular prob_outcomes para ECE
        if calibration_errors:
            avg_calibration_error = sum(calibration_errors) / len(calibration_errors)
            calibration_passed = avg_calibration_error <= self.config.max_ece
            dimensions.append(EvidenceDimension(
                name="calibration",
                passed=calibration_passed,
                value=avg_calibration_error,
                threshold=self.config.max_ece,
                reason=f"avg calibration error = {avg_calibration_error:.4f} vs {self.config.max_ece:.4f}"
            ))
        else:
            dimensions.append(EvidenceDimension(
                name="calibration",
                passed=False,
                value=0.0,
                threshold=self.config.max_ece,
                reason="no calibration data"
            ))
        
        # Determinar estado
        all_passed = all(d.passed for d in dimensions)
        
        if all_passed:
            status = EvidenceStatus.EVIDENCE_SUPPORTED
        elif dimensions[0].passed:  # Sample size passed but others failed
            status = EvidenceStatus.EVIDENCE_DEGRADED
        else:
            status = EvidenceStatus.INSUFFICIENT_EVIDENCE
        
        return EvidenceVerdict(
            context=context,
            status=status,
            dimensions=tuple(dimensions),
            n=n,
            history_days=history_days,
            accuracy=accuracy,
            wilson_lb=wilson_lb_val,
            expected_net=expected_net,
            calibration_ece=sum(calibration_errors) / len(calibration_errors) if calibration_errors else 0.0,
        )
    
    def batch_evaluate(
        self,
        contexts_data: dict[str, dict],  # context -> {n, history_days, accuracy, magnitude_errors, expected_returns, costs, recency_accuracies, calibration_errors}
    ) -> dict[str, EvidenceVerdict]:
        """Evalúa múltiples contextos."""
        results = {}
        for context, data in contexts_data.items():
            results[context] = self.evaluate(
                context=context,
                n=data["n"],
                history_days=data["history_days"],
                accuracy=data["accuracy"],
                magnitude_errors=data.get("magnitude_errors", []),
                expected_returns=data.get("expected_returns", []),
                costs=data.get("costs", []),
                recency_accuracies=data.get("recency_accuracies", []),
                calibration_errors=data.get("calibration_errors", []),
            )
        return results


def render_evidence_report(verdict: EvidenceVerdict) -> str:
    """Renderiza reporte ASCII de un veredicto de evidencia."""
    lines = [
        f"EVIDENCE ENGINE — {verdict.context}",
        "=" * (len(f"EVIDENCE ENGINE — {verdict.context}")),
        "",
        f"STATUS: {verdict.status.value}",
        f"Summary: {verdict.summary}",
        "",
        f"Context metrics:",
        f"  n: {verdict.n:,}",
        f"  history_days: {verdict.history_days}",
        f"  accuracy: {verdict.accuracy:.4f}",
        f"  Wilson 95% LB: {verdict.wilson_lb:.4f}",
        f"  expected_net: {verdict.expected_net:+.4f}",
        f"  calibration_ece: {verdict.calibration_ece:.4f}",
        "",
        "DIMENSION EVALUATION:",
        "-" * 25,
    ]
    
    for dim in verdict.dimensions:
        status = "✓ PASS" if dim.passed else "✗ FAIL"
        lines.append(f"  {dim.name:<20} {status:<8} value={dim.value:.4f} threshold={dim.threshold:.4f}")
        lines.append(f"    Reason: {dim.reason}")
        lines.append("")
    
    return "\n".join(lines)