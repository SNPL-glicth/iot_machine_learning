"""Hampel outlier filter for engine perceptions (IMP-2).

Median + k · 1.4826 · MAD (Median Absolute Deviation) rule. The
1.4826 scale factor (= 1/Φ⁻¹(0.75)) converts MAD to a σ estimate
under a Gaussian assumption, so ``k=3`` is a ≈3σ threshold.

Hampel Filter — opera sobre ENGINE PREDICTIONS (post-predicción, pre-fusión).

Fórmula: median ± k × 1.4826 × MAD donde k=ML_HAMPEL_K=3.0

Diferencia con IQR Detector:
- Hampel opera en PREDICCIONES de motores (no datos crudos)
- Hampel usa MAD (robusto a outliers): apropiado para predicciones
  donde errores extremos son esperables en motores erróneos
- k=3.0 × 1.4826 ≈ 4.45σ efectivos bajo normalidad
  (más permisivo que Z_SCORE_UPPER=3.0σ intencionalmente)

¿Por qué más permisivo que Z-score?
- Predicciones tienen mayor varianza inherente que datos crudos
- Hampel filtra solo predicciones EXTREMADAMENTE atípicas
- Filtrado agresivo reduciría diversidad del ensemble (ver ENS-3)

Bajo normalidad: MAD ≈ 0.6745σ → 3.0×MAD ≈ 2.0σ via MAD scaling
Con factor 1.4826: 3.0×1.4826×MAD ≈ 3.0σ (equivalente a Z_SCORE_UPPER)

Pure function — no state, no I/O. Only looks at ``predicted_value``;
confidence is handled upstream by ``InhibitionGate``.

Edge cases:
    * fewer than ``min_perceptions`` entries → no filtering.
    * MAD == 0 (all identical, or flat-line cluster) → no filtering.
    * ``k <= 0`` → no filtering (defensive).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from ..analysis.types import EnginePerception


MAD_GAUSSIAN_SCALE: float = 1.4826  # 1 / Φ⁻¹(0.75)


@dataclass(frozen=True)
class HampelResult:
    """Outcome of :func:`hampel_filter`.

    Attributes:
        kept: Perceptions surviving the filter (order preserved).
        rejected: ``(engine_name, predicted_value, z_score)`` triples
            for every dropped perception. ``z_score`` is
            ``|value − median| / (1.4826 · MAD)``.
        median: Median of the input ``predicted_value`` distribution.
        mad: Median absolute deviation of the same distribution.
    """

    kept: List[EnginePerception]
    rejected: List[Tuple[str, float, float]] = field(default_factory=list)
    median: float = 0.0
    mad: float = 0.0

    def to_dict(self) -> dict:
        return {
            "median": self.median,
            "mad": self.mad,
            "rejected": [
                {"engine": n, "predicted_value": v, "z_score": z}
                for (n, v, z) in self.rejected
            ],
        }


def _median(values: Sequence[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    ordered = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def hampel_filter(
    perceptions: List[EnginePerception],
    *,
    k: float = 3.0,
    min_perceptions: int = 3,
) -> HampelResult:
    """Drop perceptions whose predicted_value lies beyond ``k·σ̂``.

    ``σ̂ = 1.4826 · MAD``. Order of ``kept`` matches input order.
    """
    if not perceptions:
        return HampelResult(kept=[], rejected=[], median=0.0, mad=0.0)

    if len(perceptions) < min_perceptions or k <= 0.0:
        return HampelResult(kept=list(perceptions), rejected=[], median=0.0, mad=0.0)

    predicted_values = [float(p.predicted_value) for p in perceptions]
    median = _median(predicted_values)
    deviations = [abs(v - median) for v in predicted_values]
    mad = _median(deviations)

    if mad <= 0.0:
        return HampelResult(kept=list(perceptions), rejected=[], median=median, mad=0.0)

    sigma_hat = MAD_GAUSSIAN_SCALE * mad
    threshold = k * sigma_hat

    kept: List[EnginePerception] = []
    rejected: List[Tuple[str, float, float]] = []
    for p, v in zip(perceptions, predicted_values):
        z = abs(v - median) / sigma_hat
        if abs(v - median) > threshold:
            rejected.append((p.engine_name, v, z))
        else:
            kept.append(p)

    return HampelResult(kept=kept, rejected=rejected, median=median, mad=mad)


def hampel_filter_with_profile(
    perceptions: List[Any],
    *,
    sensor_profile: Optional[object] = None,
    event_context: Optional[object] = None,
    min_perceptions: int = 3,
) -> HampelResult:
    """Wrapper de hampel_filter() que selecciona k y ventana desde SensorProfile."""
    k = 3.0
    if sensor_profile is not None:
        k = getattr(sensor_profile, "hampel_k", 3.0)
    if event_context is not None:
        detected = getattr(event_context, "detected_event", None)
        if detected is not None:
            k *= 1.5
    window = getattr(sensor_profile, "hampel_window", min_perceptions) if sensor_profile else min_perceptions
    effective_min = max(min_perceptions, min(window, len(perceptions)))
    return hampel_filter(perceptions, k=k, min_perceptions=effective_min)


__all__ = ["HampelResult", "hampel_filter", "hampel_filter_with_profile", "MAD_GAUSSIAN_SCALE"]
