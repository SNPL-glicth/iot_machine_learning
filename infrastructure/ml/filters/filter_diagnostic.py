"""Diagnóstico de calidad de filtrado — métricas observables.

Value object que cuantifica cuánto ruido eliminó un filtro y si
está distorsionando la señal.  Consumido por el orquestador cognitivo
para evaluar y seleccionar filtros dinámicamente.

Sin I/O, sin estado, sin dependencias de infraestructura.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FilterDiagnostic:
    """Métricas de calidad de un paso de filtrado.

    Attributes:
        n_points: Número de puntos procesados.
        raw_std: Desviación estándar de la señal cruda.
        filtered_std: Desviación estándar de la señal filtrada.
        noise_reduction_ratio: 1 - (filtered_std / raw_std).
            0.0 = sin reducción, 1.0 = ruido eliminado por completo.
            Negativo si el filtro *añade* variabilidad.
        mean_absolute_error: MAE entre señal cruda y filtrada.
            Proxy de cuánto se desvía el filtro de la entrada.
        max_absolute_error: Máxima desviación puntual.
        lag_estimate: Estimación de retardo del filtro en muestras.
            0 = sin lag (ideal).  Calculado por correlación cruzada.
        signal_distortion: Ratio de energía perdida/añadida.
            abs(mean_raw - mean_filtered) / max(raw_std, ε).
            0.0 = sin distorsión de nivel.
    """

    n_points: int = 0
    raw_std: float = 0.0
    filtered_std: float = 0.0
    noise_reduction_ratio: float = 0.0
    mean_absolute_error: float = 0.0
    max_absolute_error: float = 0.0
    lag_estimate: int = 0
    signal_distortion: float = 0.0

    @property
    def is_effective(self) -> bool:
        """True si el filtro reduce ruido sin distorsionar demasiado."""
        return self.noise_reduction_ratio > 0.05 and self.signal_distortion < 0.5

    @property
    def is_distorting(self) -> bool:
        """True si el filtro distorsiona significativamente la señal."""
        return self.signal_distortion > 0.3

    def to_dict(self) -> dict:
        """Serializa para audit logging / metadata."""
        return {
            "n_points": self.n_points,
            "raw_std": round(self.raw_std, 8),
            "filtered_std": round(self.filtered_std, 8),
            "noise_reduction_ratio": round(self.noise_reduction_ratio, 6),
            "mean_absolute_error": round(self.mean_absolute_error, 8),
            "max_absolute_error": round(self.max_absolute_error, 8),
            "lag_estimate": self.lag_estimate,
            "signal_distortion": round(self.signal_distortion, 6),
        }

    @classmethod
    def empty(cls) -> FilterDiagnostic:
        """Factory para serie vacía o insuficiente."""
        return cls()


def compute_filter_diagnostic(
    raw: List[float],
    filtered: List[float],
) -> FilterDiagnostic:
    """Computa diagnóstico comparando señal cruda vs filtrada.

    Args:
        raw: Valores originales (sin filtrar).
        filtered: Valores después de aplicar el filtro.
            Debe tener la misma longitud que ``raw``.

    Returns:
        ``FilterDiagnostic`` con métricas de calidad.
    """
    n = len(raw)
    if n == 0 or len(filtered) != n:
        return FilterDiagnostic.empty()

    # --- Estadísticas básicas ---
    raw_mean = sum(raw) / n
    filt_mean = sum(filtered) / n

    raw_var = sum((v - raw_mean) ** 2 for v in raw) / max(n - 1, 1)
    filt_var = sum((v - filt_mean) ** 2 for v in filtered) / max(n - 1, 1)

    raw_std = math.sqrt(raw_var)
    filt_std = math.sqrt(filt_var)

    # --- Noise reduction ---
    if raw_std > 1e-12:
        noise_reduction = 1.0 - (filt_std / raw_std)
    else:
        noise_reduction = 0.0

    # --- Error metrics ---
    abs_errors = [abs(r - f) for r, f in zip(raw, filtered)]
    mae = sum(abs_errors) / n
    max_ae = max(abs_errors)

    # --- Signal distortion ---
    ref = max(raw_std, 1e-12)
    distortion = abs(raw_mean - filt_mean) / ref

    # --- Lag estimation (simplified cross-correlation) ---
    lag = _estimate_lag(raw, filtered)

    return FilterDiagnostic(
        n_points=n,
        raw_std=raw_std,
        filtered_std=filt_std,
        noise_reduction_ratio=noise_reduction,
        mean_absolute_error=mae,
        max_absolute_error=max_ae,
        lag_estimate=lag,
        signal_distortion=distortion,
    )


def _estimate_lag(raw: List[float], filtered: List[float], max_lag: int = 10) -> int:
    """Estima el retardo del filtro por correlación cruzada simplificada.

    Un filtro con lag k produce ``filtered[i] ≈ raw[i - k]``, es decir
    la señal filtrada está *retrasada* k muestras respecto a la cruda.

    Para detectar esto, buscamos el k que maximiza la correlación entre
    ``raw[:n-k]`` y ``filtered[k:]``  (alinear filtered adelantándolo k).

    Args:
        raw: Señal original.
        filtered: Señal filtrada.
        max_lag: Máximo retardo a evaluar.

    Returns:
        Retardo estimado en muestras.
    """
    n = len(raw)
    if n < 4:
        return 0

    max_lag = min(max_lag, n // 4)
    best_lag = 0
    best_corr = -math.inf

    for k in range(max_lag + 1):
        seg_len = n - k
        if seg_len < 2:
            break

        r_seg = raw[:seg_len]
        f_seg = filtered[k:]

        r_mean = sum(r_seg) / seg_len
        f_mean = sum(f_seg) / seg_len

        num = sum((r - r_mean) * (f - f_mean) for r, f in zip(r_seg, f_seg))
        den_r = sum((r - r_mean) ** 2 for r in r_seg)
        den_f = sum((f - f_mean) ** 2 for f in f_seg)
        den = math.sqrt(den_r * den_f)

        if den > 1e-12:
            corr = num / den
        else:
            corr = 0.0

        if corr > best_corr:
            best_corr = corr
            best_lag = k

    return best_lag
