"""Calibrador no paramétrico de conformidad para el régimen nominal.

Estima las funciones de distribución empíricas F_x y F_dx durante el periodo
de warmup sin asumir normalidad gaussiana, colas ligeras ni simetría.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EmpiricalDistributionProfile:
    """Perfil no paramétrico del régimen nominal construido en warmup."""

    sample_size: int
    median: float
    q_low: float  # e.g. p = alpha/2
    q_high: float  # e.g. p = 1 - alpha/2
    q_shock_high: float  # e.g. p = 1 - alpha_shock para |dx/dt|
    quantiles_raw: dict[float, float]
    interquartile_range: float
    support_min: float
    support_max: float


class NonParametricConformalCalibrator:
    """Construye perfiles empíricos sin asumir normalidad gaussiana.

    Funciona exactamente igual sobre temperaturas, precios, retornos o volatilidad.
    No calcula (x - mean)/std como regla de corte rígida.
    """

    def __init__(self, alpha_regime: float = 0.02, alpha_shock: float = 0.01) -> None:
        self.alpha_regime = alpha_regime
        self.alpha_shock = alpha_shock

    def fit_warmup(
        self, values: np.ndarray, block_size: int = 10
    ) -> tuple[EmpiricalDistributionProfile, EmpiricalDistributionProfile]:
        """Aprende los perfiles nominales F_x(x) y F_dx(|dx/dt|)."""
        n = len(values)
        if n < 50:
            raise ValueError(f"Warmup insuficiente: {n} puntos.")

        # 1. Perfil de nivel y medias de bloque F_block_mean
        n_blocks = n // block_size
        if n_blocks > 1:
            reshaped = values[: n_blocks * block_size].reshape(n_blocks, block_size)
            block_means = np.mean(reshaped, axis=1)
        else:
            block_means = values

        q_levels = [
            self.alpha_regime / 2.0,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            1.0 - (self.alpha_regime / 2.0),
        ]
        q_vals = np.quantile(block_means, q_levels)
        q_dict = {round(p, 4): float(v) for p, v in zip(q_levels, q_vals, strict=False)}

        iqr = float(q_dict[0.75] - q_dict[0.25])
        profile_level = EmpiricalDistributionProfile(
            sample_size=len(block_means),
            median=float(q_dict[0.50]),
            q_low=float(q_dict[round(self.alpha_regime / 2.0, 4)]),
            q_high=float(q_dict[round(1.0 - (self.alpha_regime / 2.0), 4)]),
            q_shock_high=0.0,
            quantiles_raw=q_dict,
            interquartile_range=max(iqr, 1e-9),
            support_min=float(np.min(block_means)),
            support_max=float(np.max(block_means)),
        )

        # 2. Perfil de velocidad/choque temporal F_dx(|dx/dt|)
        diffs = np.abs(np.diff(values))
        q_shock = float(np.quantile(diffs, 1.0 - self.alpha_shock))
        profile_shock = EmpiricalDistributionProfile(
            sample_size=len(diffs),
            median=float(np.median(diffs)),
            q_low=0.0,
            q_high=q_shock,
            q_shock_high=q_shock,
            quantiles_raw={round(1.0 - self.alpha_shock, 4): q_shock},
            interquartile_range=float(np.percentile(diffs, 75) - np.percentile(diffs, 25)),
            support_min=float(np.min(diffs)),
            support_max=float(np.max(diffs)),
        )

        return profile_level, profile_shock
