"""Data structures for the Taylor prediction engine.

Pure value objects — no I/O, no state, no logic beyond serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class DerivativeMethod(Enum):
    """Available derivative estimation methods.

    - BACKWARD: First-order backward differences, O(Δt).
      Requires only past data. Amplifies noise.
    - CENTRAL: Second-order central differences, O(Δt²).
      More accurate but uses the point *before* the last as
      expansion center (needs one future-adjacent point).
    - LEAST_SQUARES: Local polynomial least-squares fit.
      Most noise-resistant. Requires >= order+1 points.
      Fits a polynomial of degree ``order`` to the last
      ``2*order + 1`` points (or all available) and reads
      derivatives from the fitted coefficients.
    """

    BACKWARD = "backward"
    CENTRAL = "central"
    LEAST_SQUARES = "least_squares"


@dataclass(frozen=True)
class TaylorCoefficients:
    """Coefficients of the local Taylor polynomial at the expansion point.

    Scaled Taylor coefficients:
        c_0 = f(t)
        c_1 = f'(t)
        c_2 = f''(t) / 2!
        c_3 = f'''(t) / 3!

    The raw (unscaled) derivatives are stored for diagnostics.

    Attributes:
        f_t: Function value at the expansion point.
        f_prime: First derivative (slope / velocity).
        f_double_prime: Second derivative (curvature / acceleration).
        f_triple_prime: Third derivative (jerk).
        estimated_order: Effective order computed (may be < requested).
        method: Derivative estimation method used.
    """

    f_t: float
    f_prime: float = 0.0
    f_double_prime: float = 0.0
    f_triple_prime: float = 0.0
    estimated_order: int = 0
    method: str = "backward"

    @property
    def coefficients(self) -> List[float]:
        """Scaled Taylor coefficients [c_0, c_1, c_2, c_3].

        Polynomial: P(h) = c_0 + c_1·h + c_2·h² + c_3·h³
        """
        return [
            self.f_t,
            self.f_prime,
            self.f_double_prime / 2.0,
            self.f_triple_prime / 6.0,
        ]

    @property
    def local_slope(self) -> float:
        """First derivative — instantaneous rate of change."""
        return self.f_prime

    @property
    def curvature(self) -> float:
        """Second derivative — acceleration / concavity."""
        return self.f_double_prime

    def to_dict(self) -> dict:
        """Serialize for metadata / logging."""
        return {
            "f_t": self.f_t,
            "f_prime": self.f_prime,
            "f_double_prime": self.f_double_prime,
            "f_triple_prime": self.f_triple_prime,
        }


@dataclass(frozen=True)
class TaylorDiagnostic:
    """Diagnostic output for a Taylor prediction.

    Attributes:
        estimated_order: Effective Taylor order used.
        coefficients: Scaled polynomial coefficients [c_0 .. c_3].
        local_slope: f'(t).
        curvature: f''(t).
        stability_indicator: 0.0 (stable) to 1.0 (unstable).
        accel_variance: Raw variance of f'' across the window.
        local_fit_error: RMS residual of the polynomial fit (0 if N/A).
        dt: Time step used.
        method: Derivative estimation method name.
    """

    estimated_order: int
    coefficients: List[float]
    local_slope: float
    curvature: float
    stability_indicator: float
    accel_variance: float
    local_fit_error: float
    dt: float
    method: str = "backward"

    def to_dict(self) -> dict:
        """Serialize for metadata / logging."""
        return {
            "estimated_order": self.estimated_order,
            "coefficients": [round(c, 8) for c in self.coefficients],
            "local_slope": round(self.local_slope, 8),
            "curvature": round(self.curvature, 8),
            "stability_indicator": round(self.stability_indicator, 6),
            "accel_variance": round(self.accel_variance, 8),
            "local_fit_error": round(self.local_fit_error, 8),
            "dt": self.dt,
            "method": self.method,
        }
