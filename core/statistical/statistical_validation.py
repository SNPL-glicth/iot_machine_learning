"""Validación de supuestos estadísticos.

Principio: Single Responsibility - solo valida distribuciones.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats

from core.parameters.numerical_constants import EPSILON

# statsmodels is optional for stationarity tests
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class DistributionType(Enum):
    NORMAL = "normal"
    NON_NORMAL_SYMMETRIC = "non_normal_symmetric"
    SKEWED = "skewed"
    HEAVY_TAILED = "heavy_tailed"
    UNKNOWN = "unknown"


class StationarityType(Enum):
    STATIONARY = "stationary"
    TREND_STATIONARY = "trend_stationary"
    NON_STATIONARY = "non_stationary"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class NormalityTestResult:
    """Resultado de test de normalidad."""
    is_normal: bool
    distribution_type: DistributionType
    shapiro_stat: float
    shapiro_p_value: float
    anderson_stat: float
    anderson_critical_value: float
    skewness: float
    kurtosis: float
    recommendation: str  # "use_z_score", "use_mad", "use_iqr", etc.


@dataclass
class StationarityTestResult:
    """Resultado de test de estacionariedad."""
    is_stationary: bool
    stationarity_type: StationarityType
    adf_stat: float
    adf_p_value: float
    kpss_stat: float
    kpss_critical_value: float
    recommendation: str  # "use_taylor", "detrend_first", "use_simple_average", etc.


class NormalityValidator:
    """Valida normalidad de distribución."""

    def __init__(
        self,
        shapiro_alpha: float = 0.05,
        anderson_level: float = 5.0,  # 5% significance
        min_samples: int = 20,
    ) -> None:
        self.shapiro_alpha = shapiro_alpha
        self.anderson_level = anderson_level
        self.min_samples = min_samples

    def validate(self, data: np.ndarray) -> NormalityTestResult:
        """
        Valida normalidad usando Shapiro-Wilk y Anderson-Darling.

        Shapiro-Wilk: H0 = data is normal
        Anderson-Darling: H0 = data is normal
        """
        if len(data) < self.min_samples:
            return NormalityTestResult(
                is_normal=False,
                distribution_type=DistributionType.UNKNOWN,
                shapiro_stat=0.0,
                shapiro_p_value=0.0,
                anderson_stat=0.0,
                anderson_critical_value=0.0,
                skewness=0.0,
                kurtosis=0.0,
                recommendation="insufficient_data",
            )

        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(data)

        # Anderson-Darling test
        anderson_result = stats.anderson(data, dist="norm")
        anderson_stat = anderson_result.statistic
        # Critical values for [15%, 10%, 5%, 2.5%, 1%]
        anderson_critical = anderson_result.critical_values[2]  # 5% level

        # Compute skewness and kurtosis
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        # Decision logic
        shapiro_normal = shapiro_p > self.shapiro_alpha
        anderson_normal = anderson_stat < anderson_critical

        is_normal = shapiro_normal and anderson_normal

        # Classify distribution type
        if is_normal:
            dist_type = DistributionType.NORMAL
            recommendation = "use_z_score"
        elif abs(skewness) < 0.5 and abs(kurtosis) < 1.0:
            dist_type = DistributionType.NON_NORMAL_SYMMETRIC
            recommendation = "use_iqr"
        elif abs(skewness) >= 0.5:
            dist_type = DistributionType.SKEWED
            recommendation = "use_mad"  # Median Absolute Deviation
        elif abs(kurtosis) >= 3.0:
            dist_type = DistributionType.HEAVY_TAILED
            recommendation = "use_mad"
        else:
            dist_type = DistributionType.UNKNOWN
            recommendation = "use_mad"

        return NormalityTestResult(
            is_normal=is_normal,
            distribution_type=dist_type,
            shapiro_stat=shapiro_stat,
            shapiro_p_value=shapiro_p,
            anderson_stat=anderson_stat,
            anderson_critical_value=anderson_critical,
            skewness=skewness,
            kurtosis=kurtosis,
            recommendation=recommendation,
        )


class StationarityValidator:
    """Valida estacionariedad de serie temporal."""

    def __init__(
        self,
        adf_alpha: float = 0.05,
        kpss_alpha: float = 0.05,
        min_samples: int = 30,
    ) -> None:
        self.adf_alpha = adf_alpha
        self.kpss_alpha = kpss_alpha
        self.min_samples = min_samples

    def validate(self, data: np.ndarray) -> StationarityTestResult:
        """
        Valida estacionariedad usando ADF y KPSS.

        ADF: H0 = unit root (non-stationary)
        KPSS: H0 = stationary
        """
        if len(data) < self.min_samples:
            return StationarityTestResult(
                is_stationary=False,
                stationarity_type=StationarityType.INSUFFICIENT_DATA,
                adf_stat=0.0,
                adf_p_value=1.0,
                kpss_stat=0.0,
                kpss_critical_value=0.0,
                recommendation="insufficient_data",
            )

        if not STATSMODELS_AVAILABLE:
            # Fallback: assume stationary if statsmodels not available
            return StationarityTestResult(
                is_stationary=True,
                stationarity_type=StationarityType.STATIONARY,
                adf_stat=0.0,
                adf_p_value=0.0,
                kpss_stat=0.0,
                kpss_critical_value=0.0,
                recommendation="statsmodels_not_available",
            )

        # ADF test
        adf_result = adfuller(data)
        adf_stat = adf_result[0]
        adf_p = adf_result[1]

        # KPSS test
        kpss_result = kpss(data, regression="c")
        kpss_stat = kpss_result[0]
        kpss_critical = kpss_result[3]["5%"]  # 5% level

        # Decision logic
        adf_stationary = adf_p < self.adf_alpha  # Reject H0 = stationary
        kpss_stationary = kpss_stat < kpss_critical  # Accept H0 = stationary

        # Combined decision
        if adf_stationary and kpss_stationary:
            stat_type = StationarityType.STATIONARY
            recommendation = "use_taylor"
        elif adf_stationary and not kpss_stationary:
            stat_type = StationarityType.TREND_STATIONARY
            recommendation = "detrend_first"
        elif not adf_stationary:
            stat_type = StationarityType.NON_STATIONARY
            recommendation = "use_differencing"
        else:
            stat_type = StationarityType.NON_STATIONARY
            recommendation = "use_simple_average"

        return StationarityTestResult(
            is_stationary=adf_stationary and kpss_stationary,
            stationarity_type=stat_type,
            adf_stat=adf_stat,
            adf_p_value=adf_p,
            kpss_stat=kpss_stat,
            kpss_critical_value=kpss_critical,
            recommendation=recommendation,
        )
