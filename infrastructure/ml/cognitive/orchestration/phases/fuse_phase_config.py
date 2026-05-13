"""FusePhase Configuration (Clean Architecture Refactor).

Consolidates all magic numbers and configuration parameters for FusePhase.

Applies SRP: Configuration is separate from phase logic.
Applies DIP: FusePhase depends on this config abstraction, not on env vars.
Applies OCP: Adding new parameters only requires extending this dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass


# Student's t-distribution critical values for two-tailed test at p=0.05
# df = n - 2 (degrees of freedom for correlation test)
# Source: Standard t-distribution tables
T_CRITICAL_TABLE = {
    20: 2.086,   # df=18
    25: 2.060,   # df=23
    30: 2.042,   # df=28
    40: 2.021,   # df=38
    60: 2.000,   # df=58
    120: 1.980,  # df=118
    # For df >= 120, use 1.960 (approaches normal distribution z=1.96)
}


@dataclass(frozen=True)
class FusePhaseConfig:
    """Configuration for FusePhase spatial correction and fusion.
    
    All parameters have documented justifications and calibration guidance.
    
    Attributes:
        max_correction_pct: Maximum spatial correction as % of base prediction.
            Justification: Conservative 15% limit prevents overcorrection from
            spurious spatial correlations. Industrial sensors may require 0.05-0.10
            (tighter tolerance). Environmental sensors may allow 0.20-0.25.
            Valid range: [0.0, 0.5]. Calibrate by domain volatility.
        
        min_gradient_samples: Minimum samples required for OLS gradient.
            Justification: Statistical minimum for reliable linear regression.
            3 samples was insufficient (COG-SEV-2 audit). 5 is recommended minimum.
            Valid range: [3, 10]. Increase for noisy domains.
        
        min_correlation: Minimum Pearson correlation to consider neighbor relevant.
            Justification: 0.7 threshold ensures strong correlation. Applied
            consistently in both spatial correction AND field smoothing (single
            source of truth). 0.5 was too permissive (false spatial relationships).
            Valid range: [0.5, 0.95]. Increase for critical systems (0.8-0.9).
        
        smoothing_factor: Field smoothing interpolation factor.
            Justification: 0.2 = 80% local prediction, 20% neighbor influence.
            Conservative approach prioritizes local model over spatial field.
            0.0 = no smoothing, 1.0 = full neighbor average.
            Valid range: [0.0, 0.5]. Increase for spatially homogeneous fields.
        
        field_smoothing_min_neighbors: Minimum neighbors for field smoothing.
            Justification: Requires at least 2 series (self + 1 neighbor) to
            apply spatial smoothing. Prevents smoothing with insufficient context.
            Valid range: [2, 5]. Increase for dense sensor networks.
        
        min_samples_for_significance: Minimum samples for correlation t-test.
            Justification: 20 samples provides adequate statistical power for
            correlation significance test. Below 20, correlations are unreliable.
            Valid range: [10, 50]. Increase for high-stakes decisions (30-50).
        
        p_value_threshold: P-value threshold for correlation significance.
            Justification: Standard 0.05 threshold (95% confidence). Correlations
            with p > 0.05 are considered statistically insignificant and rejected.
            Valid range: [0.01, 0.10]. Use 0.01 for critical systems.
        
        hampel_k: MAD multiplier for Hampel outlier detection.
            Justification: 3.0 is standard Hampel identifier (≈ 3σ under normality).
            Detects values > median ± 3×MAD as outliers.
            Valid range: [2.0, 5.0]. Increase for noisy data (4.0-5.0).
        
        hampel_enabled: Enable/disable Hampel outlier filter.
            Justification: Kill switch for Hampel filter. Disable for debugging
            or when all engines are trusted. Enabled by default for robustness.
        
        spatial_bias_multiplier: Multiplier for spatial trend bias correction.
            Justification: Factor empírico pendiente calibración. Controla cuánto
            influyen sensores vecinos en fused_value vía bias_factor × signal_std.
            0.1 = 10% de std como máximo ajuste por tendencia espacial.
            Valid range: [0.0, 0.5]. Increase for spatially homogeneous fields (0.2-0.3).
    """
    
    # Spatial correction parameters
    max_correction_pct: float = 0.15
    min_gradient_samples: int = 5
    min_correlation: float = 0.7
    spatial_bias_multiplier: float = 0.1  # FASE-22: Spatial trend bias factor
    
    # Field smoothing parameters
    smoothing_factor: float = 0.2
    field_smoothing_min_neighbors: int = 2
    
    # Correlation quality validation
    min_samples_for_significance: int = 20
    p_value_threshold: float = 0.05
    
    # Hampel filter parameters
    hampel_k: float = 3.0
    hampel_enabled: bool = True
    
    def get_t_critical(self, df: int) -> float:
        """Get critical t-value for correlation significance test.
        
        Args:
            df: Degrees of freedom (n_samples - 2).
        
        Returns:
            Critical t-value for two-tailed test at p=0.05.
        
        Uses Student's t-distribution table for accurate critical values.
        For df >= 120, returns 1.960 (normal approximation).
        """
        if df < 18:  # Below min_samples_for_significance - 2
            return 2.2  # Conservative fallback
        
        # Find smallest threshold >= df
        for threshold, t_crit in sorted(T_CRITICAL_TABLE.items()):
            if df <= threshold:
                return t_crit
        
        # df > 120, use normal approximation
        return 1.960
