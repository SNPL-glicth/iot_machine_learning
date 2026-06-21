"""
AnomalyThresholds dataclass for regime-specific anomaly thresholds.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnomalyThresholds:
    """Thresholds of anomaly specific per regime."""
    
    # Multiplicadores de threshold base
    normal_multiplier: float = 1.0
    peak_load_multiplier: float = 0.7  # Menos sensible en peak
    transition_multiplier: float = 1.2  # Más sensible en transiciones
    startup_multiplier: float = 0.8  # Menos sensible en startup
    
    # Thresholds absolutos
    anomaly_threshold: float = 0.75  # Voting threshold
    
    # Z-score thresholds por régimen
    z_score_threshold_normal: float = 2.5
    z_score_threshold_peak: float = 3.5
    z_score_threshold_startup: float = 3.0
    z_score_threshold_anomalous: float = 2.0
    
    @classmethod
    def default(cls) -> 'AnomalyThresholds':
        """Create default configuration."""
        return cls()
