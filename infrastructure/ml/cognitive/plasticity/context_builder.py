"""PlasticityContext builder — creates context from signal profile.

Pure logic for mapping signal profiles to plasticity contexts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from iot_machine_learning.domain.entities.plasticity.plasticity_context import (
    PlasticityContext,
    RegimeType,
)


def build_plasticity_context(
    profile,
    series_id: str,
    consecutive_failures: int = 0,
    error_magnitude: float = 0.0,
    is_critical_zone: bool = False,
) -> PlasticityContext:
    """Create PlasticityContext from signal profile.
    
    Args:
        profile: Signal profile from analyzer (StructuralAnalysis or similar)
        series_id: Series identifier
        consecutive_failures: Number of consecutive failures
        error_magnitude: Current error magnitude
        is_critical_zone: Whether in critical operational zone
    
    Returns:
        PlasticityContext with regime, volatility, noise, time
    """
    # Map regime string to RegimeType enum
    regime_str = profile.regime.value if hasattr(profile, 'regime') else 'unknown'
    regime_map = {
        "stable": RegimeType.STABLE,
        "trending": RegimeType.TRENDING,
        "volatile": RegimeType.VOLATILE,
        "noisy": RegimeType.NOISY,
    }
    regime_type = regime_map.get(regime_str.lower(), RegimeType.UNKNOWN)
    
    # Get volatility from profile
    volatility = getattr(profile, 'volatility', 0.0)
    if hasattr(profile, 'volatility_level'):
        vol_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
        volatility = vol_map.get(profile.volatility_level.value.lower(), 0.5)
    
    # Estimate noise ratio from signal quality
    noise_ratio = 1.0 - getattr(profile, 'signal_quality', 0.5)
    
    # Create context
    now = datetime.now()
    return PlasticityContext(
        regime=regime_type,
        noise_ratio=min(max(noise_ratio, 0.0), 1.0),
        volatility=min(max(volatility, 0.0), 1.0),
        time_of_day=now.hour,
        consecutive_failures=consecutive_failures,
        error_magnitude=error_magnitude,
        is_critical_zone=is_critical_zone,
        timestamp=now,
    )
