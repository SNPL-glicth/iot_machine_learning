"""Regime weight loading from external config."""

from __future__ import annotations

from typing import Dict


def load_regime_weights() -> Dict[str, Dict[str, float]]:
    """Load weights from external config; fallback to defaults."""
    try:
        from ..config.moe_config import REGIME_WEIGHTS
        return REGIME_WEIGHTS
    except Exception:
        pass
    return {
        "stable": {"baseline": 0.80, "statistical": 0.15, "taylor": 0.05, "kalman": 0.00},
        "trending": {"baseline": 0.05, "statistical": 0.55, "taylor": 0.35, "kalman": 0.05},
        "volatile": {"baseline": 0.05, "statistical": 0.25, "taylor": 0.50, "kalman": 0.20},
        "noisy": {"baseline": 0.10, "statistical": 0.20, "taylor": 0.20, "kalman": 0.50},
    }


# NOTE: Equipment-specific weights should be loaded from config file, not hardcoded.
# EQUIPMENT_REGIME_WEIGHTS moved to external config (JSON/YAML).
def load_equipment_weights() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load equipment-specific weights from external config."""
    try:
        from ..config.moe_config import EQUIPMENT_REGIME_WEIGHTS
        return EQUIPMENT_REGIME_WEIGHTS
    except Exception:
        return {}