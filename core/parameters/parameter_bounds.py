"""Enforcement de bounds en parámetros adaptativos.

Previene parameter explosion y colapso.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BoundsConfig:
    """Configuración de bounds para un parámetro."""
    min_value: float
    max_value: float
    soft_min: Optional[float] = None
    soft_max: Optional[float] = None
    clip_strategy: str = "hard"


@dataclass
class BoundsResult:
    """Resultado del enforcement de bounds."""
    original_value: float
    clipped_value: float
    was_clipped: bool
    clip_reason: str


class ParameterBoundsEnforcer:
    """
    Enforcea bounds en parámetros adaptativos.

    Bounds definidos en audit para parámetros críticos:
    - learning_rate (alpha): [0.01, 0.5]
    - contamination: [0.001, 0.1]
    - drift_decay: [0.1, 0.9]
    - confidence_temperature: [0.5, 3.0]

    Uso:
        enforcer = ParameterBoundsEnforcer()
        enforcer.register_bounds("ML_BAYES_ALPHA", BoundsConfig(0.01, 0.5))
        result = enforcer.enforce("ML_BAYES_ALPHA", new_value)
    """

    DEFAULT_BOUNDS = {
        "ML_BAYES_ALPHA": BoundsConfig(0.01, 0.5, soft_min=0.05, soft_max=0.4),
        "contamination": BoundsConfig(0.001, 0.1),
        "drift_decay_factor": BoundsConfig(0.1, 0.9),
        "ML_CONFIDENCE_TEMPERATURE": BoundsConfig(0.5, 3.0),
        "ema_smoothing_alpha": BoundsConfig(0.01, 0.99),
    }

    def __init__(self, use_defaults: bool = True) -> None:
        self._bounds: Dict[str, BoundsConfig] = {}
        if use_defaults:
            self._bounds.update(self.DEFAULT_BOUNDS)

    def register_bounds(self, name: str, config: BoundsConfig) -> None:
        """Registra bounds para un parámetro."""
        self._bounds[name] = config

    def enforce(self, name: str, value: float) -> BoundsResult:
        """Enforcea bounds y retorna resultado con logging."""
        if name not in self._bounds:
            return BoundsResult(
                original_value=value,
                clipped_value=value,
                was_clipped=False,
                clip_reason="no_bounds_registered",
            )

        config = self._bounds[name]
        clipped_value = value
        was_clipped = False
        clip_reason = "ok"

        # Check hard bounds
        if value < config.min_value:
            clipped_value = config.min_value
            was_clipped = True
            clip_reason = "below_min"
            logger.warning(
                "parameter_clipped_below_min",
                extra={
                    "parameter": name,
                    "original": round(value, 6),
                    "clipped_to": config.min_value,
                },
            )
        elif value > config.max_value:
            clipped_value = config.max_value
            was_clipped = True
            clip_reason = "above_max"
            logger.warning(
                "parameter_clipped_above_max",
                extra={
                    "parameter": name,
                    "original": round(value, 6),
                    "clipped_to": config.max_value,
                },
            )

        # Check soft bounds (warning only)
        if not was_clipped:
            if config.soft_min is not None and value < config.soft_min:
                clip_reason = "soft_warning_below"
                logger.info(
                    "parameter_below_soft_min",
                    extra={
                        "parameter": name,
                        "value": round(value, 6),
                        "soft_min": config.soft_min,
                    },
                )
            elif config.soft_max is not None and value > config.soft_max:
                clip_reason = "soft_warning_above"
                logger.info(
                    "parameter_above_soft_max",
                    extra={
                        "parameter": name,
                        "value": round(value, 6),
                        "soft_max": config.soft_max,
                    },
                )

        return BoundsResult(
            original_value=value,
            clipped_value=clipped_value,
            was_clipped=was_clipped,
            clip_reason=clip_reason,
        )

    def enforce_all(self, params: Dict[str, float]) -> Dict[str, BoundsResult]:
        """Enforcea bounds en múltiples parámetros."""
        results = {}
        for name, value in params.items():
            results[name] = self.enforce(name, value)
        return results
