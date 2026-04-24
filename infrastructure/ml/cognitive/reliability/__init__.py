"""Reliability package \u2014 Beta-Bernoulli engine reliability estimation.

Public API::

    from iot_machine_learning.infrastructure.ml.cognitive.reliability import (
        EngineReliabilityTracker,
    )

Replaces the three hardcoded rules previously baked into
``InhibitionGate`` (instability / fit-error / recent-error thresholds)
with a single Beta-Bernoulli posterior per ``(series_id, engine_name)``.
"""

from __future__ import annotations

from .engine_reliability_tracker import EngineReliabilityTracker

__all__ = ["EngineReliabilityTracker"]
