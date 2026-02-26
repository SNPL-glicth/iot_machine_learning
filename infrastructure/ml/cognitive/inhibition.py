"""Neural-inspired inhibition gate for engine weight suppression.

Analogous to lateral inhibition in the brain: if an engine shows
signs of unreliability (high instability, high fit error, excessive
recent prediction error), its weight is suppressed toward zero.

The suppression is *temporary* — once the engine's diagnostics
improve, its weight recovers.  This prevents a single noisy engine
from corrupting the fused prediction.

Suppression rules (applied multiplicatively):
    1. Instability suppression: stability > threshold → suppress
    2. Fit error suppression: local_fit_error > threshold → suppress
    3. Recent error suppression: mean |error| > threshold → suppress

Pure logic — no I/O, no persistence.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .analysis.types import EnginePerception, InhibitionState


@dataclass
class InhibitionConfig:
    """Thresholds for inhibition decisions.

    Attributes:
        stability_threshold: Stability above this triggers suppression.
        fit_error_threshold: Local fit error above this triggers suppression.
        recent_error_threshold: Mean recent |error| above this triggers suppression.
        min_weight: Floor — never suppress below this.
        max_suppression: Maximum suppression factor (0.95 = reduce to 5%).
        suppression_smoothing: EMA alpha for smoothing suppression (0.3 = moderate smoothing).
    """

    stability_threshold: float = 0.6
    fit_error_threshold: float = 5.0
    recent_error_threshold: float = 10.0
    min_weight: float = 0.02
    max_suppression: float = 0.95
    suppression_smoothing: float = 0.3


class InhibitionGate:
    """Computes inhibited weights for each engine.

    Stateless per call — receives perceptions and recent errors,
    returns inhibition states.
    """

    def __init__(self, config: Optional[InhibitionConfig] = None, max_entries: int = 10000) -> None:
        self._cfg = config or InhibitionConfig()
        self._prev_suppression: OrderedDict[Tuple[str, str], float] = OrderedDict()
        self._last_update: Dict[Tuple[str, str], float] = {}
        self._max_entries = max_entries

    def compute(
        self,
        perceptions: List[EnginePerception],
        base_weights: Dict[str, float],
        recent_errors: Optional[Dict[str, List[float]]] = None,
        series_id: Optional[str] = None,
    ) -> List[InhibitionState]:
        """Apply inhibition rules to each engine's weight.

        Args:
            perceptions: Each engine's perception from current step.
            base_weights: Pre-inhibition weights per engine name.
            recent_errors: Optional history of |prediction - actual|.

        Returns:
            List of InhibitionState, one per perception.
        """
        errors = recent_errors or {}
        states: List[InhibitionState] = []
        now = time.monotonic()
        sid = series_id if series_id else "_default"

        for p in perceptions:
            bw = base_weights.get(p.engine_name, 0.0)
            instant_suppression = 0.0
            reason = "none"

            # Rule 1: instability
            if p.stability > self._cfg.stability_threshold:
                s = min(
                    (p.stability - self._cfg.stability_threshold)
                    / (1.0 - self._cfg.stability_threshold + 1e-9),
                    self._cfg.max_suppression,
                )
                if s > instant_suppression:
                    instant_suppression = s
                    reason = f"instability={p.stability:.3f}"

            # Rule 2: local fit error
            if p.local_fit_error > self._cfg.fit_error_threshold:
                s = min(
                    (p.local_fit_error - self._cfg.fit_error_threshold)
                    / (self._cfg.fit_error_threshold + 1e-9),
                    self._cfg.max_suppression,
                )
                if s > instant_suppression:
                    instant_suppression = s
                    reason = f"fit_error={p.local_fit_error:.3f}"

            # Rule 3: recent prediction errors
            eng_errors = errors.get(p.engine_name, [])
            if eng_errors:
                mean_err = sum(eng_errors) / len(eng_errors)
                if mean_err > self._cfg.recent_error_threshold:
                    s = min(
                        (mean_err - self._cfg.recent_error_threshold)
                        / (self._cfg.recent_error_threshold + 1e-9),
                        self._cfg.max_suppression,
                    )
                    if s > instant_suppression:
                        instant_suppression = s
                        reason = f"recent_error={mean_err:.3f}"

            # Apply exponential decay to previous suppression
            key = (sid, p.engine_name)
            if key in self._prev_suppression:
                elapsed = now - self._last_update.get(key, now)
                decay_rate = 0.1
                decay_factor = math.exp(-decay_rate * elapsed)
                prev = self._prev_suppression[key] * decay_factor
            else:
                prev = 0.0
            
            # Apply EMA smoothing to suppression
            smoothed_suppression = (
                (1.0 - self._cfg.suppression_smoothing) * prev +
                self._cfg.suppression_smoothing * instant_suppression
            )
            
            # LRU eviction
            if len(self._prev_suppression) >= self._max_entries:
                oldest = min(self._last_update, key=self._last_update.get)
                self._prev_suppression.pop(oldest, None)
                self._last_update.pop(oldest, None)
            
            self._prev_suppression[key] = smoothed_suppression
            self._last_update[key] = now
            if key in self._prev_suppression:
                self._prev_suppression.move_to_end(key)

            inhibited = max(
                self._cfg.min_weight,
                bw * (1.0 - smoothed_suppression),
            )

            states.append(InhibitionState(
                engine_name=p.engine_name,
                base_weight=bw,
                inhibited_weight=inhibited,
                inhibition_reason=reason,
                suppression_factor=smoothed_suppression,
            ))

        return states
