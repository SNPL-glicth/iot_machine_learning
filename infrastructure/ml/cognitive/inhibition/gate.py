"""Inhibition gate for engine weight suppression.

IMP-4b: the authoritative decision is taken from an
:class:`EngineReliabilityTracker` Beta-Bernoulli posterior when one is
injected. ``is_reliable(engine, series) == False`` → the engine's
inhibited weight is driven to ``0.0`` (hard exclusion, not min_weight).

The legacy path (three hardcoded thresholds on stability, fit error and
recent error) is kept **only for backward compatibility** with callers
that do not yet inject a reliability tracker. It will be deleted in a
subsequent cleanup once all call sites are wired.

FASE-21: Interacción Inhibition vs Bayesian Weight Tracking
------------------------------------------------------------
Inhibition opera DESPUÉS de Bayesian weight tracking en el pipeline.
Inhibition puede suprimir engines que Bayesian está promoviendo.

Resolución de conflictos:
- **Inhibition tiene precedencia** cuando detecta señales hard:
  * stability > threshold (señal inestable)
  * fit_error > threshold (error de ajuste alto)
  * recent_error > threshold (errores recientes altos)
  → Supresión inmediata con exponential decay

- **Bayesian tiene precedencia** en ajuste gradual (señales soft):
  * Actualización incremental de pesos basada en historial
  * EMA con alpha=0.15 (convergencia suave)
  * No hay oscilación porque ambos sistemas son convergentes

La combinación es estable porque:
1. Inhibition usa exponential decay (no oscila)
2. Bayesian usa EMA con alpha bajo (convergencia lenta)
3. Inhibition actúa como circuit breaker para casos extremos
4. Bayesian ajusta finamente en condiciones normales

Ver también: `infrastructure/ml/cognitive/bayesian_weight_tracker/base.py`

Pure logic — no I/O, no persistence.
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.parameters.numerical_constants import EPSILON, INHIBITION_THRESHOLDS
from ..analysis.types import EnginePerception, InhibitionState
from ..reliability import EngineReliabilityTracker

logger = logging.getLogger(__name__)


@dataclass
class InhibitionConfig:
    """Thresholds for inhibition decisions.

    Attributes:
        stability_threshold: Stability above this triggers suppression.
        fit_error_threshold: Local fit error above this triggers suppression.
            FASE-21: SCALE-DEPENDENT threshold (MSE units).
            Válido para sensores con valores en rango típico [0, 100].
            Para otros rangos, ajustar proporcionalmente o usar use_relative_error.
        recent_error_threshold: Mean recent |error| above this triggers suppression.
            FASE-21: SCALE-DEPENDENT threshold (MSE units).
            Válido para sensores con valores en rango típico [0, 100].
            Para otros rangos, ajustar proporcionalmente o usar use_relative_error.
        min_weight: Floor — never suppress below this.
            FASE-21: min_weight=0.02 para inhibition (post-supresión).
            Bayesian usa min_weight=0.05 (tracking histórico).
            Ver: infrastructure/ml/cognitive/bayesian_weight_tracker/bayesian_weight_config.py
        max_suppression: Maximum suppression factor (0.95 = reduce to 5%).
        suppression_smoothing: EMA alpha for smoothing suppression (0.3 = moderate smoothing).
            FASE-22: Inhibition usa alpha más alto (0.3) que Bayesian (0.15) porque
            necesita respuesta rápida a degradación de engines. Bayesian usa alpha
            bajo (0.15) porque ajusta gradualmente confianza histórica basada en
            accuracy. Diferencia intencional: Inhibition = circuit breaker rápido,
            Bayesian = aprendizaje gradual.
        anomaly_z_score_threshold: Z-score above which signal is considered anomalous (CRIT-2).
        anomaly_override_enabled: If True, ignore error-based inhibition when signal is anomalous.
        use_relative_error: If True, use relative_error = fit_error / signal_mean
            for scale-invariant thresholds. NOT IMPLEMENTED YET (reserved for future).
    """

    stability_threshold: float = INHIBITION_THRESHOLDS.STABILITY
    fit_error_threshold: float = INHIBITION_THRESHOLDS.FIT_ERROR
    recent_error_threshold: float = INHIBITION_THRESHOLDS.RECENT_ERROR
    min_weight: float = 0.02
    max_suppression: float = 0.95
    suppression_half_life_seconds: float = 300.0  # FASE-22: See ML_INHIBITION_SUPPRESSION_HALF_LIFE_S
    # 5 minutos = balance entre respuesta rápida y estabilidad.
    # Para datos de alta frecuencia (>1Hz) considerar reducir a 60s.
    suppression_smoothing: float = 0.3
    anomaly_z_score_threshold: float = 3.0  # CRIT-2: 3-sigma rule
    anomaly_override_enabled: bool = True   # CRIT-2: Enable by default
    use_relative_error: bool = False  # FASE-21: Reserved for future scale-invariant implementation


class InhibitionGate:
    """Computes inhibited weights for each engine.

    Stateless per call — receives perceptions and recent errors,
    returns inhibition states.
    """

    def __init__(
        self,
        config: Optional[InhibitionConfig] = None,
        max_entries: int = 10000,
        reliability_tracker: Optional[EngineReliabilityTracker] = None,
        *,
        entry_ttl_seconds: float = 3600.0,
    ) -> None:
        if entry_ttl_seconds < 0:
            raise ValueError(f"entry_ttl_seconds must be >= 0, got {entry_ttl_seconds}")
        self._cfg = config or InhibitionConfig()
        self._prev_suppression: OrderedDict[Tuple[str, str], float] = OrderedDict()
        self._last_update: Dict[Tuple[str, str], float] = {}
        self._max_entries = max_entries
        self._reliability = reliability_tracker
        self._entry_ttl_seconds = entry_ttl_seconds

    def compute(
        self,
        perceptions: List[EnginePerception],
        base_weights: Dict[str, float],
        recent_errors: Optional[Dict[str, List[float]]] = None,
        series_id: Optional[str] = None,
        signal_z_score: float = 0.0,
    ) -> List[InhibitionState]:
        """Apply inhibition rules to each engine's weight.

        IMP-4b: when a reliability tracker is injected it is the sole
        authority. Engines for which ``is_reliable`` returns ``False``
        are driven to ``inhibited_weight = 0.0``. Otherwise the base
        weight is passed through unchanged.

        Args:
            perceptions: Each engine's perception from current step.
            base_weights: Pre-inhibition weights per engine name.
            recent_errors: Optional history of |prediction - actual| per engine.
            series_id: Series identifier for logging (CRIT-1).
            signal_z_score: Z-score of current signal (CRIT-2: anomaly override).

        Returns:
            List of InhibitionState, one per perception.
        """
        if self._reliability is not None:
            return self._compute_via_reliability(
                perceptions, base_weights, series_id or "_default"
            )

        errors = recent_errors or {}
        states: List[InhibitionState] = []
        now = time.monotonic()
        sid = series_id if series_id else "_default"
        
        # CRIT-2: Check if signal is anomalous (anomaly override)
        is_anomalous_signal = abs(signal_z_score) > self._cfg.anomaly_z_score_threshold
        if is_anomalous_signal and self._cfg.anomaly_override_enabled:
            logger.info(
                "inhibition_anomaly_override_active",
                extra={
                    "series_id": sid,
                    "signal_z_score": round(signal_z_score, 3),
                    "threshold": self._cfg.anomaly_z_score_threshold,
                    "reason": "high_prediction_error_expected_during_anomaly",
                },
            )

        for p in perceptions:
            bw = base_weights.get(p.engine_name, 0.0)
            instant_suppression = 0.0
            reason = "none"

            # Rule 1: instability
            if p.stability > self._cfg.stability_threshold:
                s = min(
                    (p.stability - self._cfg.stability_threshold)
                    / (1.0 - self._cfg.stability_threshold + EPSILON.DIVISION),
                    self._cfg.max_suppression,
                )
                if s > instant_suppression:
                    instant_suppression = s
                    reason = f"instability={p.stability:.3f}"

            # Rule 2: local fit error
            if p.local_fit_error > self._cfg.fit_error_threshold:
                s = min(
                    (p.local_fit_error - self._cfg.fit_error_threshold)
                    / (self._cfg.fit_error_threshold + EPSILON.DIVISION),
                    self._cfg.max_suppression,
                )
                if s > instant_suppression:
                    instant_suppression = s
                    reason = f"fit_error={p.local_fit_error:.3f}"

            # Rule 3: recent prediction errors (CRIT-2: skip if anomalous signal)
            eng_errors = errors.get(p.engine_name, [])
            if eng_errors and not (is_anomalous_signal and self._cfg.anomaly_override_enabled):
                mean_err = sum(eng_errors) / len(eng_errors)
                if mean_err > self._cfg.recent_error_threshold:
                    s = min(
                        (mean_err - self._cfg.recent_error_threshold)
                        / (self._cfg.recent_error_threshold + EPSILON.DIVISION),
                        self._cfg.max_suppression,
                    )
                    if s > instant_suppression:
                        instant_suppression = s
                        reason = f"recent_error={mean_err:.3f}"
            elif is_anomalous_signal and self._cfg.anomaly_override_enabled and eng_errors:
                # CRIT-2: Log that we're ignoring errors due to anomaly override
                logger.debug(
                    "inhibition_error_override_applied",
                    extra={
                        "series_id": sid,
                        "engine_name": p.engine_name,
                        "mean_error": round(sum(eng_errors) / len(eng_errors), 3),
                        "signal_z_score": round(signal_z_score, 3),
                    },
                )

            # Apply exponential decay to previous suppression (MATH-SEV-4)
            key = (sid, p.engine_name)
            if key in self._prev_suppression and not self._is_expired(key, now):
                elapsed = now - self._last_update.get(key, now)
                prev = self._apply_decay(self._prev_suppression[key], elapsed)
            else:
                prev = 0.0
            
            # Apply EMA smoothing to suppression
            smoothed_suppression = (
                (1.0 - self._cfg.suppression_smoothing) * prev +
                self._cfg.suppression_smoothing * instant_suppression
            )
            
            # Lazy purge of expired entries on write
            self.purge_expired(now)

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

    def _apply_decay(
        self,
        previous_suppression: float,
        elapsed_seconds: float,
    ) -> float:
        """Apply exponential decay to suppression (MATH-SEV-4).
        
        Uses half-life formula: decay_rate = ln(2) / half_life
        
        Args:
            previous_suppression: Previous suppression value.
            elapsed_seconds: Time elapsed since last update.
        
        Returns:
            Decayed suppression value.
        
        Applies SRP: Decay logic is separate from inhibition decision.
        """
        decay_rate = math.log(2) / self._cfg.suppression_half_life_seconds
        decay_factor = math.exp(-decay_rate * elapsed_seconds)
        return previous_suppression * decay_factor
    
    def _is_expired(
        self, key: Tuple[str, str], now: Optional[float] = None
    ) -> bool:
        """Check if a suppression entry has exceeded its TTL."""
        if self._entry_ttl_seconds <= 0:
            return False
        if key not in self._last_update:
            return True
        t = now if now is not None else time.monotonic()
        return (t - self._last_update[key]) > self._entry_ttl_seconds

    def purge_expired(self, now: Optional[float] = None) -> int:
        """Remove all entries whose TTL has expired. Returns count removed."""
        if self._entry_ttl_seconds <= 0:
            return 0
        t = now if now is not None else time.monotonic()
        expired = [
            key for key in self._last_update
            if (t - self._last_update[key]) > self._entry_ttl_seconds
        ]
        for key in expired:
            self._prev_suppression.pop(key, None)
            self._last_update.pop(key, None)
        return len(expired)

    def _compute_via_reliability(
        self,
        perceptions: List[EnginePerception],
        base_weights: Dict[str, float],
        series_id: str,
    ) -> List[InhibitionState]:
        """IMP-4b path: Beta-Bernoulli reliability drives inhibition.

        A reliable engine passes through at its base weight with
        ``suppression_factor=0``. An unreliable engine is hard-excluded
        with ``inhibited_weight=0.0`` and ``suppression_factor=1.0``.
        """
        states: List[InhibitionState] = []
        for p in perceptions:
            bw = base_weights.get(p.engine_name, 0.0)
            reliable = self._reliability.is_reliable(series_id, p.engine_name)
            if reliable:
                states.append(InhibitionState(
                    engine_name=p.engine_name,
                    base_weight=bw,
                    inhibited_weight=bw,
                    inhibition_reason="none",
                    suppression_factor=0.0,
                ))
            else:
                p_broken = self._reliability.p_broken(series_id, p.engine_name)
                states.append(InhibitionState(
                    engine_name=p.engine_name,
                    base_weight=bw,
                    inhibited_weight=0.0,
                    inhibition_reason=f"unreliable p_broken={p_broken:.3f}",
                    suppression_factor=1.0,
                ))
        return states
