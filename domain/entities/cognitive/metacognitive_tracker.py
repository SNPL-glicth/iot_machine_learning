"""Tracker metacognitivo para cuantificar el auto-entendimiento del sistema."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

from .post_mortem_evaluator import PostMortemRecord


@dataclass(frozen=True)
class MetacognitiveStatus:
    """Estado de auto-competencia y comprensión del régimen actual."""

    regime: str
    meta_competence_score: float  # Ω_t ∈ [0.0, 1.0]
    calibration_error: float
    correlated_failure_rate: float
    systemic_blindness_detected: bool
    suggested_lambda_penalty: float
    sample_size: int


class MetacognitiveTracker:
    """Evalúa si ZENIN comprende el entorno o si sufre de ceguera sistémica."""

    def __init__(self, window_size: int = 20, min_samples: int = 5) -> None:
        self._window_size = max(5, window_size)
        self._min_samples = min_samples
        self._regime_windows: Dict[str, deque[PostMortemRecord]] = {}

    def record_outcome(self, record: PostMortemRecord) -> None:
        """Registra el resultado madurado de una predicción."""
        if record.regime not in self._regime_windows:
            self._regime_windows[record.regime] = deque(maxlen=self._window_size)
        self._regime_windows[record.regime].append(record)

    def get_status(self, regime: str = "default") -> MetacognitiveStatus:
        """Calcula el índice de meta-competencia Ω_t para el régimen especificado."""
        window = self._regime_windows.get(regime)
        if not window or len(window) < self._min_samples:
            # Estado neutral/frío: sin datos suficientes, no penaliza agresivamente
            return MetacognitiveStatus(
                regime=regime,
                meta_competence_score=0.8,
                calibration_error=0.0,
                correlated_failure_rate=0.0,
                systemic_blindness_detected=False,
                suggested_lambda_penalty=0.0,
                sample_size=len(window) if window else 0,
            )

        n = len(window)
        correlated_failures = 0
        conf_sum = 0.0
        acc_sum = 0.0

        for rec in window:
            # Calibración: confianza esperada vs acierto real del árbitro
            conf_sum += rec.diagnostic.confidence_at_time
            if not rec.diagnostic.is_failure:
                acc_sum += 1.0

            # Ceguera correlacionada: ¿fallaron todos los expertos al unísono?
            all_experts_failed = (
                len(rec.directional_correctness) > 0
                and not any(rec.directional_correctness.values())
            )
            if all_experts_failed:
                correlated_failures += 1

        avg_conf = conf_sum / n
        avg_acc = acc_sum / n
        calibration_error = abs(avg_conf - avg_acc)
        correlated_failure_rate = correlated_failures / n

        # Detección de ceguera sistémica: mayoría de expertos fallando juntos
        systemic_blindness = correlated_failure_rate >= 0.6

        # Meta-competencia: penaliza error de calibración y fallos correlacionados
        penalty = (calibration_error * 0.5) + (correlated_failure_rate * 0.5)
        omega = max(0.0, min(1.0, 1.0 - penalty))

        if systemic_blindness:
            omega = min(omega, 0.2)

        # Sugerencia de modulación para lambda_t (freno de mano)
        suggested_lambda_penalty = 1.0 - omega

        return MetacognitiveStatus(
            regime=regime,
            meta_competence_score=omega,
            calibration_error=calibration_error,
            correlated_failure_rate=correlated_failure_rate,
            systemic_blindness_detected=systemic_blindness,
            suggested_lambda_penalty=suggested_lambda_penalty,
            sample_size=n,
        )

    def modulate_exploration_factor(self, base_lambda: float, regime: str = "default") -> float:
        """Ajusta dinámicamente lambda_t: si Ω_t colapsa, fuerza lambda_t -> 1."""
        status = self.get_status(regime)
        if status.systemic_blindness_detected:
            return 1.0
        return max(base_lambda, status.suggested_lambda_penalty)
