"""Motor de acumulación secuencial de evidencia para MoE asimétrico.

SRP: Integra las evidencias emitidas por los expertos afines a lo largo del tiempo.
Utiliza Sequential Testing / Anytime Martingale (SPRT con fuga de CUSUM)
para evitar acumulación espuria en regímenes nominales y garantizar detección rápida.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    RepresentationLevel,
)


@dataclass(frozen=True)
class IntegratedEvidence:
    """Evidencia agregada e inmutable tras la evaluación secuencial del MoE."""

    step_index: int
    active_level: RepresentationLevel
    accumulated_evidence: float  # S_t >= 0.0 (Log-likelihood acumulado)
    martingale_value: float  # M_t = exp(S_t) (Factor de Anytime Martingale)
    is_anomaly: bool  # S_t >= alarm_threshold
    alarm_level: str  # "NOMINAL", "SUSPECT", "CRITICAL"
    dominant_expert: str
    mean_probability: float
    contributing_scores: tuple[EvidenceScore, ...]
    metadata: dict[str, Any] | None = None


class EvidenceAccumulator:
    """Acumulador secuencial de evidencia con fuga temporal y control de martingala."""

    def __init__(
        self,
        prior_nominal_prob: float = 0.05,
        leak_rate: float = 0.08,
        warn_threshold: float = 2.302,  # ln(1 / 0.10)
        alarm_threshold: float = 4.605,  # ln(1 / 0.01) (Cota de Ville a nivel alpha=0.01)
        aggregation_mode: str = "max",  # "max" (conservador para seguridad) o "mean"
        max_accumulator: float = 25.0,
        reset_on_alarm: bool = False,
    ) -> None:
        self.prior_nominal_prob = prior_nominal_prob
        self.leak_rate = leak_rate
        self.warn_threshold = warn_threshold
        self.alarm_threshold = alarm_threshold
        self.aggregation_mode = aggregation_mode
        self.max_accumulator = max_accumulator
        self.reset_on_alarm = reset_on_alarm

        # Prior log-odds
        p0 = max(1e-6, min(1.0 - 1e-6, prior_nominal_prob))
        self._prior_log_odds = math.log(p0 / (1.0 - p0))

        # Estado mutable del acumulador secuencial
        self._s_t: float = 0.0
        self._last_integrated: IntegratedEvidence | None = None

    @property
    def current_evidence(self) -> float:
        """Valor actual acumulado de la evidencia S_t."""
        return self._s_t

    def reset(self) -> None:
        """Reinicia el acumulador a cero (estado nominal)."""
        self._s_t = 0.0
        self._last_integrated = None

    def accumulate(
        self,
        scores: Sequence[EvidenceScore],
        active_level: RepresentationLevel,
        step_index: int,
    ) -> IntegratedEvidence:
        """Acumula la evidencia de los expertos activos en el paso actual.

        Args:
            scores: Lista de EvidenceScore emitidos por los expertos afines.
            active_level: Nivel de resolución temporal activo.
            step_index: Índice temporal o secuencial del bloque/punto.

        Returns:
            IntegratedEvidence con el veredicto consolidado.
        """
        if not scores:
            # Sin evaluaciones en este paso: aplicar fuga temporal
            self._s_t = max(0.0, self._s_t - self.leak_rate)
            martingale_val = math.exp(min(self._s_t, 50.0))
            verdict = IntegratedEvidence(
                step_index=step_index,
                active_level=active_level,
                accumulated_evidence=round(self._s_t, 4),
                martingale_value=round(martingale_val, 4),
                is_anomaly=self._s_t >= self.alarm_threshold,
                alarm_level=self._resolve_alarm_level(self._s_t),
                dominant_expert="none",
                mean_probability=0.0,
                contributing_scores=(),
                metadata={"reason": "no_scores_decayed"},
            )
            self._last_integrated = verdict
            return verdict

        # Calcular log-likelihood ratio para cada experto:
        # Delta L_i = log(p_i / (1 - p_i)) - log(p_0 / (1 - p_0))
        delta_ls: list[float] = []
        probabilities: list[float] = []
        dominant_expert = scores[0].expert_name
        max_prob = -1.0

        for sc in scores:
            p = max(1e-6, min(1.0 - 1e-6, sc.anomaly_probability))
            probabilities.append(p)
            log_odds = math.log(p / (1.0 - p))
            llr = log_odds - self._prior_log_odds
            delta_ls.append(llr)

            if p > max_prob:
                max_prob = p
                dominant_expert = sc.expert_name

        if self.aggregation_mode == "mean":
            step_delta = sum(delta_ls) / len(delta_ls)
        else:  # "max" por defecto (máxima sensibilidad ante cualquier experto afín)
            step_delta = max(delta_ls)

        # Actualizar S_t con fuga: S_t = max(0, S_{t-1} + step_delta - leak_rate)
        new_s_t = self._s_t + step_delta - self.leak_rate
        new_s_t = max(0.0, min(self.max_accumulator, new_s_t))

        is_alarm = new_s_t >= self.alarm_threshold
        alarm_level = self._resolve_alarm_level(new_s_t)
        martingale_val = math.exp(min(new_s_t, 50.0))

        self._s_t = new_s_t

        verdict = IntegratedEvidence(
            step_index=step_index,
            active_level=active_level,
            accumulated_evidence=round(self._s_t, 4),
            martingale_value=round(martingale_val, 4),
            is_anomaly=is_alarm,
            alarm_level=alarm_level,
            dominant_expert=dominant_expert,
            mean_probability=round(sum(probabilities) / len(probabilities), 4),
            contributing_scores=tuple(scores),
            metadata={
                "step_delta": round(step_delta, 4),
                "max_probability": round(max_prob, 4),
            },
        )

        if is_alarm and self.reset_on_alarm:
            self._s_t = 0.0

        self._last_integrated = verdict
        return verdict

    def _resolve_alarm_level(self, s: float) -> str:
        if s >= self.alarm_threshold:
            return "CRITICAL"
        if s >= self.warn_threshold:
            return "SUSPECT"
        return "NOMINAL"
