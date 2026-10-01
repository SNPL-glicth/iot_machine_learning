"""Puerta de decisión multiobjetivo certificada por cota de Ville y sensible al presupuesto.

SRP: Integra e-values ponderados, mantiene el proceso de martingala M_t
y evalúa el stopping time contra un umbral dinámico tau_t ajustado por régimen y latencia/budget.
Cumple el contrato AdaptiveMetaGatePort del dominio.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from iot_machine_learning.domain.entities.conformal_risk import (
    AdaptiveGateDecision,
    ConformalBound,
    RiskCertificationStatus,
)
from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    SystemOperationalState,
)
from iot_machine_learning.domain.ports.meta_gate_port import (
    AdaptiveMetaGatePort,
    OnlineCalibratorPort,
)


class LatencyBudgetAwareGate(AdaptiveMetaGatePort):
    """Puerta adaptativa de decisión multiobjetivo con garantías de cota de Ville."""

    def __init__(
        self,
        calibrator: OnlineCalibratorPort,
        alpha_target: float = 0.01,  # Ville cota base = 1 / 0.01 = 100.0
        budget_penalty_weight: float = 1.5,
        shrinkage_lambda: float = 0.02,  # Parámetro de mezcla para estabilidad
        state_multipliers: Mapping[SystemOperationalState, float] | None = None,
        max_martingale: float = 1e6,
        reset_on_trigger: bool = False,
    ) -> None:
        self.calibrator = calibrator
        self.alpha_target = alpha_target
        self.base_threshold = 1.0 / max(1e-6, alpha_target)
        self.budget_penalty_weight = budget_penalty_weight
        self.shrinkage_lambda = shrinkage_lambda
        self.max_martingale = max_martingale
        self.reset_on_trigger = reset_on_trigger

        self.state_multipliers = dict(state_multipliers or {
            SystemOperationalState.RESTING: 1.5,   # Más conservador para blindar reposo
            SystemOperationalState.DRIFTING: 1.0,  # Cota canónica de Ville
            SystemOperationalState.SHOCKED: 0.6,   # Mayor reactividad instantánea
        })

        # Estado mutable de la martingala M_t (M_0 = 1.0)
        self._m_t: float = 1.0
        self._log_m_t: float = 0.0
        self._last_decision: AdaptiveGateDecision | None = None
        self._current_tau: float = self.base_threshold

    @property
    def current_martingale(self) -> float:
        """Valor actual de la martingala M_t."""
        return self._m_t

    @property
    def bounds(self) -> ConformalBound:
        """Retorna las cotas conformales activas."""
        return ConformalBound(
            alpha_target=self.alpha_target,
            base_threshold=self.base_threshold,
            current_dynamic_threshold=self._current_tau,
        )

    def reset_martingale(self) -> None:
        """Reinicia la martingala a su estado canónico M_0 = 1.0."""
        self._m_t = 1.0
        self._log_m_t = 0.0
        self._last_decision = None

    def evaluate_step(
        self,
        step: int,
        evidences: Sequence[EvidenceScore],
        operational_state: SystemOperationalState,
        budget_remaining_ratio: float = 1.0,
    ) -> AdaptiveGateDecision:
        """Calcula el e-value combinado, actualiza la martingala M_t y resuelve el stopping time."""
        # Factor por estrés de presupuesto computacional / latencia
        # Si budget_remaining_ratio < 1.0, incrementa el umbral para desincentivar falsas alarmas costosas
        clamped_budget = max(0.0, min(1.0, budget_remaining_ratio))
        budget_penalty = 1.0 + self.budget_penalty_weight * (1.0 - clamped_budget)

        # Multiplicador por estado operativo
        state_factor = self.state_multipliers.get(operational_state, 1.0)

        # Umbral dinámico multiobjetivo tau_t
        tau_t = self.base_threshold * budget_penalty * state_factor
        self._current_tau = tau_t

        if not evidences:
            # Sin evidencias activas en este ciclo: decaimiento leve hacia 1.0
            self._log_m_t = max(0.0, self._log_m_t - self.shrinkage_lambda)
            self._m_t = math.exp(self._log_m_t)
            cert = RiskCertificationStatus.NOMINAL
            decision = AdaptiveGateDecision(
                step=step,
                operational_state=operational_state,
                martingale_value=round(self._m_t, 4),
                dynamic_threshold=round(tau_t, 4),
                certification=cert,
                is_triggered=False,
                active_expert_weights=self.calibrator.current_weights,
                budget_penalty_factor=round(budget_penalty, 4),
                reason="no_active_evidence_decay",
                metadata={"budget_remaining": clamped_budget},
            )
            self._last_decision = decision
            return decision

        # 1. Transformar evidencias a e-values mediante el calibrador online
        e_values = self.calibrator.compute_e_values(evidences)

        # 2. Obtener pesos convexos w_{i, t}
        weights = self.calibrator.current_weights

        # 3. Combinación convexa de e-values: E_t = sum_i w_i * E_{i, t}
        active_names = [ev.expert_name for ev in evidences]
        sub_weights = [weights.get(name, 1.0 / len(active_names)) for name in active_names]
        tot_sub = sum(sub_weights) or 1.0
        norm_sub_weights = [w / tot_sub for w in sub_weights]

        combined_e = sum(w * e for w, e in zip(norm_sub_weights, e_values, strict=False))

        # 4. Actualizar proceso log-martingale acotado inferiormente por H0 (M_t >= 1.0):
        # s_t = max(0.0, s_{t-1} + ln(E_t) - shrinkage_lambda)
        log_e = math.log(max(1e-4, combined_e))
        new_log_m_t = max(
            0.0,
            min(math.log(self.max_martingale), self._log_m_t + log_e - self.shrinkage_lambda)
        )
        self._log_m_t = new_log_m_t
        new_m_t = math.exp(new_log_m_t)
        self._m_t = new_m_t

        # 5. Actualizar los pesos del calibrador con la evidencia y estado observados
        self.calibrator.update_weights(evidences, e_values, operational_state)

        # 6. Evaluar condición de stopping time certificado bajo Ville
        is_triggered = new_m_t >= tau_t

        if is_triggered:
            certification = RiskCertificationStatus.CERTIFIED_ALARM
            reason = f"ville_threshold_exceeded_m_{round(new_m_t, 2)}_ge_tau_{round(tau_t, 2)}"
        elif new_m_t >= 0.5 * tau_t:
            certification = RiskCertificationStatus.BORDERLINE
            reason = "warning_evidence_accumulating"
        else:
            certification = RiskCertificationStatus.NOMINAL
            reason = "within_null_hypothesis_envelope"

        decision = AdaptiveGateDecision(
            step=step,
            operational_state=operational_state,
            martingale_value=round(new_m_t, 4),
            dynamic_threshold=round(tau_t, 4),
            certification=certification,
            is_triggered=is_triggered,
            active_expert_weights=self.calibrator.current_weights,
            budget_penalty_factor=round(budget_penalty, 4),
            reason=reason,
            metadata={
                "combined_e_value": round(combined_e, 4),
                "budget_remaining": round(clamped_budget, 4),
            },
        )

        if is_triggered and self.reset_on_trigger:
            self.reset_martingale()

        self._last_decision = decision
        return decision
