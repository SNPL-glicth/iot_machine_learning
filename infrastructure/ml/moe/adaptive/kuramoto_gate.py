"""Puerta de decisión por sincronización topológica de fase (Adler-Kuramoto).

Complejidad computacional: O(N) por paso. Latencia media: < 2 microsegundos.
Cumple el contrato ConsensusGatePort / AdaptiveMetaGatePort del dominio.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from iot_machine_learning.domain.entities.consensus import ConsensusDecision, KuramotoGateConfig
from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    RepresentationLevel,
    SystemOperationalState,
)
from iot_machine_learning.domain.ports.meta_gate_port import ConsensusGatePort

_LEVEL_DOWNSAMPLE: dict[RepresentationLevel, int] = {
    RepresentationLevel.TEN_X: 10,
    RepresentationLevel.TWO_X: 2,
    RepresentationLevel.RAW: 1,
}


class KuramotoConsensusGate(ConsensusGatePort):
    """Puerta de consenso no lineal por campo medio con forzamiento Adler."""

    def __init__(
        self,
        expert_names: Sequence[str],
        config: KuramotoGateConfig | None = None,
        expert_levels: Mapping[str, RepresentationLevel] | None = None,
    ) -> None:
        self.config, self.expert_names = config or KuramotoGateConfig(), list(expert_names)
        self.n = max(1, len(self.expert_names))
        self.omegas: dict[str, float] = {}
        for name in self.expert_names:
            if self.config.expert_frequencies and name in self.config.expert_frequencies:
                self.omegas[name] = self.config.expert_frequencies[name]
            elif expert_levels and name in expert_levels:
                self.omegas[name] = 2.0 * math.pi / max(1, _LEVEL_DOWNSAMPLE.get(expert_levels[name], 1))
            else:
                scale = 10 if ("10x" in name or "inv" in name) else (2 if ("2x" in name or "drift" in name) else 1)
                self.omegas[name] = 2.0 * math.pi / scale

        mean_w = sum(self.omegas.values()) / self.n
        self.k_critical = 2.0 * max(1.0, math.sqrt(sum((w - mean_w) ** 2 for w in self.omegas.values()) / self.n))
        self.state_thresholds: dict[SystemOperationalState, float] = dict(
            self.config.base_state_thresholds
            or {SystemOperationalState.RESTING: 0.85, SystemOperationalState.DRIFTING: 0.40, SystemOperationalState.SHOCKED: 0.40}
        )
        self._thetas = {name: (2.0 * math.pi * i) / self.n for i, name in enumerate(self.expert_names)}
        self._prev_r: float = 0.0
        self._refractory_countdown: int = 0
        self._last_decision: ConsensusDecision | None = None

    def calibrate_from_warmup(self, nominal_orders: Sequence[float]) -> None:
        """Calibra el umbral de reposo automáticamente según el orden en warmup."""
        if not nominal_orders:
            return
        mean_r = sum(nominal_orders) / len(nominal_orders)
        std_r = math.sqrt(sum((x - mean_r) ** 2 for x in nominal_orders) / len(nominal_orders))
        self.state_thresholds[SystemOperationalState.RESTING] = min(0.95, max(nominal_orders) + 3.0 * std_r)

    def reset(self) -> None:
        """Reinicia las fases a dispersión uniforme y desactiva el quenching."""
        for i, name in enumerate(self.expert_names):
            self._thetas[name] = (2.0 * math.pi * i) / self.n
        self._prev_r, self._refractory_countdown, self._last_decision = 0.0, 0, None

    def _compute_mean_field(self) -> tuple[float, float]:
        """Calcula r in [0, 1] y psi in [-pi, pi] en O(N)."""
        sum_c = sum(math.cos(self._thetas[k]) for k in self.expert_names)
        sum_s = sum(math.sin(self._thetas[k]) for k in self.expert_names)
        return min(1.0, math.sqrt(sum_c * sum_c + sum_s * sum_s) / self.n), math.atan2(sum_s, sum_c)

    def _scramble_phases(self) -> tuple[float, float]:
        """Topological Quenching: dispersa fases a simetría antipodal determinista (r=0)."""
        for i, name in enumerate(self.expert_names):
            self._thetas[name] = (2.0 * math.pi * i) / self.n
        return 0.0, 0.0

    def evaluate_step(
        self,
        step: int,
        evidences: Sequence[EvidenceScore],
        operational_state: SystemOperationalState,
        budget_remaining_ratio: float = 1.0,
    ) -> ConsensusDecision:
        """Integra evidencias, avanza Euler discreto y resuelve la decisión de fase."""
        if self._refractory_countdown > 0:
            self._refractory_countdown -= 1
            real_r, psi = self._scramble_phases()
            self._prev_r = real_r
            return ConsensusDecision(
                step=step, operational_state=operational_state, order_parameter=round(real_r, 4),
                phase_velocity=0.0, dynamic_threshold=self.state_thresholds.get(operational_state, 0.80),
                is_triggered=False, reason="refractory_quenching_cooldown",
                active_expert_weights={name: 1.0 / self.n for name in self.expert_names},
                metadata={"global_phase": round(psi, 4)},
            )

        ev_map = {ev.expert_name: max(0.0, min(1.0, ev.anomaly_probability)) for ev in evidences}
        active_p = (sum(ev_map.values()) / len(ev_map)) if ev_map else 0.0
        dt = self.config.delta_t
        r, psi = self._compute_mean_field()
        k_eff = (0.25 + 1.50 * active_p) * self.k_critical
        forcing_gain = self.config.forcing_gain_ratio * self.k_critical

        for name in self.expert_names:
            th = self._thetas[name]
            coupling = k_eff * r * math.sin(psi - th)
            forcing = forcing_gain * ev_map.get(name, 0.0) * math.sin(self.config.alarm_phase - th)
            self._thetas[name] = (th + dt * (self.omegas[name] + coupling + forcing)) % (2.0 * math.pi)

        new_r, new_psi = self._compute_mean_field()
        r_vel = (new_r - self._prev_r) / dt
        self._prev_r = new_r

        base_rc = self.state_thresholds.get(operational_state, 0.80)
        budget_factor = 1.0 + self.config.budget_penalty_weight * (1.0 - max(0.0, min(1.0, budget_remaining_ratio)))
        r_c = min(0.98, base_rc * budget_factor)
        is_burst = (operational_state == SystemOperationalState.SHOCKED and r_vel >= self.config.burst_velocity_threshold and new_r >= self.config.min_shock_threshold)
        is_trig = (active_p >= self.config.evidence_floor) and ((new_r >= r_c) or is_burst)

        reason = (
            f"order_collapse_r_{round(new_r, 3)}_ge_rc_{round(r_c, 3)}" if (new_r >= r_c and is_trig)
            else (f"phase_burst_velocity_{round(r_vel, 2)}_ge_{self.config.burst_velocity_threshold}" if is_trig else "phase_desynchronized_nominal")
        )
        expert_weights = {name: round(max(0.0, math.cos(self._thetas[name] - new_psi)), 4) for name in self.expert_names}
        decision = ConsensusDecision(
            step=step, operational_state=operational_state, order_parameter=round(new_r, 4),
            phase_velocity=round(r_vel, 4), dynamic_threshold=round(r_c, 4), is_triggered=is_trig,
            reason=reason, active_expert_weights=expert_weights,
            metadata={"global_phase": round(new_psi, 4), "budget_factor": round(budget_factor, 3)},
        )
        if is_trig:
            self._refractory_countdown = self.config.refractory_steps
        self._last_decision = decision
        return decision
