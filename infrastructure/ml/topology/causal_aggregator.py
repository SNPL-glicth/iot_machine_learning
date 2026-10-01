"""Orquestador de supresión de alertas en cascada y atribución de causa raíz (RCA).

SRP: Ingiere decisiones de compuerta adaptativa multivariada, correlaciona retardos
mediante el grafo causal y colapsa cascadas de alarmas en un SystemWideAlarm consolidado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from iot_machine_learning.domain.entities.conformal_risk import AdaptiveGateDecision
from iot_machine_learning.domain.entities.representation_evidence import SystemOperationalState
from iot_machine_learning.domain.entities.topology import (
    CausalEdge,
    RootCauseDiagnosis,
    SystemWideAlarm,
)
from iot_machine_learning.domain.ports.topology_port import (
    CausalGatingOrchestratorPort,
    CausalTopologyPort,
)
from .sparse_causal_graph import SparseCausalGraph


@dataclass
class _ActiveIncident:
    """Estado interno mutable de un incidente sistémico en curso."""

    alarm_id: str
    root_id: str
    root_decision: AdaptiveGateDecision
    start_step: int
    last_trigger_step: int
    affected_series: set[str] = field(default_factory=set)
    suppressed_cascade_count: int = 0
    peak_martingale: float = 0.0
    total_compute_saved: float = 0.0


class CausalGatingAggregator(CausalGatingOrchestratorPort):
    """Agregador topológico y compuerta causal para supresión de cascadas y diagnóstico RCA."""

    def __init__(
        self,
        causal_graph: SparseCausalGraph | CausalTopologyPort,
        cooldown_steps: int = 50,
        lag_tolerance: int = 15,
        nominal_expert_cost: float = 1.0,
    ) -> None:
        if isinstance(causal_graph, SparseCausalGraph):
            self._graph: SparseCausalGraph = causal_graph
        elif hasattr(causal_graph, "graph") and isinstance(getattr(causal_graph, "graph"), SparseCausalGraph):
            self._graph = getattr(causal_graph, "graph")
        else:
            # Fallback a grafo disperso vacío
            self._graph = SparseCausalGraph()

        self.cooldown_steps = cooldown_steps
        self.lag_tolerance = lag_tolerance
        self.nominal_expert_cost = nominal_expert_cost

        # Incidentes activos indexados por alarm_id
        self._active_incidents: dict[str, _ActiveIncident] = {}

        # Métricas y telemetría
        self.total_raw_alerts_received: int = 0
        self.total_cascade_alerts_suppressed: int = 0
        self.total_system_alarms_emitted: int = 0

    @property
    def cascade_suppression_rate(self) -> float:
        """Tasa porcentual de alarmas secundarias en cascada suprimidas [0.0, 1.0]."""
        if self.total_raw_alerts_received == 0:
            return 0.0
        return float(self.total_cascade_alerts_suppressed / self.total_raw_alerts_received)

    @property
    def active_incident_count(self) -> int:
        """Número de incidentes activos actualmente en seguimiento."""
        return len(self._active_incidents)

    def _is_ancestor_of(self, ancestor_id: str, target_id: str) -> tuple[bool, int, float]:
        """Consulta el grafo causal para determinar si ancestor_id precede a target_id."""
        return self._graph.is_ancestor(ancestor_id, target_id)

    def process_decisions(
        self,
        step: int,
        node_decisions: Mapping[str, AdaptiveGateDecision],
    ) -> Sequence[SystemWideAlarm]:
        """Unifica alertas individuales, suprime efectos retardados y emite el RCA consolidado."""
        # 1. Identificar nodos que dispararon alerta en el paso actual
        triggered: dict[str, AdaptiveGateDecision] = {
            nid: dec for nid, dec in node_decisions.items() if dec.is_triggered
        }
        self.total_raw_alerts_received += len(triggered)

        # 2. Expirar incidentes inactivos más allá del período de enfriamiento
        expired_ids = [
            aid for aid, inc in self._active_incidents.items()
            if (step - inc.last_trigger_step) > self.cooldown_steps
        ]
        for aid in expired_ids:
            del self._active_incidents[aid]

        if not triggered:
            return []

        handled_nodes: set[str] = set()
        updated_incidents: set[str] = set()

        # 3. Intentar asociar alertas a incidentes ya activos
        for nid, dec in triggered.items():
            for inc in self._active_incidents.values():
                # Caso A: Re-disparo del nodo raíz
                if nid == inc.root_id:
                    inc.last_trigger_step = step
                    inc.peak_martingale = max(inc.peak_martingale, dec.martingale_value)
                    handled_nodes.add(nid)
                    updated_incidents.add(inc.alarm_id)
                    break

                # Caso B: Cascada hacia nodo dependiente
                is_anc, cum_lag, _ = self._is_ancestor_of(inc.root_id, nid)
                if is_anc:
                    earliest_arrival = inc.start_step + cum_lag - self.lag_tolerance
                    if step >= earliest_arrival:
                        # Supresión exitosa de la alarma aguas abajo
                        inc.affected_series.add(nid)
                        inc.suppressed_cascade_count += 1
                        inc.last_trigger_step = step
                        inc.peak_martingale = max(inc.peak_martingale, dec.martingale_value)
                        inc.total_compute_saved += self.nominal_expert_cost
                        self.total_cascade_alerts_suppressed += 1
                        handled_nodes.add(nid)
                        updated_incidents.add(inc.alarm_id)
                        break

        # 4. Procesar nodos disparados no explicados por incidentes previos
        unhandled_nodes = [nid for nid in triggered if nid not in handled_nodes]
        if unhandled_nodes:
            # Identificar raíces topológicas entre los nodos disparados no explicados
            root_candidates: list[str] = []
            for u in unhandled_nodes:
                has_parent_in_unhandled = False
                for v in unhandled_nodes:
                    if u != v:
                        is_anc, _, _ = self._is_ancestor_of(v, u)
                        if is_anc:
                            has_parent_in_unhandled = True
                            break
                if not has_parent_in_unhandled:
                    root_candidates.append(u)

            # Si existiera ciclo o ambigüedad, fallback a todos los unhandled
            if not root_candidates:
                root_candidates = list(unhandled_nodes)

            for root_id in root_candidates:
                dec_root = triggered[root_id]
                alarm_id = f"alarm_{root_id}_{step}"
                new_inc = _ActiveIncident(
                    alarm_id=alarm_id,
                    root_id=root_id,
                    root_decision=dec_root,
                    start_step=step,
                    last_trigger_step=step,
                    affected_series={root_id},
                    suppressed_cascade_count=0,
                    peak_martingale=dec_root.martingale_value,
                    total_compute_saved=0.0,
                )
                self._active_incidents[alarm_id] = new_inc
                updated_incidents.add(alarm_id)
                handled_nodes.add(root_id)

                # Suprimir cualquier otro nodo no manejado que sea descendiente de este nuevo root
                for other_nid in unhandled_nodes:
                    if other_nid not in handled_nodes:
                        is_anc, _, _ = self._is_ancestor_of(root_id, other_nid)
                        if is_anc:
                            dec_other = triggered[other_nid]
                            new_inc.affected_series.add(other_nid)
                            new_inc.suppressed_cascade_count += 1
                            new_inc.peak_martingale = max(new_inc.peak_martingale, dec_other.martingale_value)
                            new_inc.total_compute_saved += self.nominal_expert_cost
                            self.total_cascade_alerts_suppressed += 1
                            handled_nodes.add(other_nid)

        # 5. Sintetizar SystemWideAlarm para cada incidente con actividad en este paso
        emitted_alarms: list[SystemWideAlarm] = []
        for aid in sorted(updated_incidents):
            inc = self._active_incidents[aid]

            # Atribución de mecanismo dominante del experto con mayor peso
            weights = inc.root_decision.active_expert_weights
            if weights:
                dominant_mech = max(weights.keys(), key=lambda k: weights[k])
                confidence = float(weights[dominant_mech])
            else:
                dominant_mech = "unknown_mechanism"
                confidence = 0.0

            diagnosis = RootCauseDiagnosis(
                root_series_id=inc.root_id,
                dominant_mechanism=dominant_mech,
                mechanism_confidence=confidence,
                detection_step=inc.start_step,
                operational_state=inc.root_decision.operational_state,
                active_expert_weights=dict(weights),
            )

            affected_tuple = tuple(sorted(inc.affected_series))
            summary = (
                f"Root cause '{inc.root_id}' triggered at step {inc.start_step} "
                f"via '{dominant_mech}' (confidence {confidence:.2f}). "
                f"Cascades suppressed: {inc.suppressed_cascade_count} downstream alerts "
                f"across nodes {affected_tuple}."
            )

            alarm = SystemWideAlarm(
                alarm_id=inc.alarm_id,
                root_cause=diagnosis,
                affected_series_ids=affected_tuple,
                suppressed_cascade_count=inc.suppressed_cascade_count,
                peak_martingale_value=float(inc.peak_martingale),
                total_compute_saved_by_suppression=float(inc.total_compute_saved),
                explanation_summary=summary,
            )
            emitted_alarms.append(alarm)
            self.total_system_alarms_emitted += 1

        return emitted_alarms
