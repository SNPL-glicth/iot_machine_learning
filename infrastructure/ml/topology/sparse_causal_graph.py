"""Grafo causal disperso para topología multivariada de alta velocidad en streaming.

SRP: Gestiona aristas dirigidas, dependencias con retardo y búsqueda de ancestros/causa raíz
con complejidad O(|E|), garantizando consultas sub-milisegundo.
"""

from __future__ import annotations

from typing import Sequence

from iot_machine_learning.domain.entities.topology import CausalEdge, CausalRelationType
from iot_machine_learning.domain.ports.topology_port import CausalTopologyPort


class SparseCausalGraph(CausalTopologyPort):
    """Estructura de grafo dirigido disperso para relaciones de precedencia y retardo temporal."""

    def __init__(self, coupling_threshold: float = 0.30) -> None:
        self.coupling_threshold = coupling_threshold
        # Adyacencia saliente: source -> dict[target, CausalEdge]
        self._forward_adj: dict[str, dict[str, CausalEdge]] = {}
        # Adyacencia entrante: target -> dict[source, CausalEdge]
        self._backward_adj: dict[str, dict[str, CausalEdge]] = {}

    def add_edge(self, edge: CausalEdge) -> None:
        """Registra o actualiza una arista causal dirigida."""
        src, dst = edge.source_series_id, edge.target_series_id
        if src not in self._forward_adj:
            self._forward_adj[src] = {}
        self._forward_adj[src][dst] = edge

        if dst not in self._backward_adj:
            self._backward_adj[dst] = {}
        self._backward_adj[dst][src] = edge

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Elimina una arista causal."""
        if source_id in self._forward_adj:
            self._forward_adj[source_id].pop(target_id, None)
        if target_id in self._backward_adj:
            self._backward_adj[target_id].pop(source_id, None)

    def register_pair_observation(
        self,
        source_id: str,
        target_id: str,
        source_value: float,
        target_value: float,
        step: int,
    ) -> None:
        """Implementa el método de CausalTopologyPort (en grafo estático/manual actúa como no-op)."""
        pass

    def get_active_edges(self) -> Sequence[CausalEdge]:
        """Retorna todas las aristas dirigidas que superan el umbral de acoplamiento."""
        edges: list[CausalEdge] = []
        for targets in self._forward_adj.values():
            for edge in targets.values():
                if edge.coupling_strength >= self.coupling_threshold:
                    edges.append(edge)
        return edges

    def get_downstream_edges(self, node_id: str) -> Sequence[CausalEdge]:
        """Obtiene las aristas dirigidas salientes desde node_id hacia sus efectos."""
        return [
            e for e in self._forward_adj.get(node_id, {}).values()
            if e.coupling_strength >= self.coupling_threshold
        ]

    def get_upstream_edges(self, node_id: str) -> Sequence[CausalEdge]:
        """Obtiene las aristas dirigidas entrantes hacia node_id desde sus causas potenciales."""
        return [
            e for e in self._backward_adj.get(node_id, {}).values()
            if e.coupling_strength >= self.coupling_threshold
        ]

    def is_ancestor(
        self,
        ancestor_id: str,
        target_id: str,
        max_hops: int = 3,
    ) -> tuple[bool, int, float]:
        """Comprueba si ancestor_id precede a target_id, calculando retardo acumulado y fuerza mínima."""
        if ancestor_id == target_id:
            return True, 0, 1.0

        visited: set[str] = set()
        # Cola BFS: (current_node, cumulative_lag, min_strength, hops)
        queue: list[tuple[str, int, float, int]] = [(ancestor_id, 0, 1.0, 0)]

        while queue:
            curr, cum_lag, min_s, hops = queue.pop(0)
            if curr == target_id:
                return True, cum_lag, min_s
            if hops >= max_hops or curr in visited:
                continue

            visited.add(curr)
            for edge in self.get_downstream_edges(curr):
                dst = edge.target_series_id
                if dst not in visited:
                    queue.append((
                        dst,
                        cum_lag + edge.lag_steps,
                        min(min_s, edge.coupling_strength),
                        hops + 1,
                    ))

        return False, 0, 0.0
