"""Estimador en streaming de entropía de transferencia y dependencias causales con retardo.

SRP: Ingiere flujos multivariados online, calcula la correlación cruzada direccional
y actualiza aristas dirigidas en un SparseCausalGraph con complejidad sub-milisegundo.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from iot_machine_learning.domain.entities.topology import CausalEdge, CausalRelationType
from iot_machine_learning.domain.ports.topology_port import CausalTopologyPort
from .sparse_causal_graph import SparseCausalGraph


class StreamingTransferEntropyEstimator(CausalTopologyPort):
    """Estimador online de dependencias causales y retardo temporal óptimo."""

    def __init__(
        self,
        graph: SparseCausalGraph | None = None,
        max_lag: int = 30,
        window_size: int = 200,
        coupling_threshold: float = 0.30,
        update_interval: int = 10,
        min_samples: int = 40,
    ) -> None:
        self._graph: SparseCausalGraph = graph if graph is not None else SparseCausalGraph(coupling_threshold)
        self.max_lag = max_lag
        self.window_size = window_size
        self.coupling_threshold = coupling_threshold
        self.update_interval = update_interval
        self.min_samples = min_samples

        # Buffers deslizantes para cada serie observada
        self._buffers: dict[str, deque[float]] = {}
        # Último paso registrado por nodo para evitar duplicados si un nodo participa en varios pares
        self._last_step: dict[str, int] = {}
        # Contador de observaciones por par para controlar update_interval
        self._pair_steps: dict[tuple[str, str], int] = {}
        # Registro de pares observados
        self._monitored_pairs: set[tuple[str, str]] = set()

    @property
    def graph(self) -> SparseCausalGraph:
        """Acceso al grafo causal subyacente."""
        return self._graph

    def register_pair_observation(
        self,
        source_id: str,
        target_id: str,
        source_value: float,
        target_value: float,
        step: int,
    ) -> None:
        """Registra una observación conjunta en streaming y actualiza la topología causal."""
        if self._last_step.get(source_id) != step:
            if source_id not in self._buffers:
                self._buffers[source_id] = deque(maxlen=self.window_size)
            self._buffers[source_id].append(float(source_value))
            self._last_step[source_id] = step

        if self._last_step.get(target_id) != step:
            if target_id not in self._buffers:
                self._buffers[target_id] = deque(maxlen=self.window_size)
            self._buffers[target_id].append(float(target_value))
            self._last_step[target_id] = step

        pair = (source_id, target_id)
        self._monitored_pairs.add(pair)
        self._pair_steps[pair] = self._pair_steps.get(pair, 0) + 1

        # Re-estimar dependencias periódicamente si tenemos suficientes muestras
        if (
            self._pair_steps[pair] % self.update_interval == 0
            and len(self._buffers[source_id]) >= self.min_samples
            and len(self._buffers[target_id]) >= self.min_samples
        ):
            self._estimate_causal_coupling(source_id, target_id)

    def _estimate_causal_coupling(self, src: str, dst: str) -> None:
        """Calcula el desfase temporal óptimo delta* y la asimetría direccional."""
        buf_src = np.array(self._buffers[src], dtype=np.float64)
        buf_dst = np.array(self._buffers[dst], dtype=np.float64)

        n = min(len(buf_src), len(buf_dst))
        if n < self.min_samples:
            return

        x = buf_src[-n:]
        y = buf_dst[-n:]

        std_x = float(np.std(x))
        std_y = float(np.std(y))

        # Si alguna serie es casi constante, no hay información transferible
        if std_x < 1e-7 or std_y < 1e-7:
            self._graph.remove_edge(src, dst)
            return

        x_norm = (x - np.mean(x)) / std_x
        y_norm = (y - np.mean(y)) / std_y

        effective_max_lag = min(self.max_lag, n // 3)
        if effective_max_lag < 1:
            return

        xn = (x - np.mean(x)) / std_x
        yn = (y - np.mean(y)) / std_y

        full_corr = np.correlate(xn, yn, mode="full")
        mid = n - 1
        lags = np.arange(1, effective_max_lag + 1)

        # Forward: x antecede a y (x_{t-lag} vs y_t)
        fwd_corrs = full_corr[mid - lags] / (n - lags)
        # Backward: y antecede a x (y_{t-lag} vs x_t)
        bwd_corrs = full_corr[mid + lags] / (n - lags)

        best_fwd_idx = int(np.argmax(np.abs(fwd_corrs)))
        best_fwd_lag = int(lags[best_fwd_idx])
        abs_fwd = float(np.clip(abs(fwd_corrs[best_fwd_idx]), 0.0, 1.0))
        abs_bwd = float(np.clip(np.max(np.abs(bwd_corrs)), 0.0, 1.0))

        # Comprobar si ya existe una arista establecida previamente
        existing_edge = self._graph._forward_adj.get(src, {}).get(dst)

        # Regla de dirección causal robusta con histéresis:
        # Se establece o retiene arista si supera el umbral y fwd domina o la arista ya estaba consolidada
        if abs_fwd >= self.coupling_threshold and (abs_fwd >= (abs_bwd - 0.05) or existing_edge is not None):
            final_lag = best_fwd_lag
            # Si la arista ya estaba sólidamente establecida, preservar su retardo característico
            if existing_edge is not None and existing_edge.coupling_strength > 0.7:
                est_lag = existing_edge.lag_steps
                if est_lag <= effective_max_lag:
                    r_at_est = float(abs(fwd_corrs[est_lag - 1]))
                    if r_at_est >= self.coupling_threshold:
                        final_lag = est_lag
                        abs_fwd = max(abs_fwd, r_at_est)

            edge = CausalEdge(
                source_series_id=src,
                target_series_id=dst,
                lag_steps=final_lag,
                coupling_strength=abs_fwd,
                relation_type=CausalRelationType.DIRECT_LEAD,
            )
            self._graph.add_edge(edge)
        elif abs_fwd < self.coupling_threshold:
            self._graph.remove_edge(src, dst)

    def get_active_edges(self) -> Sequence[CausalEdge]:
        """Retorna las aristas causales activas del grafo subyacente."""
        return self._graph.get_active_edges()
