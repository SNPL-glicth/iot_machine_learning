"""Adaptador de SubDetector a AsymmetricExpertPort para MoE asimétrico.

Permite registrar cualquier SubDetector clásico de infraestructura dentro del catálogo
de AsymmetricDispatcher con afinidad de representación (10X, 2X, RAW).
"""

from __future__ import annotations

from typing import Any, Sequence

from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    RepresentationLevel,
)
from iot_machine_learning.domain.ports.asymmetric_expert_port import (
    AsymmetricExpertPort,
)


class SubDetectorExpertAdapter(AsymmetricExpertPort):
    """Adapta un SubDetector al contrato AsymmetricExpertPort."""

    def __init__(
        self,
        sub_detector: Any,
        affinity: RepresentationLevel,
        calibration_layer: Any | None = None,
        scaler: Any | None = None,
        compute_cost_estimate: float = 1.0,
    ) -> None:
        self.sub_detector = sub_detector
        self.name = sub_detector.method_name
        self.affinity = affinity
        self.calibration_layer = calibration_layer
        self.scaler = scaler
        self.compute_cost_estimate = compute_cost_estimate
        self._current_context: dict[str, Any] = {}

    def set_context(self, context: dict[str, Any]) -> None:
        """Actualiza el contexto temporal/ventana para la evaluación del subdetector."""
        self._current_context = dict(context)

    def evaluate(self, series_slice: Sequence[float]) -> EvidenceScore:
        """Calcula la evidencia de anomalía dado un slice en su escala nativa."""
        val = series_slice[-1] if series_slice else 0.0
        val_norm = (
            float(self.scaler.transform([[val]])[0, 0])
            if self.scaler is not None
            else val
        )
        ctx = dict(self._current_context)
        if "nd_features" in ctx and self.scaler is not None:
            ctx["nd_features"][0, 0] = val_norm

        if self.calibration_layer is not None:
            raw_s = self.sub_detector.raw_score(val_norm, **ctx)
            v = self.calibration_layer.transform(self.sub_detector.method_name, raw_s)
        else:
            v = self.sub_detector.vote(val_norm, **ctx)

        prob = float(v) if v is not None else 0.0
        return EvidenceScore(
            expert_name=self.name,
            representation_affinity=self.affinity,
            anomaly_probability=prob,
            compute_cost_estimate=self.compute_cost_estimate,
        )
