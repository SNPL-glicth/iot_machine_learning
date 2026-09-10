"""RosaRojaMoEEngine: Cabeza orquestadora de MoE que opera como PredictionEngine."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from .algorithms.engine import RosaRojaEngine
from .algorithms.modules.module1_ingestion import (
    MahalanobisFilter,
)
from .algorithms.modules.module3_moe_gating import (
    MultiplicativeMoEGating,
)
from .algorithms.modules.rhythm_generator import (
    RhythmTrajectoryGenerator,
)
from .algorithms.ports.expert_jury import ExpertJuryPort
from iot_machine_learning.infrastructure.ml.engines.core.factory import register_engine
from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)
from iot_machine_learning.infrastructure.ml.moe.experts.rosa_roja_expert import (
    RosaRojaResult,
)

from .jury_builder import build_default_moe_jury
from .pipeline_runner import (
    execute_time_series,
    map_to_prediction_result,
    map_to_rosa_roja_result,
    sanitize_inputs,
)

logger = logging.getLogger(__name__)


@register_engine("rosa_roja")
class RosaRojaMoEEngine(PredictionEngine):
    """Cabeza orquestadora de Mixture of Experts basada en Rosa Roja.

    Opera como motor de predicción unificado ('PredictionEngine') que evalúa
    la coherencia y seguridad estocástica gobernando un jurado de expertos.
    """

    def __init__(
        self,
        warmup_size: int = 11,
        jury: Optional[Sequence[ExpertJuryPort]] = None,
        noise_threshold: float = 3.0,
        variance_penalty: float = 0.5,
    ) -> None:
        self._warmup_size = max(11, warmup_size)
        self._jury = build_default_moe_jury(jury)

        self._ingestion = MahalanobisFilter(
            noise_threshold=noise_threshold,
            history_window=100,
            min_samples_for_cov=20,
        )
        self._rhythm = RhythmTrajectoryGenerator(
            min_trajectory_len=11,
            max_trajectory_len=15,
            top_k=4,
            oversample_factor=2,
        )
        self._gating = MultiplicativeMoEGating(variance_penalty=variance_penalty)
        self._core_engine = RosaRojaEngine(
            ingestion_filter=self._ingestion,
            rhythm_generator=self._rhythm,
            moe_gating=self._gating,
            expert_jury=self._jury,
            drift_sensors=[],
        )

    @property
    def name(self) -> str:
        return "rosa_roja"

    @property
    def core_engine(self) -> RosaRojaEngine:
        """Acceso al orquestador central subyacente."""
        return self._core_engine

    def can_handle(self, n_points: int) -> bool:
        """Determina si la ventana contiene suficientes datos para proyectar trayectorias."""
        return n_points >= self._warmup_size

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        """Genera predicción ejecutando el ciclo de orquestación MoE completo."""
        clean_v, clean_ts = sanitize_inputs(values, timestamps)
        if not self.can_handle(len(clean_v)):
            return PredictionResult(
                predicted_value=clean_v[-1] if clean_v else 0.0,
                confidence=0.0,
                trend="stable",
                metadata={"reason": "insufficient_history", "min_required": self._warmup_size},
            )

        plan, delta_next, inv_step = execute_time_series(
            engine=self._core_engine,
            values=clean_v,
            timestamps=clean_ts,
        )
        return map_to_prediction_result(
            plan=plan,
            delta_next=delta_next,
            last_value=clean_v[-1],
            invalidation_step=inv_step,
        )

    def analyze(self, s_t: Dict[str, Any]) -> RosaRojaResult:
        """Contrato directo para compatibilidad con RosaRojaExpert del pool MoE."""
        values = s_t.get("values", [])
        timestamps = s_t.get("timestamps")
        clean_v, clean_ts = sanitize_inputs(values, timestamps)

        if not self.can_handle(len(clean_v)):
            return RosaRojaResult(
                trajectory=[],
                trajectory_score=0.0,
                rhythm_score=0.0,
                lambda_val=0.0,
                theta_entropy=1.0,
                regime_alert=None,
                invalidation_step=None,
                expected_direction="stable",
                expected_magnitude=0.0,
                confidence=0.0,
                evidence={},
                status="insufficient_history",
            )

        plan, delta_next, inv_step = execute_time_series(
            engine=self._core_engine,
            values=clean_v,
            timestamps=clean_ts,
        )
        return map_to_rosa_roja_result(plan, delta_next, inv_step, s_t)
