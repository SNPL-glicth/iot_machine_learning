"""Caso de uso: Enriquecer predicción con análisis cognitivo.

Responsabilidad única: Computar structural analysis + metacognitive classification.
Fail-safe: si algo falla, retorna dict vacío sin romper el flujo.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from iot_machine_learning.domain.ports.storage_port import StoragePort
    from iot_machine_learning.application.use_cases.predict_sensor_value import (
        PredictionDTO,
    )

from iot_machine_learning.application.explainability.explanation_renderer import (
    ExplanationRenderer,
)
from iot_machine_learning.domain.entities.explainability.explanation import (
    Explanation,
    Outcome,
)
from iot_machine_learning.domain.entities.explainability.signal_snapshot import (
    SignalSnapshot,
)

logger = logging.getLogger(__name__)


class EnrichPredictionUseCase:
    """Enriquece predicción con análisis estructural y metacognitivo.
    
    Computa:
    1. Structural analysis (regime, slope, curvature, noise, stability)
    2. Metacognitive classification (certainty, complexity, risk)
    
    Fail-safe: retorna dict vacío si falla, no rompe el flujo principal.
    """
    
    def __init__(self, storage: "StoragePort"):
        """Inicializa con storage port.
        
        Args:
            storage: Port para cargar ventana de sensor
        """
        self._storage = storage
        self._renderer = ExplanationRenderer()
    
    def execute(
        self,
        *,
        sensor_id: int,
        window_size: int,
        dto: "PredictionDTO",
    ) -> Dict[str, Any]:
        """Computa enriquecimiento cognitivo.
        
        Args:
            sensor_id: ID del sensor
            window_size: Tamaño de ventana para análisis
            dto: DTO con predicción base
        
        Returns:
            Dict con structural_analysis y metacognitive, o {} si falla
        """
        try:
            # 1. Cargar ventana para structural analysis
            sensor_window = self._storage.load_sensor_window(
                sensor_id=sensor_id,
                limit=window_size,
            )
            if sensor_window.is_empty:
                logger.debug(
                    "enrichment_empty_window",
                    extra={"sensor_id": sensor_id},
                )
                return {}
            
            # 2. Compute structural analysis (domain pure function)
            sa = sensor_window.structural_analysis
            structural_dict = {
                "regime": sa.regime.value,
                "slope": round(sa.slope, 6),
                "curvature": round(sa.curvature, 6),
                "noise_ratio": round(sa.noise_ratio, 6),
                "stability": round(sa.stability, 6),
                "trend_strength": round(sa.trend_strength, 6),
                "mean": round(sa.mean, 6),
                "std": round(sa.std, 6),
                "n_points": sa.n_points,
            }
            
            # 3. Build minimal Explanation for metacognitive classification
            signal = SignalSnapshot(
                n_points=sa.n_points,
                mean=sa.mean,
                std=sa.std,
                noise_ratio=sa.noise_ratio,
                slope=sa.slope,
                curvature=sa.curvature,
                regime=sa.regime.value,
                dt=sa.dt if hasattr(sa, "dt") else 1.0,
            )
            outcome = Outcome(
                kind="prediction",
                predicted_value=dto.predicted_value,
                confidence=dto.confidence_score,
                trend=dto.trend,
            )
            explanation = Explanation(
                series_id=dto.series_id,
                signal=signal,
                outcome=outcome,
                audit_trace_id=dto.audit_trace_id,
            )
            
            # 4. Render metacognitive classifications
            rendered = self._renderer.render_structured_json(explanation)
            metacognitive = rendered.get("metacognitive", {})
            
            logger.debug(
                "enrichment_computed",
                extra={
                    "sensor_id": sensor_id,
                    "regime": structural_dict["regime"],
                    "certainty": metacognitive.get("certainty"),
                },
            )
            
            return {
                "structural_analysis": structural_dict,
                "metacognitive": metacognitive,
            }
        
        except Exception as exc:
            logger.warning(
                "enrichment_failed",
                extra={
                    "sensor_id": sensor_id,
                    "error": str(exc),
                },
            )
            return {}
