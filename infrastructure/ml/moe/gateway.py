"""MoEGateway — Gateway principal de arquitectura Mixture of Experts.

Implementa PredictionPort del dominio usando MoE internamente.
Coordina: encoding → gating → dispatch → fusion → explicación.

Arquitectura Hexagonal:
- Implementa PredictionPort (contrato del dominio)
- Usa ExpertRegistry, GatingNetwork (infrastructure)
- Expone métodos de dominio (predict), oculta complejidad MoE

ISO 42001: Registra todas las decisiones para AI governance.
ISO 25010: Performance eficiente bajo diferentes cargas.
"""

from __future__ import annotations

import time
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from collections import OrderedDict

from domain.ports.prediction_port import PredictionPort
from domain.ports.expert_port import ExpertPort, ExpertOutput
from domain.entities.prediction import Prediction
from domain.entities.sensor_reading import SensorWindow
from domain.entities.series.structural_analysis import StructuralAnalysis
from domain.validators.structural_analysis import compute_structural_analysis

from .registry import ExpertRegistry
from .gating.base import GatingNetwork, ContextVector
from .fusion import SparseFusionLayer, FusedResult
from .scheduler import CapacityScheduler


@dataclass(frozen=True)
class MoEMetadata:
    """Metadata de ejecución MoE para explicabilidad.
    
    Attributes:
        selected_experts: Expertos efectivamente ejecutados.
        gating_probs: Probabilidades de todos los expertos.
        sparsity_k: Número de expertos ejecutados.
        total_available: Total de expertos en el pool.
        computation_saved_pct: Porcentaje de computación ahorrada.
        gating_entropy: Entropía del gating (incertidumbre).
        gating_explanation: Explicación textual de la decisión.
        fusion_uncertainty: Divergencia entre expertos.
        execution_latencies_ms: Latencia por experto.
        total_latency_ms: Latencia total del gateway.
        capacity_k_used: k decidido por CapacityScheduler.
    """
    selected_experts: List[str]
    gating_probs: Dict[str, float]
    sparsity_k: int
    total_available: int
    computation_saved_pct: float
    gating_entropy: float
    gating_explanation: str
    fusion_uncertainty: float
    execution_latencies_ms: Dict[str, float]
    total_latency_ms: float
    capacity_k_used: int


class MoEGateway(PredictionPort):
    """Gateway de Mixture of Experts.
    
    Implementa PredictionPort del dominio, permitiendo drop-in replacement
    de cualquier motor de predicción existente.
    
    Flujo de ejecución:
    1. encode_context() → ContextVector
    2. gating.route() → GatingProbs
    3. scheduler.compute_k() → k dinámico
    4. dispatch() → ejecutar k expertos
    5. fusion.fuse() → FusedResult
    6. build_prediction() → Prediction + metadata
    
    Thread-safe: todas las operaciones de estado protegidas.
    
    Args:
        expert_registry: Catálogo de expertos disponibles.
        gating_network: Estrategia de routing.
        fusion_layer: Capa de fusión sparse.
        capacity_scheduler: Scheduler adaptativo de k (opcional).
        max_history: Tamaño de histórico de predicciones.
    """
    
    def __init__(
        self,
        expert_registry: ExpertRegistry,
        gating_network: GatingNetwork,
        fusion_layer: SparseFusionLayer,
        capacity_scheduler: Optional[CapacityScheduler] = None,
        max_history: int = 1000,
    ):
        self._registry = expert_registry
        self._gating = gating_network
        self._fusion = fusion_layer
        self._scheduler = capacity_scheduler
        
        # Estado
        self._lock = threading.RLock()
        self._prediction_history: OrderedDict[str, MoEMetadata] = OrderedDict()
        self._max_history = max_history
        self._total_predictions = 0
        
        # Actualizar gating con expertos actuales
        self._sync_gating_experts()
    
    @property
    def name(self) -> str:
        """Nombre del gateway (para PredictionPort)."""
        return "moe_gateway"
    
    def predict(self, window: SensorWindow) -> Prediction:
        """Genera predicción usando arquitectura MoE.
        
        Implementa PredictionPort.predict().
        
        Args:
            window: Ventana temporal de lecturas.
            
        Returns:
            Prediction del dominio con metadata MoE.
            
        Raises:
            ValueError: Si no hay expertos disponibles o ventana inválida.
            RuntimeError: Si error interno no recuperable.
        """
        start_time = time.perf_counter()
        
        # 1. Codificar contexto
        context = self._encode_context(window)
        
        # 2. Gating: distribución sobre expertos
        gating_result = self._gating.route(context)
        
        # 3. Determinar k (sparsity level)
        k = self._determine_k(context, gating_result)
        
        # 4. Seleccionar top-k expertos
        selected_experts = gating_result.get_top_k(k)
        
        # 5. Ejecutar expertos seleccionados
        expert_outputs, latencies = self._dispatch(selected_experts, window)
        
        # 6. Fusionar resultados
        fused = self._fuse(expert_outputs, gating_result.probabilities)
        
        # 7. Construir metadata
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        moe_metadata = MoEMetadata(
            selected_experts=selected_experts,
            gating_probs=gating_result.probabilities,
            sparsity_k=len(selected_experts),
            total_available=len(self._registry),
            computation_saved_pct=fused.computation_saved_pct,
            gating_entropy=gating_result.entropy,
            gating_explanation=self._gating.explain(context, gating_result),
            fusion_uncertainty=fused.uncertainty,
            execution_latencies_ms=latencies,
            total_latency_ms=total_latency_ms,
            capacity_k_used=k,
        )
        
        # 8. Actualizar histórico
        self._update_history(window.series_id or "unknown", moe_metadata)
        
        # 9. Construir y retornar predicción
        return self._build_prediction(fused, moe_metadata, window)
    
    def can_handle(self, n_points: int) -> bool:
        """Indica si puede operar con n_points datos.
        
        True si al menos un experto puede manejar esos puntos.
        
        Args:
            n_points: Número de puntos disponibles.
            
        Returns:
            True si puede generar predicción.
        """
        candidates = self._registry.get_candidates(
            regime="stable",  # Cualquier régimen sirve para este check
            domain="iot",
            n_points=n_points
        )
        return len(candidates) > 0
    
    def _encode_context(self, window: SensorWindow) -> ContextVector:
        """Codifica ventana a vector de contexto.
        
        Args:
            window: Ventana de datos.
            
        Returns:
            ContextVector con features.
        """
        # Extraer valores y timestamps
        values = [r.value for r in window.readings]
        timestamps = [r.timestamp for r in window.readings]
        
        # Análisis estructural (dominio puro)
        analysis = compute_structural_analysis(values, timestamps)
        
        # Features de señal
        signal_features = {
            "mean": analysis.mean,
            "std": analysis.std,
            "slope": analysis.slope,
            "curvature": analysis.curvature,
            "noise_ratio": analysis.noise_ratio,
            "stability": analysis.stability,
            "trend_strength": analysis.trend_strength,
        }
        
        # Features temporales (si hay timestamps)
        temporal_features = {}
        if timestamps:
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamps[-1])
            temporal_features = {
                "hour": dt.hour,
                "day_of_week": dt.weekday(),
                "is_weekend": dt.weekday() >= 5,
            }
        
        return ContextVector(
            regime=analysis.regime,
            domain="iot",  # TODO: detectar desde window
            n_points=len(values),
            signal_features=signal_features,
            temporal_features=temporal_features,
        )
    
    def _determine_k(self, context: ContextVector, gating_result) -> int:
        """Decide cuántos expertos ejecutar.
        
        Args:
            context: Contexto de la predicción.
            gating_result: Resultado del gating.
            
        Returns:
            Número de expertos (k).
        """
        if self._scheduler:
            return self._scheduler.compute_k(
                context=context,
                gating_probs=gating_result,
                estimated_latency_per_expert_ms=10.0,
            )
        
        # Default: usar top-2 para balance
        return min(2, len(self._registry))
    
    def _dispatch(
        self,
        expert_ids: List[str],
        window: SensorWindow
    ) -> tuple[Dict[str, ExpertOutput], Dict[str, float]]:
        """Ejecuta expertos seleccionados.
        
        Args:
            expert_ids: IDs de expertos a ejecutar.
            window: Ventana de datos.
            
        Returns:
            Tuple de (outputs, latencias).
        """
        outputs = {}
        latencies = {}
        
        for expert_id in expert_ids:
            expert = self._registry.get(expert_id)
            if not expert:
                continue
            
            if not expert.can_handle(window):
                continue
            
            start = time.perf_counter()
            try:
                output = expert.predict(window)
                latency_ms = (time.perf_counter() - start) * 1000
                
                outputs[expert_id] = output
                latencies[expert_id] = latency_ms
            except Exception:
                # Fail-silent: ignorar experto que falla
                latencies[expert_id] = (time.perf_counter() - start) * 1000
                continue
        
        return outputs, latencies
    
    def _fuse(
        self,
        outputs: Dict[str, ExpertOutput],
        gating_probs: Dict[str, float]
    ) -> FusedResult:
        """Fusiona salidas de expertos.
        
        Args:
            outputs: Salidas de expertos ejecutados.
            gating_probs: Probabilidades del gating.
            
        Returns:
            Resultado fusionado.
            
        Raises:
            RuntimeError: Si no hay outputs para fusionar.
        """
        if not outputs:
            raise RuntimeError("Ningún experto produjo output válido")
        
        return self._fusion.fuse(
            outputs=outputs,
            gating_probs=gating_probs,
            total_experts_available=len(self._registry)
        )
    
    def _build_prediction(
        self,
        fused: FusedResult,
        moe_metadata: MoEMetadata,
        window: SensorWindow
    ) -> Prediction:
        """Construye Prediction del dominio.
        
        Args:
            fused: Resultado de fusión.
            moe_metadata: Metadata de ejecución MoE.
            window: Ventana original.
            
        Returns:
            Prediction con metadata enriquecida.
        """
        metadata = {
            "moe": {
                "selected_experts": moe_metadata.selected_experts,
                "sparsity_k": moe_metadata.sparsity_k,
                "total_available": moe_metadata.total_available,
                "computation_saved_pct": round(moe_metadata.computation_saved_pct, 1),
                "gating_entropy": round(moe_metadata.gating_entropy, 4),
                "fusion_uncertainty": round(moe_metadata.fusion_uncertainty, 4),
                "gating_explanation": moe_metadata.gating_explanation,
                "latencies_ms": {k: round(v, 2) for k, v in moe_metadata.execution_latencies_ms.items()},
                "total_latency_ms": round(moe_metadata.total_latency_ms, 2),
            },
            "fusion_weights": moe_metadata.gating_probs,
            "selected_expert": fused.selected_expert,
        }
        
        return Prediction(
            series_id=str(window.sensor_id),
            predicted_value=fused.value,
            confidence=fused.confidence,
            trend=fused.trend,
            timestamp=time.time(),
            metadata=metadata,
        )
    
    def _update_history(self, series_id: str, metadata: MoEMetadata) -> None:
        """Actualiza histórico de predicciones (LRU).
        
        Args:
            series_id: ID de la serie.
            metadata: Metadata de la predicción.
        """
        with self._lock:
            key = f"{series_id}:{self._total_predictions}"
            self._prediction_history[key] = metadata
            self._prediction_history.move_to_end(key)
            
            # Evict LRU si excede tamaño
            while len(self._prediction_history) > self._max_history:
                self._prediction_history.popitem(last=False)
            
            self._total_predictions += 1
    
    def _sync_gating_experts(self) -> None:
        """Sincroniza lista de expertos con gating network."""
        expert_ids = self._registry.list_all()
        self._gating.expert_ids = expert_ids
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas de operación del gateway.
        
        Returns:
            Dict con métricas.
        """
        with self._lock:
            if not self._prediction_history:
                avg_saved = 0.0
                avg_k = 0.0
            else:
                recent = list(self._prediction_history.values())[-100:]
                avg_saved = sum(m.computation_saved_pct for m in recent) / len(recent)
                avg_k = sum(m.sparsity_k for m in recent) / len(recent)
            
            return {
                "total_predictions": self._total_predictions,
                "expert_pool_size": len(self._registry),
                "avg_computation_saved_pct": round(avg_saved, 1),
                "avg_sparsity_k": round(avg_k, 2),
                "gating_type": self._gating.__class__.__name__,
                "has_scheduler": self._scheduler is not None,
            }
    
    def explain_last(self, series_id: str) -> Optional[str]:
        """Explica última predicción para una serie.
        
        Args:
            series_id: ID de la serie.
            
        Returns:
            Explicación textual o None si no hay histórico.
        """
        with self._lock:
            # Buscar última entrada para esta serie
            for key in reversed(self._prediction_history.keys()):
                if key.startswith(f"{series_id}:"):
                    metadata = self._prediction_history[key]
                    return metadata.gating_explanation
            return None
