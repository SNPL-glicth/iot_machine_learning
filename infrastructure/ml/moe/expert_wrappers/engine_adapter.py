"""EngineExpertAdapter — adapta PredictionEngine a ExpertPort.

Permite usar engines existentes (Taylor, Statistical, Baseline) como
expertos dentro de la arquitectura MoE.

Patrón Adapter: convierte interface de PredictionEngine a ExpertPort.
"""

from __future__ import annotations

from typing import Optional

from iot_machine_learning.domain.ports.expert_port import ExpertPort, ExpertOutput, ExpertCapability
from iot_machine_learning.domain.ports.prediction_port import PredictionPort
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow


class EngineExpertAdapter(ExpertPort):
    """Adapta un PredictionPort a ExpertPort.
    
    Wrapper que permite usar cualquier motor existente como experto
    en el pool MoE.
    
    Responsabilidades:
    1. Traducir llamadas ExpertPort → PredictionPort
    2. Convertir Prediction → ExpertOutput
    3. Declarar capacidades apropiadas
    
    Example:
        >>> from infrastructure.ml.engines import TaylorPredictionEngine
        >>> engine = TaylorPredictionEngine()
        >>> expert = EngineExpertAdapter(
        ...     engine=engine,
        ...     capabilities=ExpertCapability(
        ...         regimes=("volatile", "trending"),
        ...         min_points=5,
        ...     )
        ... )
        >>> registry.register("taylor", expert)
    """
    
    def __init__(
        self,
        engine: PredictionPort,
        capabilities: ExpertCapability,
        name_override: Optional[str] = None,
    ):
        """Inicializa adaptador.
        
        Args:
            engine: Motor de predicción a adaptar.
            capabilities: Capacidades declaradas del experto.
            name_override: Nombre alternativo (si None, usa engine.name).
        """
        self._engine = engine
        self._capabilities = capabilities
        self._name = name_override or engine.name
    
    @property
    def name(self) -> str:
        """Nombre identificador del experto."""
        return self._name
    
    @property
    def capabilities(self) -> ExpertCapability:
        """Capacidades declaradas."""
        return self._capabilities
    
    def predict(self, window: SensorWindow) -> ExpertOutput:
        """Genera predicción adaptando el engine.
        
        Args:
            window: Ventana de lecturas.
            
        Returns:
            ExpertOutput con resultado del engine.
        """
        # Ejecutar engine
        prediction = self._engine.predict(window)
        
        # Convertir a ExpertOutput
        return ExpertOutput(
            prediction=prediction.predicted_value,
            confidence=prediction.confidence,
            trend=prediction.trend,
            metadata=prediction.metadata,
        )
    
    def can_handle(self, window: SensorWindow) -> bool:
        """Verifica si puede procesar la ventana.
        
        Args:
            window: Ventana a evaluar.
            
        Returns:
            True si el engine puede operar con esos datos.
        """
        n_points = len(window.readings)
        return self._engine.can_handle(n_points)
    
    def estimate_latency_ms(self, n_points: int) -> float:
        """Estima latencia basada en costo computacional.
        
        Args:
            n_points: Número de puntos.
            
        Returns:
            Estimación de milisegundos.
        """
        base = self._capabilities.computational_cost * 10.0
        # Ajustar por complejidad O(n) o O(n²) típica
        return base * (1 + n_points / 50)


class EnsembleExpertAdapter(ExpertPort):
    """Adapta un ensemble de engines como un solo experto.
    
    Útil para encapsular VotingAnomalyDetector o similar como
    un experto más en el pool MoE.
    """
    
    def __init__(
        self,
        ensemble_engine: PredictionPort,
        capabilities: ExpertCapability,
        name: str = "ensemble",
    ):
        """Inicializa adaptador de ensemble.
        
        Args:
            ensemble_engine: Engine ensemble (ej: VotingAnomalyDetector).
            capabilities: Capacidades declaradas.
            name: Nombre identificador.
        """
        self._engine = ensemble_engine
        self._capabilities = capabilities
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def capabilities(self) -> ExpertCapability:
        return self._capabilities
    
    def predict(self, window: SensorWindow) -> ExpertOutput:
        prediction = self._engine.predict(window)
        return ExpertOutput(
            prediction=prediction.predicted_value,
            confidence=prediction.confidence,
            trend=prediction.trend,
            metadata=prediction.metadata,
        )
    
    def can_handle(self, window: SensorWindow) -> bool:
        return self._engine.can_handle(len(window.readings))


def create_baseline_expert(engine: PredictionPort) -> EngineExpertAdapter:
    """Factory para experto baseline.
    
    Args:
        engine: BaselineMovingAverageEngine o similar.
        
    Returns:
        Expert adaptado con capacidades apropiadas.
    """
    return EngineExpertAdapter(
        engine=engine,
        capabilities=ExpertCapability(
            regimes=("stable",),
            domains=("iot", "finance", "healthcare"),
            min_points=3,
            max_points=1000,
            specialties=(),
            computational_cost=0.5,  # Muy eficiente
        ),
        name_override="baseline",
    )


def create_statistical_expert(engine: PredictionPort) -> EngineExpertAdapter:
    """Factory para experto statistical.
    
    Args:
        engine: StatisticalPredictionEngine o similar.
        
    Returns:
        Expert adaptado con capacidades apropiadas.
    """
    return EngineExpertAdapter(
        engine=engine,
        capabilities=ExpertCapability(
            regimes=("stable", "trending"),
            domains=("iot", "finance"),
            min_points=5,
            max_points=500,
            specialties=("seasonality",),
            computational_cost=1.0,
        ),
        name_override="statistical",
    )


def create_taylor_expert(engine: PredictionPort) -> EngineExpertAdapter:
    """Factory para experto taylor.
    
    Args:
        engine: TaylorPredictionEngine.
        
    Returns:
        Expert adaptado con capacidades apropiadas.
    """
    return EngineExpertAdapter(
        engine=engine,
        capabilities=ExpertCapability(
            regimes=("trending", "volatile"),
            domains=("iot",),
            min_points=5,
            max_points=200,
            specialties=("non_linear", "derivatives"),
            computational_cost=2.0,  # Más costoso
        ),
        name_override="taylor",
    )
