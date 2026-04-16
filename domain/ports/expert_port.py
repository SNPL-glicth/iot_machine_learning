"""Port de experto — contrato para expertos dentro de una arquitectura MoE.

Este port define el contrato mínimo que todo experto especializado debe cumplir
para ser orquestado por el MoEGateway. Es más ligero que PredictionPort porque
los expertos son componentes internos, no expuestos directamente al dominio.

Diseñado siguiendo ISO 42001 (AI governance) para trazabilidad de decisiones.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, Any, Optional
from dataclasses import dataclass

from ..entities.sensor_reading import SensorWindow


@dataclass(frozen=True)
class ExpertOutput:
    """Resultado de la ejecución de un experto.
    
    Contiene la predicción, metadados de confianza y diagnósticos
    necesarios para la fusión y explicabilidad del MoE.
    
    Attributes:
        prediction: Valor predicho por el experto.
        confidence: Confianza del experto en [0.0, 1.0].
        trend: Dirección de tendencia detectada ("up", "down", "stable").
        latency_ms: Tiempo de ejecución en milisegundos.
        stability: Indicador de estabilidad del experto [0.0, 1.0].
        local_fit_error: Error de ajuste local del modelo.
        metadata: Datos adicionales específicos del experto.
    """
    prediction: float
    confidence: float
    trend: str = "stable"
    latency_ms: float = 0.0
    stability: float = 0.0
    local_fit_error: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        # Validación post-inicialización para ISO 42001 compliance
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence debe estar en [0.0, 1.0], got {self.confidence}")
        if self.trend not in ("up", "down", "stable"):
            raise ValueError(f"trend debe ser 'up', 'down' o 'stable', got {self.trend}")


@dataclass(frozen=True)
class ExpertCapability:
    """Capacidades declaradas de un experto.
    
    Usado por el ExpertRegistry para matching de contexto.
    Permite filtrado eficiente de expertos elegibles.
    
    Attributes:
        regimes: Lista de regímenes que maneja (ej: ["stable", "trending"]).
        domains: Dominios de aplicación (ej: ["iot", "finance"]).
        min_points: Mínimo de puntos requeridos.
        max_points: Máximo recomendado (0 = ilimitado).
        specialties: Especialidades específicas (ej: ["seasonality", "anomalies"]).
        computational_cost: Costo relativo (1.0 = baseline).
    """
    regimes: tuple = ("stable",)
    domains: tuple = ("iot",)
    min_points: int = 3
    max_points: int = 0
    specialties: tuple = ()
    computational_cost: float = 1.0
    
    def supports_regime(self, regime: str) -> bool:
        """Check if capability supports given regime."""
        return regime in self.regimes
    
    def supports_domain(self, domain: str) -> bool:
        """Check if capability supports given domain."""
        return domain in self.domains
    
    def can_handle_points(self, n_points: int) -> bool:
        """Check if n_points is within acceptable range."""
        if n_points < self.min_points:
            return False
        if self.max_points > 0 and n_points > self.max_points:
            return False
        return True
    
    def matches_context(self, regime: str, domain: str, n_points: int) -> bool:
        """Full context matching."""
        return (
            self.supports_regime(regime) and
            self.supports_domain(domain) and
            self.can_handle_points(n_points)
        )


@runtime_checkable
class ExpertPort(Protocol):
    """Contrato para expertos especializados en arquitectura MoE.
    
    Todo experto que implemente este port puede ser:
    1. Registrado en ExpertRegistry
    2. Seleccionado por GatingNetwork
    3. Ejecutado por SparseDispatcher
    4. Fusionado en SparseFusionLayer
    
    El port es un Protocol (no ABC) para permitir implementaciones
    ligeras sin herencia forzada, siguiendo el principio de
    segregación de interfaces (SOLID ISP).
    
    Example:
        >>> class TaylorExpert:
        ...     @property
        ...     def name(self) -> str:
        ...         return "taylor"
        ...     
        ...     def predict(self, window: SensorWindow) -> ExpertOutput:
        ...         # ... lógica de predicción
        ...         return ExpertOutput(prediction=22.5, confidence=0.85)
        ...     
        ...     def can_handle(self, window: SensorWindow) -> bool:
        ...         return len(window.readings) >= 5
        ...     
        ...     @property
        ...     def capabilities(self) -> ExpertCapability:
        ...         return ExpertCapability(
        ...             regimes=("volatile", "trending"),
        ...             min_points=5
        ...         )
    """
    
    @property
    def name(self) -> str:
        """Nombre único identificador del experto.
        
        Debe ser único dentro del ExpertRegistry.
        
        Returns:
            String identificador (ej: "taylor", "baseline", "statistical").
        """
        ...
    
    def predict(self, window: SensorWindow) -> ExpertOutput:
        """Genera predicción a partir de la ventana de datos.
        
        Este es el método core del experto. Debe ser puro (sin side effects)
        y idempotente para permitir re-ejecución si es necesario.
        
        Args:
            window: Ventana temporal con lecturas del sensor.
            
        Returns:
            ExpertOutput con predicción, confianza y metadatos.
            
        Raises:
            ValueError: Si la ventana no cumple requisitos mínimos.
            RuntimeError: Si ocurre error interno no recuperable.
        """
        ...
    
    def can_handle(self, window: SensorWindow) -> bool:
        """Indica si el experto puede procesar esta ventana.
        
        Verifica precondiciones: número de puntos, calidad de datos,
        rangos válidos, etc.
        
        Args:
            window: Ventana a evaluar.
            
        Returns:
            True si el experto puede generar predicción válida.
        """
        ...
    
    @property
    def capabilities(self) -> ExpertCapability:
        """Declaración de capacidades del experto.
        
        Usado por el GatingNetwork para routing inteligente.
        Las capacidades deben ser honestas (no sobre-declarar).
        
        Returns:
            ExpertCapability con metadatos de especialización.
        """
        ...
    
    def estimate_latency_ms(self, n_points: int) -> float:
        """Estima latencia de ejecución dado tamaño de entrada.
        
        Opcional pero recomendado para CapacityScheduler.
        Permite decisiones de routing basadas en presupuesto temporal.
        
        Args:
            n_points: Número de puntos en la ventana.
            
        Returns:
            Estimación de milisegundos. Default: computational_cost * 10ms.
        """
        return self.capabilities.computational_cost * 10.0
