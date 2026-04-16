"""Base definitions for Gating Networks in MoE architecture.

Implementa el patrón Strategy para diferentes algoritmos de routing.
Cada estrategia de gating hereda de GatingNetwork y implementa route().

ISO 42001: Las decisiones de routing deben ser explicables y trazables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass(frozen=True)
class GatingProbs:
    """Distribución de probabilidades sobre expertos.
    
    Resultado del gating network. Contiene probabilidades para cada experto,
    métricas de incertidumbre (entropy) y top selection.
    
    Attributes:
        probabilities: Dict mapping expert_id -> probabilidad [0.0, 1.0].
        entropy: Entropía de la distribución (incertidumbre).
        top_expert: Experto con mayor probabilidad.
        metadata: Datos adicionales específicos del gating usado.
    """
    probabilities: Dict[str, float]
    entropy: float
    top_expert: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Validación: probabilidades deben sumar aprox 1.0
        total = sum(self.probabilities.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"Probabilidades deben sumar ~1.0, sumaron {total:.4f}"
            )
    
    def get_top_k(self, k: int = 2) -> List[str]:
        """Obtiene top-k expertos por probabilidad.
        
        Args:
            k: Número de expertos a retornar.
            
        Returns:
            Lista de expert_id ordenados por probabilidad descendente.
        """
        sorted_items = sorted(
            self.probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [expert_id for expert_id, _ in sorted_items[:k]]
    
    def get_probability(self, expert_id: str) -> float:
        """Obtiene probabilidad de un experto específico.
        
        Args:
            expert_id: Identificador del experto.
            
        Returns:
            Probabilidad o 0.0 si no existe.
        """
        return self.probabilities.get(expert_id, 0.0)
    
    @property
    def max_probability(self) -> float:
        """Probabilidad del experto más probable."""
        return max(self.probabilities.values()) if self.probabilities else 0.0
    
    @property
    def min_probability(self) -> float:
        """Probabilidad del experto menos probable."""
        return min(self.probabilities.values()) if self.probabilities else 0.0


@dataclass(frozen=True)
class ContextVector:
    """Vector de contexto para decisión de routing.
    
    Contiene todas las features necesarias para que el gating
    network decida qué expertos activar.
    
    Attributes:
        regime: Régimen detectado (stable, trending, volatile, noisy).
        domain: Dominio del contexto (iot, finance, healthcare).
        n_points: Número de puntos en la ventana.
        signal_features: Features extraídos de la señal.
        temporal_features: Features temporales (hora, día, etc).
        historical_performance: MAE histórico por experto (opcional).
    """
    regime: str
    domain: str
    n_points: int
    signal_features: Dict[str, float]
    temporal_features: Optional[Dict[str, any]] = None
    historical_performance: Optional[Dict[str, float]] = None
    
    def to_array(self) -> List[float]:
        """Convierte a lista numérica para modelos ML.
        
        Returns:
            Lista de valores numéricos en orden consistente.
        """
        # Orden determinístico para reproducibilidad
        base = [
            self.n_points,
            self.signal_features.get("mean", 0.0),
            self.signal_features.get("std", 0.0),
            self.signal_features.get("slope", 0.0),
            self.signal_features.get("curvature", 0.0),
            self.signal_features.get("noise_ratio", 0.0),
            self.signal_features.get("stability", 0.0),
        ]
        
        # One-hot encoding simple para régimen
        regimes = ["stable", "trending", "volatile", "noisy"]
        regime_encoded = [1.0 if r == self.regime else 0.0 for r in regimes]
        
        return base + regime_encoded
    
    def to_dict(self) -> Dict[str, any]:
        """Serializa a diccionario."""
        return {
            "regime": self.regime,
            "domain": self.domain,
            "n_points": self.n_points,
            "signal_features": self.signal_features,
            "temporal_features": self.temporal_features,
        }


class GatingNetwork(ABC):
    """Abstract base class para estrategias de gating.
    
    Implementa el patrón Strategy: diferentes algoritmos de routing
    pueden intercambiarse sin modificar el MoEGateway.
    
    Estrategias concretas:
    - RegimeBasedGating: Reglas fijas por régimen (heurístico)
    - TreeGatingNetwork: Árbol de decisión (interpretable)
    - NeuralGatingNetwork: Red neuronal (precisión máxima)
    - ThompsonSamplingGating: Multi-armed bandit (exploración)
    
    Subclases deben implementar:
    - route(): Lógica de routing principal
    - explain(): Explicabilidad de decisiones (ISO 42001)
    """
    
    def __init__(self, expert_ids: Optional[List[str]] = None):
        """Inicializa gating network.
        
        Args:
            expert_ids: Lista de expertos disponibles. Si None,
                       se infiere en la primera llamada a route().
        """
        self._expert_ids = expert_ids or []
        self._routing_count = 0
    
    @property
    def expert_ids(self) -> List[str]:
        """Expertos conocidos por este gating network."""
        return self._expert_ids.copy()
    
    @expert_ids.setter
    def expert_ids(self, value: List[str]):
        """Actualiza lista de expertos (útil para registro dinámico)."""
        self._expert_ids = list(value)
    
    @abstractmethod
    def route(self, context: ContextVector) -> GatingProbs:
        """Decide distribución de probabilidades sobre expertos.
        
        Este es el método core que todas las estrategias deben implementar.
        Recibe un contexto y retorna probabilidades para cada experto.
        
        Args:
            context: Vector de contexto con features del input.
            
        Returns:
            GatingProbs con distribución de probabilidades.
            
        Raises:
            ValueError: Si contexto inválido o incompleto.
            RuntimeError: Si error interno durante routing.
        """
        ...
    
    @abstractmethod
    def explain(self, context: ContextVector, probs: GatingProbs) -> str:
        """Explica por qué se tomó la decisión de routing.
        
        Requerido por ISO 42001 para AI governance y explicabilidad.
        
        Args:
            context: Contexto usado para routing.
            probs: Probabilidades resultantes.
            
        Returns:
            Explicación textual de la decisión.
        """
        ...
    
    def route_batch(
        self,
        contexts: List[ContextVector]
    ) -> List[GatingProbs]:
        """Rutea múltiples contextos en batch (optimización).
        
        Default implementation llama a route() individualmente.
        Subclases pueden override para vectorización (neural networks).
        
        Args:
            contexts: Lista de vectores de contexto.
            
        Returns:
            Lista de GatingProbs en mismo orden.
        """
        return [self.route(ctx) for ctx in contexts]
    
    def get_stats(self) -> Dict[str, any]:
        """Estadísticas de operación.
        
        Returns:
            Dict con métricas del gating.
        """
        return {
            "routing_count": self._routing_count,
            "expert_count": len(self._expert_ids),
            "gating_type": self.__class__.__name__,
        }
