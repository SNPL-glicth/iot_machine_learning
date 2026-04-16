"""CapacityScheduler — control de carga para MoE.

Decide dinámicamente cuántos expertos (k) ejecutar según:
- Carga actual del sistema
- Latencia objetivo
- Entropía del gating (confianza en la decisión)

ISO 25010: Eficiencia de performance bajo diferentes cargas.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List
from collections import deque

from .gating.base import GatingProbs, ContextVector


@dataclass
class SystemLoadMetrics:
    """Métricas de carga del sistema.
    
    Attributes:
        cpu_percent: Uso de CPU [0.0, 1.0].
        memory_percent: Uso de memoria [0.0, 1.0].
        queue_depth: Profundidad de cola de predicciones pendientes.
        recent_latency_ms: Latencia promedio reciente.
        timestamp: Timestamp de la medición.
    """
    cpu_percent: float = 0.5
    memory_percent: float = 0.5
    queue_depth: int = 0
    recent_latency_ms: float = 50.0
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        # Validar rangos
        self.cpu_percent = max(0.0, min(1.0, self.cpu_percent))
        self.memory_percent = max(0.0, min(1.0, self.memory_percent))


class CapacityScheduler:
    """Scheduler adaptativo de capacidad para MoE.
    
    Responsabilidades:
    1. Monitorear carga del sistema (CPU, memoria, latencia)
    2. Decidir k (número de expertos) dinámicamente
    3. Priorizar precisión vs. latencia según contexto
    
    Estrategias de adaptación:
    - Carga alta: k=1 (solo el mejor experto)
    - Carga media: k=2 (balance precisión/costo)
    - Carga baja: k=3 (máxima precisión)
    - Gating con baja entropía: k=1 (un experto domina)
    
    Args:
        latency_target_ms: Objetivo de latencia percentil 99.
        min_k: Mínimo de expertos (>= 1).
        max_k: Máximo de expertos.
        load_metric_source: Callable que retorna SystemLoadMetrics.
    """
    
    # Umbrales de carga
    HIGH_LOAD_THRESHOLD = 0.8
    MEDIUM_LOAD_THRESHOLD = 0.5
    
    # Entropía para decisión de confianza
    LOW_ENTROPY_THRESHOLD = 0.3  # Gating muy seguro
    
    def __init__(
        self,
        latency_target_ms: float = 100.0,
        min_k: int = 1,
        max_k: int = 3,
        load_metric_source: Optional[Callable[[], SystemLoadMetrics]] = None,
    ):
        self._latency_target = latency_target_ms
        self._min_k = max(1, min_k)
        self._max_k = max(self._min_k, max_k)
        self._load_source = load_metric_source
        
        # Histórico para promedios
        self._latency_history: deque = deque(maxlen=100)
        self._load_history: deque = deque(maxlen=50)
    
    def compute_k(
        self,
        context: ContextVector,
        gating_probs: GatingProbs,
        estimated_latency_per_expert_ms: float = 10.0,
    ) -> int:
        """Decide número de expertos a ejecutar.
        
        Args:
            context: Contexto de la predicción.
            gating_probs: Probabilidades del gating network.
            estimated_latency_per_expert_ms: Estimación de latencia por experto.
            
        Returns:
            Número de expertos (k) a ejecutar.
        """
        # Factor 1: Carga del sistema
        load_k = self._k_by_system_load()
        
        # Factor 2: Entropía del gating (confianza)
        entropy_k = self._k_by_entropy(gating_probs)
        
        # Factor 3: Presupuesto temporal
        budget_k = self._k_by_latency_budget(
            estimated_latency_per_expert_ms,
            context.n_points
        )
        
        # Combinar factores: usar el mínimo conservador
        k = min(load_k, entropy_k, budget_k)
        
        # Aplicar bounds
        k = max(self._min_k, min(self._max_k, k))
        
        # Registrar para métricas
        self._latency_history.append(estimated_latency_per_expert_ms * k)
        
        return k
    
    def _k_by_system_load(self) -> int:
        """Determina k basado en carga actual del sistema.
        
        Returns:
            k recomendado según carga.
        """
        load = self._get_current_load()
        
        if load.cpu_percent > self.HIGH_LOAD_THRESHOLD:
            return 1  # Alta carga: mínimo
        elif load.cpu_percent > self.MEDIUM_LOAD_THRESHOLD:
            return 2  # Media carga: balance
        else:
            return self._max_k  # Baja carga: máximo
    
    def _k_by_entropy(self, gating_probs: GatingProbs) -> int:
        """Determina k basado en confianza del gating.
        
        Si un experto domina (baja entropía), usar solo ese.
        
        Args:
            gating_probs: Probabilidades del gating.
            
        Returns:
            k recomendado según entropía.
        """
        entropy = gating_probs.entropy
        max_prob = gating_probs.max_probability
        
        # Si un experto tiene > 80% de probabilidad, confiar solo en él
        if max_prob > 0.8:
            return 1
        
        # Si entropía muy baja (alta confianza en top-1), usar k=1
        if entropy < self.LOW_ENTROPY_THRESHOLD:
            return 1
        
        # Entropía media: usar k=2
        if entropy < 1.0:
            return 2
        
        # Alta incertidumbre: diversificar
        return self._max_k
    
    def _k_by_latency_budget(
        self,
        latency_per_expert_ms: float,
        n_points: int
    ) -> int:
        """Determina k basado en presupuesto de latencia.
        
        Args:
            latency_per_expert_ms: Latencia estimada por experto.
            n_points: Número de puntos en ventana (afecta latencia).
            
        Returns:
            k máximo permitido por presupuesto temporal.
        """
        # Ajustar por tamaño de ventana (más puntos = más lento)
        adjusted_latency = latency_per_expert_ms * (1 + n_points / 100)
        
        # Calcular cuántos expertos caben en el presupuesto
        max_k_by_budget = int(self._latency_target / adjusted_latency)
        
        return max(1, max_k_by_budget)
    
    def _get_current_load(self) -> SystemLoadMetrics:
        """Obtiene métricas actuales de carga.
        
        Returns:
            SystemLoadMetrics actual.
        """
        if self._load_source:
            metrics = self._load_source()
            self._load_history.append(metrics)
            return metrics
        
        # Fallback: usar último conocido o default
        if self._load_history:
            return self._load_history[-1]
        
        return SystemLoadMetrics()  # Default medio
    
    def get_recommended_config(self) -> Dict[str, any]:
        """Obtiene configuración recomendada basada en histórico.
        
        Returns:
            Dict con recomendaciones.
        """
        if not self._latency_history:
            return {
                "avg_latency_ms": None,
                "recommended_k": self._max_k,
                "confidence": "low",
            }
        
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        
        # Recomendación basada en objetivo
        if avg_latency > self._latency_target * 1.5:
            recommended_k = max(1, self._max_k - 1)
            confidence = "reduce_k"
        elif avg_latency < self._latency_target * 0.5:
            recommended_k = self._max_k
            confidence = "increase_k_possible"
        else:
            recommended_k = self._max_k
            confidence = "optimal"
        
        return {
            "avg_latency_ms": round(avg_latency, 2),
            "latency_target_ms": self._latency_target,
            "recommended_k": recommended_k,
            "current_k_range": (self._min_k, self._max_k),
            "confidence": confidence,
            "samples": len(self._latency_history),
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Estadísticas del scheduler.
        
        Returns:
            Dict con métricas.
        """
        return {
            "latency_target_ms": self._latency_target,
            "k_range": (self._min_k, self._max_k),
            "latency_samples": len(self._latency_history),
            "load_samples": len(self._load_history),
            "current_recommendation": self.get_recommended_config(),
        }


def create_default_scheduler(
    latency_target_ms: float = 100.0
) -> CapacityScheduler:
    """Factory para crear scheduler con configuración default.
    
    Args:
        latency_target_ms: Objetivo de latencia.
        
    Returns:
        CapacityScheduler configurado.
    """
    # Implementación simple de métricas de carga
    def simple_load_metrics() -> SystemLoadMetrics:
        # En producción, esto leería de monitoring real
        # Por ahora, retorna valores conservadores
        return SystemLoadMetrics(
            cpu_percent=0.5,
            memory_percent=0.5,
            queue_depth=0,
            recent_latency_ms=50.0,
        )
    
    return CapacityScheduler(
        latency_target_ms=latency_target_ms,
        min_k=1,
        max_k=2,  # Conservador para inicio
        load_metric_source=simple_load_metrics,
    )
