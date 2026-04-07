"""Puertos para el motor de análisis unificado."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class InputType(Enum):
    """Tipo de entrada detectado."""
    TEXT = "text"
    TIMESERIES = "timeseries"
    DOCUMENT = "document"
    TABULAR = "tabular"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AnalysisContext:
    """Contexto de análisis con aislamiento multi-tenant."""
    tenant_id: str
    series_id: str = "unknown"
    input_type: Optional[InputType] = None
    domain_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cognitive_memory: Optional[Any] = None  # CognitiveMemoryPort
    budget_ms: float = 2000.0


@dataclass
class Signal:
    """Señal percibida del input."""
    raw_data: Any
    input_type: InputType
    domain: str
    features: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Perception:
    """Percepción de un analizador específico."""
    perspective: str  # Nombre del analizador
    score: float  # Valor predicho/detectado
    confidence: float
    evidence: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Compatibilidad con EnginePerception
    @property
    def engine_name(self) -> str:
        return self.perspective
    
    @property
    def predicted_value(self) -> float:
        return self.score


@dataclass
class Decision:
    """Decisión final del razonamiento."""
    severity: str  # 'info', 'warning', 'critical'
    confidence: float
    perceptions: List[Perception]
    weights: Dict[str, float]
    selected_engine: Optional[str] = None
    selection_reason: str = ""
    fusion_method: str = "weighted_average"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Explanation:
    """Explicación legible del análisis."""
    narrative: str
    confidence: float
    contributions: Dict[str, float]
    reasoning_trace: List[str]
    domain: str = "general"
    severity: str = "info"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Resultado completo del análisis."""
    signal: Signal
    decision: Decision
    explanation: Explanation
    pipeline_timing: Dict[str, float] = field(default_factory=dict)
    recall_context: Optional[Dict[str, Any]] = None


class AnalysisEnginePort(ABC):
    """Puerto para motores de análisis."""
    
    @abstractmethod
    def analyze(
        self,
        raw_data: Any,
        context: AnalysisContext,
    ) -> AnalysisResult:
        """Ejecuta análisis completo."""
        ...


class PhasePort(ABC):
    """Puerto para fases del pipeline."""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Ejecuta la fase."""
        ...
