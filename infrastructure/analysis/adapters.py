"""Adaptadores simples para testing del motor unificado."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List

from iot_machine_learning.domain.ports.analysis import (
    AnalysisContext,
    InputType,
    Perception,
    Signal,
)


class SimpleTypeDetector:
    """Detector de tipo simple basado en heurísticas."""
    
    def can_handle(self, data: Any) -> bool:
        """Verifica si puede manejar el dato."""
        return True
    
    def detect_type(self, data: Any) -> InputType:
        """Detecta tipo de input."""
        if isinstance(data, str):
            return InputType.TEXT
        elif isinstance(data, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in data):
                return InputType.TIMESERIES
            else:
                return InputType.MIXED
        elif isinstance(data, dict):
            return InputType.TABULAR
        else:
            return InputType.UNKNOWN


class SimpleDomainClassifier:
    """Clasificador de dominio simple basado en keywords."""
    
    def classify(self, data: Any, input_type: InputType) -> str:
        """Clasifica dominio del dato."""
        if input_type == InputType.TEXT and isinstance(data, str):
            data_lower = data.lower()
            if any(kw in data_lower for kw in ["error", "fallo", "crítico"]):
                return "infrastructure"
            elif any(kw in data_lower for kw in ["precio", "trading", "mercado"]):
                return "trading"
            elif any(kw in data_lower for kw in ["seguridad", "ataque", "vulnerabilidad"]):
                return "security"
        
        return "general"


class SimpleFeatureExtractor:
    """Extractor de features simple."""
    
    def extract(self, data: Any, input_type: InputType) -> Dict[str, Any]:
        """Extrae features del dato."""
        features = {}
        
        if input_type == InputType.TEXT and isinstance(data, str):
            features["word_count"] = len(data.split())
            features["char_count"] = len(data)
            features["avg_word_length"] = len(data) / max(len(data.split()), 1)
        
        elif input_type == InputType.TIMESERIES and isinstance(data, (list, tuple)):
            arr = np.array(data)
            features["mean"] = float(np.mean(arr))
            features["std"] = float(np.std(arr))
            features["min"] = float(np.min(arr))
            features["max"] = float(np.max(arr))
            features["n_points"] = len(data)
        
        return features


class SimplePerceptionCollector:
    """Colector de percepción simple para testing."""
    
    def __init__(self, input_type_name: str):
        """Inicializa colector."""
        self._input_type_name = input_type_name
    
    def collect(
        self,
        signal: Signal,
        context: AnalysisContext,
    ) -> List[Perception]:
        """Colecta percepciones del signal."""
        perceptions = []
        
        # Percepción basada en features
        if signal.features:
            score = 0.5  # Score neutral por defecto
            
            # Ajustar score basado en features
            if "mean" in signal.features:
                score = min(1.0, signal.features["mean"] / 100.0)
            elif "word_count" in signal.features:
                score = min(1.0, signal.features["word_count"] / 1000.0)
            
            perceptions.append(
                Perception(
                    perspective=f"{self._input_type_name}_analyzer",
                    score=score,
                    confidence=0.7,
                    evidence=signal.features,
                    metadata={"domain": signal.domain},
                )
            )
        
        return perceptions


class SimpleInhibitor:
    """Inhibidor simple basado en umbral de confianza."""
    
    def filter(self, perceptions: List[Perception]) -> List[Perception]:
        """Filtra percepciones con confianza < 0.3."""
        return [p for p in perceptions if p.confidence >= 0.3]


class SimpleFusion:
    """Fusionador simple con promedio ponderado."""
    
    def fuse(
        self,
        perceptions: List[Perception],
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """Fusiona percepciones con pesos usando numpy."""
        if not perceptions:
            return {
                "fused_score": 0.0,
                "confidence": 0.0,
                "method": "empty",
            }
        
        # Vectorizar operaciones
        scores = np.array([p.score for p in perceptions])
        confidences = np.array([p.confidence for p in perceptions])
        weight_values = np.array([
            weights.get(p.perspective, 1.0 / len(perceptions))
            for p in perceptions
        ])
        
        # Normalizar pesos
        if np.sum(weight_values) > 0:
            weight_values = weight_values / np.sum(weight_values)
        
        # Fusión ponderada
        fused_score = float(np.sum(scores * weight_values))
        fused_confidence = float(np.mean(confidences))
        
        # Seleccionar motor con mayor peso
        max_idx = int(np.argmax(weight_values))
        selected = perceptions[max_idx].perspective
        
        return {
            "fused_score": fused_score,
            "confidence": fused_confidence,
            "selected_engine": selected,
            "selection_reason": f"highest_weight: {weight_values[max_idx]:.3f}",
            "method": "weighted_average",
        }


class SimpleSeverityClassifier:
    """Clasificador de severidad simple basado en umbrales."""
    
    def classify(
        self,
        fused_score: float,
        domain: str,
        perceptions: List[Perception],
    ) -> str:
        """Clasifica severidad."""
        # Umbrales por dominio
        thresholds = {
            "infrastructure": {"critical": 0.75, "warning": 0.45},
            "security": {"critical": 0.65, "warning": 0.35},
            "trading": {"critical": 0.80, "warning": 0.50},
            "general": {"critical": 0.70, "warning": 0.40},
        }
        
        domain_thresholds = thresholds.get(domain, thresholds["general"])
        
        if fused_score >= domain_thresholds["critical"]:
            return "critical"
        elif fused_score >= domain_thresholds["warning"]:
            return "warning"
        else:
            return "info"
