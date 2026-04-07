"""Adaptadores comunes para detección y clasificación."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict

from iot_machine_learning.domain.ports.analysis import InputType


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
