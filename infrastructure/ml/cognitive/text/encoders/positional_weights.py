"""Cálculo de pesos posicionales para caracteres.

Responsabilidad: ponderar caracteres según posición
en el texto (inicio, medio, fin).
"""

from __future__ import annotations

from typing import List, Dict
import json
from pathlib import Path


class PositionalWeights:
    """Calcula pesos por posición en el texto.
    
    Lee configuración desde JSON externo.
    """
    
    def __init__(self, config_path: str | None = None) -> None:
        """Inicializa con configuración externa."""
        if config_path is None:
            base = Path(__file__).parent / "data"
            config_path = str(base / "positional_weights.json")
        
        self._config = self._load_config(config_path)
    
    def _load_config(self, path: str) -> Dict:
        """Carga config desde JSON."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Config por defecto si no existe archivo."""
        return {
            "start_ratio": 0.2,      # 20% inicial = alta prioridad
            "end_ratio": 0.2,        # 20% final = alta prioridad  
            "start_weight": 1.5,     # peso para inicio
            "middle_weight": 1.0,    # peso para medio
            "end_weight": 1.3,       # peso para fin
            "length_thresholds": {
                "short": 50,         # < 50 chars
                "medium": 200,       # < 200 chars
                "long": 1000         # >= 1000 chars
            }
        }
    
    def calculate_weights(self, text_length: int) -> List[float]:
        """Genera lista de pesos para cada posición.
        
        Args:
            text_length: Longitud del texto
            
        Returns:
            Lista de pesos [0..1] por posición
        """
        cfg = self._config
        start_cutoff = int(text_length * cfg["start_ratio"])
        end_start = int(text_length * (1 - cfg["end_ratio"]))
        
        weights = []
        for i in range(text_length):
            if i < start_cutoff:
                w = cfg["start_weight"]
            elif i >= end_start:
                w = cfg["end_weight"]
            else:
                w = cfg["middle_weight"]
            weights.append(w)
        
        # Normalizar
        max_w = max(weights) if weights else 1.0
        return [w / max_w for w in weights]
    
    def get_length_multiplier(self, text_length: int) -> float:
        """Multiplicador según longitud del texto.
        
        Returns:
            Factor para ajustar importancia por tamaño
        """
        th = self._config["length_thresholds"]
        if text_length < th["short"]:
            return 0.8  # Textos cortos menos informativos
        elif text_length < th["medium"]:
            return 1.0
        elif text_length < th["long"]:
            return 1.2
        else:
            return 1.5  # Textos largos más detallados
