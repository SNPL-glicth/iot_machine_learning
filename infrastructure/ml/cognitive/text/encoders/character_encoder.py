"""CharacterEncoder — codificación a nivel de caracteres.

Responsabilidad: convertir texto a vector numérico
basado en frecuencia ponderada de caracteres.

Máximo 95 líneas (SRP).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from iot_machine_learning.domain.ports.text_encoder_port import (
    EncodingVector,
    TextEncoderPort,
)
from .positional_weights import PositionalWeights


class CharacterEncoder(TextEncoderPort):
    """Encoder basado en frecuencia de caracteres con pesos posicionales."""
    
    def __init__(
        self,
        output_dim: int = 32,
        weights_path: str | None = None,
    ) -> None:
        """Inicializa encoder con configuración externa."""
        self._output_dim = output_dim
        self._pos_weights = PositionalWeights()
        self._char_weights = self._load_char_weights(weights_path)
    
    def _load_char_weights(self, path: str | None) -> Dict[str, float]:
        """Carga pesos por carácter desde JSON."""
        if path is None:
            base = Path(__file__).parent / "data"
            path = str(base / "character_weights.json")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("weights", {})
        except FileNotFoundError:
            return self._default_weights()
    
    def _default_weights(self) -> Dict[str, float]:
        """Pesos por defecto si no hay config."""
        # Vocales acentuadas = alta señal
        high = "áéíóúüñÁÉÍÓÚÜÑ"
        # Mayúsculas = media-alta
        medium = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # Minúsculas comunes = media
        low = "abcdefghijklmnopqrstuvwxyz"
        # Números y símbolos = baja
        numbers = "0123456789"
        
        weights = {}
        for c in high:
            weights[c] = 1.0
        for c in medium:
            weights[c] = 0.7
        for c in low:
            weights[c] = 0.5
        for c in numbers:
            weights[c] = 0.3
        return weights
    
    def encode(self, text: str) -> EncodingVector:
        """Codifica texto a vector fijo."""
        if not text:
            return EncodingVector(
                values=np.zeros(self._output_dim),
                dimension=self._output_dim,
                encoding_type="character"
            )
        
        # Limpiar texto
        clean = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑ]', '', text)
        
        # Calcular pesos por posición
        pos_weights = self._pos_weights.calculate_weights(len(clean))
        length_mult = self._pos_weights.get_length_multiplier(len(clean))
        
        # Acumular caracteres en buckets
        buckets: List[List[float]] = [[] for _ in range(self._output_dim)]
        
        for i, char in enumerate(clean):
            char_w = self._char_weights.get(char, 0.3)
            pos_w = pos_weights[i] if i < len(pos_weights) else 1.0
            combined = char_w * pos_w * length_mult
            
            # Distribuir en bucket usando hash
            bucket_idx = hash(char) % self._output_dim
            buckets[bucket_idx].append(combined)
        
        # Promediar buckets
        vector = np.array([
            sum(b) / len(b) if b else 0.0
            for b in buckets
        ], dtype=np.float32)
        
        # Normalizar L2
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return EncodingVector(
            values=vector,
            dimension=self._output_dim,
            encoding_type="character"
        )
    
    def encode_sequence(self, text: str) -> List[EncodingVector]:
        """Codifica por palabras."""
        words = re.findall(r'\b\w+\b', text.lower())
        return [self.encode(w) for w in words if w]
