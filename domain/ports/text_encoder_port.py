"""Puerto abstracto para encoders de texto.

Hexagonal architecture: el dominio define el contrato,
la infraestructura provee implementaciones concretas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class EncodingVector:
    """Vector de encoding con metadatos."""
    values: np.ndarray
    dimension: int
    encoding_type: str  # "tfidf", "character", "hybrid"


class TextEncoderPort(ABC):
    """Contrato para encoders de texto.
    
    Implementaciones:
    - WordEncoder: TF-IDF basado en vocabulario
    - CharacterEncoder: Frecuencia de caracteres + posición
    """
    
    @abstractmethod
    def encode(self, text: str) -> EncodingVector:
        """Codifica texto a vector numérico.
        
        Args:
            text: Texto de entrada
            
        Returns:
            EncodingVector con dimension fija
        """
        ...
    
    @abstractmethod
    def encode_sequence(self, text: str) -> List[EncodingVector]:
        """Codifica secuencialmente (por palabras o n-grams).
        
        Args:
            text: Texto de entrada
            
        Returns:
            Lista de vectores por unidad
        """
        ...
