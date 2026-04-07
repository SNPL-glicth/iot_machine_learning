"""Puertos para manejo de series temporales."""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class SlidingWindowPort(ABC, Generic[T]):
    """Puerto de acceso a ventanas deslizantes.
    
    Implementaciones deben ser thread-safe.
    """
    
    @abstractmethod
    def append(
        self,
        series_id: str,
        value: T,
        timestamp: Optional[float] = None,
    ) -> int:
        """Agrega punto a la ventana.
        
        Returns:
            Número de puntos en la ventana
        """
        ...
    
    @abstractmethod
    def get(self, series_id: str) -> Optional[List[Tuple[float, T]]]:
        """Obtiene ventana como lista de (timestamp, value).
        
        Returns:
            Lista ordenada por timestamp, o None si no existe
        """
        ...
    
    @abstractmethod
    def get_values(self, series_id: str) -> Optional[List[T]]:
        """Obtiene solo valores sin timestamps.
        
        Returns:
            Lista de valores, o None si no existe
        """
        ...
    
    @abstractmethod
    def size(self, series_id: str) -> int:
        """Número de puntos en la ventana.
        
        Returns:
            Tamaño de la ventana (0 si no existe)
        """
        ...
