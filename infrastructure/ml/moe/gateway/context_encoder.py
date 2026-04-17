"""ContextEncoderService — Codificación de ventanas a vectores de contexto.

Extraído de MoEGateway como servicio independiente siguiendo SRP.
"""

from __future__ import annotations

from typing import List

from iot_machine_learning.domain.model.context_vector import ContextVector
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow


class ContextEncoderService:
    """Servicio de codificación de ventanas temporales a ContextVector.
    
    Responsabilidad única: extraer features de la señal y clasificar régimen.
    
    Attributes:
        default_domain: Dominio por defecto para contextos.
    """
    
    def __init__(self, default_domain: str = "iot") -> None:
        """Inicializa el encoder.
        
        Args:
            default_domain: Dominio por defecto (iot, finance, healthcare).
        """
        self._default_domain = default_domain
    
    def encode(self, window: SensorWindow) -> ContextVector:
        """Codifica ventana a vector de contexto.
        
        Extrae features básicos de la señal y clasifica el régimen
        en stable, trending o volatile.
        
        Args:
            window: Ventana de lecturas del sensor.
            
        Returns:
            ContextVector con features y clasificación de régimen.
        """
        # Extraer valores
        values = [r.value for r in window.readings]
        
        if not values:
            return ContextVector(
                regime="stable",
                domain=self._default_domain,
                n_points=0,
                signal_features={},
            )
        
        # Calcular features básicos
        mean_val = sum(values) / len(values)
        
        # Estimación de std
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_val = variance ** 0.5
        
        # Estimación de slope (diferencia últimos puntos)
        if len(values) >= 2:
            slope = values[-1] - values[-2]
        else:
            slope = 0.0
        
        # Clasificación de régimen
        if std_val / abs(mean_val) > 0.2 if mean_val != 0 else False:
            regime = "volatile"
        elif abs(slope) > std_val * 0.5:
            regime = "trending"
        else:
            regime = "stable"
        
        return ContextVector(
            regime=regime,
            domain=self._default_domain,
            n_points=len(values),
            signal_features={
                "mean": mean_val,
                "std": std_val,
                "slope": slope,
            },
        )
