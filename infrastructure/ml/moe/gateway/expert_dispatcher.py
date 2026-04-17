"""ExpertDispatcher — Ejecución de expertos seleccionados.

Extraído de MoEGateway como servicio independiente siguiendo SRP.
"""

from __future__ import annotations

from typing import Dict, List
import logging

from iot_machine_learning.domain.ports.expert_port import ExpertPort, ExpertOutput
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from ..registry.expert_registry import ExpertRegistry

logger = logging.getLogger(__name__)


class ExpertDispatcher:
    """Servicio de dispatch y ejecución de expertos.
    
    Responsabilidad única: ejecutar expertos seleccionados y recolectar outputs.
    Implementa fail-silent: si un experto falla, se omite sin romper el flujo.
    
    Attributes:
        _registry: Catálogo de expertos disponibles.
    """
    
    def __init__(self, registry: ExpertRegistry) -> None:
        """Inicializa el dispatcher.
        
        Args:
            registry: Registro con los expertos disponibles.
        """
        self._registry = registry
    
    def dispatch(
        self,
        expert_ids: List[str],
        window: SensorWindow
    ) -> Dict[str, ExpertOutput]:
        """Ejecuta expertos seleccionados.
        
        Args:
            expert_ids: IDs de expertos a ejecutar.
            window: Ventana de datos.
            
        Returns:
            Dict {expert_id: ExpertOutput} con resultados exitosos.
            
        Note:
            Expertos que no existen, no pueden manejar la ventana,
            o lanzan excepciones, son omitidos (fail-silent).
        """
        outputs = {}
        
        for expert_id in expert_ids:
            expert = self._registry.get(expert_id)
            if expert is None:
                continue
            
            if not expert.can_handle(window):
                continue
            
            try:
                output = expert.predict(window)
                outputs[expert_id] = output
            except Exception:
                # Skip failed experts (fail-silent)
                logger.debug(f"Expert {expert_id} failed, skipping")
                continue
        
        return outputs
    
    def can_any_expert_handle(self, n_points: int) -> bool:
        """Verifica si al menos un experto puede manejar n_points.
        
        Args:
            n_points: Número de puntos disponibles.
            
        Returns:
            True si existe al menos un experto capaz.
        """
        # Obtener todos los expertos del registry
        expert_ids = self._registry.list_experts()
        for expert_id in expert_ids:
            expert = self._registry.get(expert_id)
            if expert and expert.can_handle_n_points(n_points):
                return True
        return False
