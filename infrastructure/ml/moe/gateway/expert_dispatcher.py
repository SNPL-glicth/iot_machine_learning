"""ExpertDispatcher — Ejecución de expertos seleccionados.

Extraído de MoEGateway como servicio independiente siguiendo SRP.
Soporte de dispatch paralelo con ThreadPoolExecutor y timeout por experto.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List
import logging

from iot_machine_learning.domain.ports.expert_port import ExpertPort, ExpertOutput
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from ..registry.expert_registry import ExpertRegistry

logger = logging.getLogger(__name__)


class ExpertDispatcher:
    """Servicio de dispatch y ejecución de expertos.

    Responsabilidad única: ejecutar expertos seleccionados y recolectar outputs.
    Implementa fail-silent: si un experto falla o hace timeout, se omite.

    Attributes:
        _registry: Catálogo de expertos disponibles.
        _timeout_ms: Timeout por experto en milisegundos.
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        timeout_ms: int = 200,
    ) -> None:
        """Inicializa el dispatcher.

        Args:
            registry: Registro con los expertos disponibles.
            timeout_ms: Timeout por experto (default: 200ms).
        """
        self._registry = registry
        self._timeout_ms = timeout_ms

    @property
    def timeout_ms(self) -> int:
        return self._timeout_ms

    @timeout_ms.setter
    def timeout_ms(self, value: int) -> None:
        self._timeout_ms = value

    def dispatch(
        self,
        expert_ids: List[str],
        window: SensorWindow
    ) -> Dict[str, ExpertOutput]:
        """Ejecuta expertos seleccionados en paralelo.

        Args:
            expert_ids: IDs de expertos a ejecutar.
            window: Ventana de datos.

        Returns:
            Dict {expert_id: ExpertOutput} con resultados exitosos.

        Note:
            Expertos que no existen, no pueden manejar la ventana,
            lanzan excepciones, o hacen timeout, son omitidos (fail-silent).
        """
        # Filtrar expertos válidos primero
        valid_experts = []
        for expert_id in expert_ids:
            expert = self._registry.get(expert_id)
            if expert is None:
                continue
            if not expert.can_handle(window):
                continue
            valid_experts.append((expert_id, expert))

        if not valid_experts:
            return {}

        outputs = {}
        timeout_s = self._timeout_ms / 1000.0

        with ThreadPoolExecutor(max_workers=len(valid_experts)) as executor:
            future_map = {
                executor.submit(expert.predict, window): expert_id
                for expert_id, expert in valid_experts
            }

            for future, expert_id in future_map.items():
                try:
                    output = future.result(timeout=timeout_s)
                    outputs[expert_id] = output
                except FutureTimeoutError:
                    logger.warning(
                        "expert_timeout",
                        extra={"expert_id": expert_id, "timeout_ms": self._timeout_ms},
                    )
                except Exception:
                    logger.debug(f"Expert {expert_id} failed, skipping")

        return outputs
    
