"""Estados de conexión del proveedor (FASE 6).

Enumera los estados posibles de un WebSocket live para distinguir
entre estados normales y degradados, evitando generar predicciones
cuando el contexto está incompleto.

Estados:
- CONNECTED: WebSocket funcionando normalmente
- DEGRADED: Datos parciales o latencia alta
- DISCONNECTED: WebSocket caído
- RECONNECTING: Intentando reconectar
- RECOVERED: Reconexión exitosa (verificar datos)
"""

from __future__ import annotations

from enum import Enum


class ConnectionState(Enum):
    """Estado de conexión del proveedor live."""

    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    RECOVERED = "RECOVERED"

    def is_healthy(self) -> bool:
        """True si el estado permite generar predicciones."""
        return self in (ConnectionState.CONNECTED, ConnectionState.RECOVERED)

    def is_unhealthy(self) -> bool:
        """True si el estado NO permite generar predicciones."""
        return not self.is_healthy()
