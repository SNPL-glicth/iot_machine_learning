"""Pure domain-agnostic ports for IoT interactions in ML Core.

Strict Hexagonal Architecture: These interfaces define the contracts that
external adapters (MQTT, Redis, SQL, PLCs) must implement to communicate
with the ML Core.

The domain layer defines these contracts; infrastructure adapters implement them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class SensorDataPort(Protocol):
    """Port for streaming and historical sensor data ingestion.
    
    Decouples ML ingestion pipelines and analyzers from physical sensor protocols.
    """

    def load_series_window(
        self, series_id: str, limit: int = 500
    ) -> Any:
        """Loads a recent historical window of observations for a series."""
        ...

    def list_active_series(self) -> List[str]:
        """Lists identifiers of all currently reporting series/sensors."""
        ...

    def get_series_metadata(self, series_id: str) -> Dict[str, Any]:
        """Retrieves static/dynamic metadata for a series (e.g. units, bounds)."""
        ...


@runtime_checkable
class TelemetryStoragePort(Protocol):
    """Port for persisting and retrieving ML engine state snapshots.
    
    Provides key-value persistence contract for learning engines, state machines,
    and adaptive parameters without binding to Redis, disk, or databases.
    """

    def save_snapshot(self, engine_id: str, payload: Dict[str, Any]) -> bool:
        """Persists an atomic engine state snapshot. Returns True on success."""
        ...

    def load_snapshot(self, engine_id: str) -> Optional[Dict[str, Any]]:
        """Loads the latest snapshot for an engine, or None if not found."""
        ...

    def delete_snapshot(self, engine_id: str) -> bool:
        """Deletes stored state for an engine. Returns True if deleted."""
        ...

    # Aliases for seamless drop-in interoperability with existing MLStateStore
    def save(self, engine_id: str, payload: Dict[str, Any]) -> bool:
        """Persist alias for MLStateStore compatibility."""
        ...

    def load(self, engine_id: str) -> Optional[Dict[str, Any]]:
        """Load alias for MLStateStore compatibility."""
        ...

    def delete(self, engine_id: str) -> bool:
        """Delete alias for MLStateStore compatibility."""
        ...


@runtime_checkable
class ActuatorCommandPort(Protocol):
    """Port for dispatching actuation decisions to physical or simulated IoT devices.
    
    Decouples ML decision outputs (ExecutionPlan, setpoints) from hardware drivers,
    PLCs, and communication protocols (MQTT, Modbus, OPC-UA).
    """

    async def dispatch_command(
        self,
        actuator_id: str,
        setpoint: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Dispatches an individual setpoint command to a target actuator."""
        ...

    async def dispatch_execution(self, plan: Any) -> bool:
        """Dispatches a full orchestrated decision plan to the actuation layer."""
        ...

    async def trigger_emergency_flush(self, reason: str) -> None:
        """Triggers emergency safety protocols, failsafe positions, or shutoffs."""
        ...

    def get_actuator_status(self, actuator_id: str) -> Optional[Dict[str, Any]]:
        """Queries health, connectivity, and current setpoint of an actuator."""
        ...
