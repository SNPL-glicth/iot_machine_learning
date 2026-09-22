"""Unit tests for ML Core IoT Ports (Fase 5.1).

Validates pure interface contracts, runtime checkability, and decoupling.
"""

from typing import Any, Dict, List, Optional
import pytest

from domain.ports.iot_ports import (
    ActuatorCommandPort,
    SensorDataPort,
    TelemetryStoragePort,
)


class DummySensorData(SensorDataPort):
    def load_series_window(self, series_id: str, limit: int = 500) -> Any:
        return [1.0, 2.0, 3.0]

    def list_active_series(self) -> List[str]:
        return ["series_1", "series_2"]

    def get_series_metadata(self, series_id: str) -> Dict[str, Any]:
        return {"units": "degC"}


class DummyTelemetryStorage(TelemetryStoragePort):
    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}

    def save_snapshot(self, engine_id: str, payload: Dict[str, Any]) -> bool:
        self.store[engine_id] = payload
        return True

    def load_snapshot(self, engine_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get(engine_id)

    def delete_snapshot(self, engine_id: str) -> bool:
        return self.store.pop(engine_id, None) is not None

    def save(self, engine_id: str, payload: Dict[str, Any]) -> bool:
        return self.save_snapshot(engine_id, payload)

    def load(self, engine_id: str) -> Optional[Dict[str, Any]]:
        return self.load_snapshot(engine_id)

    def delete(self, engine_id: str) -> bool:
        return self.delete_snapshot(engine_id)


class DummyActuatorCommand(ActuatorCommandPort):
    async def dispatch_command(
        self, actuator_id: str, setpoint: float, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        return True

    async def dispatch_execution(self, plan: Any) -> bool:
        return True

    async def trigger_emergency_flush(self, reason: str) -> None:
        pass

    def get_actuator_status(self, actuator_id: str) -> Optional[Dict[str, Any]]:
        return {"status": "ok", "setpoint": 50.0}


def test_sensor_data_port_protocol():
    dummy = DummySensorData()
    assert isinstance(dummy, SensorDataPort)
    assert dummy.list_active_series() == ["series_1", "series_2"]
    assert dummy.load_series_window("series_1") == [1.0, 2.0, 3.0]
    assert dummy.get_series_metadata("series_1") == {"units": "degC"}


def test_telemetry_storage_port_protocol():
    storage = DummyTelemetryStorage()
    assert isinstance(storage, TelemetryStoragePort)
    assert storage.save_snapshot("engine_test", {"param": 42}) is True
    assert storage.load_snapshot("engine_test") == {"param": 42}
    assert storage.delete_snapshot("engine_test") is True
    assert storage.load_snapshot("engine_test") is None


def test_telemetry_storage_legacy_aliases():
    storage = DummyTelemetryStorage()
    assert storage.save("engine_test", {"legacy": True}) is True
    assert storage.load("engine_test") == {"legacy": True}
    assert storage.delete("engine_test") is True


@pytest.mark.asyncio
async def test_actuator_command_port_protocol():
    actuator = DummyActuatorCommand()
    assert isinstance(actuator, ActuatorCommandPort)
    assert await actuator.dispatch_command("vfd_1", 75.0) is True
    assert await actuator.dispatch_execution(object()) is True
    await actuator.trigger_emergency_flush("test_flush")
    status = actuator.get_actuator_status("vfd_1")
    assert status == {"status": "ok", "setpoint": 50.0}
