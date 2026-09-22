"""Unit tests for Consolidated IoT Actuator and Drift Adapters.

Tests the lean, decoupled actuator and drift adapters in infrastructure/adapters/iot/.
"""

import pytest

from infrastructure.adapters.iot import (
    ActuatorCommand,
    ActuatorConfig,
    CallbackActuatorClient,
    DriftSensorAdapter,
    IoTActuatorHandler,
    IoTDriftSensorAdapter,
    MockActuatorClient,
    create_drift_detector,
)
from domain.entities.rosa_roja.execution import (
    ActionEnvelope,
    ExecutionPlan,
)
from domain.ports.rosa_roja.drift_sensor import DriftSensorPort
from domain.ports.rosa_roja.execution_port import ExecutionPort
from domain.ports.iot_ports import ActuatorCommandPort


@pytest.mark.asyncio
async def test_actuator_handler_dispatch_and_execution():
    client = MockActuatorClient()
    configs = {
        "actuator_1": ActuatorConfig(
            actuator_id="actuator_1",
            min_setpoint=0.0,
            max_setpoint=100.0,
        )
    }
    handler = IoTActuatorHandler(actuator_client=client, actuators=configs)

    assert isinstance(handler, ExecutionPort)
    assert isinstance(handler, ActuatorCommandPort)

    # 1. Direct command dispatch
    assert await handler.dispatch_command("actuator_1", 75.5) is True
    assert len(client.commands_sent) == 1
    assert client.commands_sent[0].actuator_id == "actuator_1"
    assert client.commands_sent[0].setpoint == 75.5

    # 2. ExecutionPlan: HOLD
    hold_plan = ExecutionPlan.HOLD(reason="RISK_VETO")
    assert await handler.dispatch_execution(hold_plan) is True

    # 3. ExecutionPlan: EXECUTE
    envelope = ActionEnvelope(magnitude=0.6, bounds={}, max_steps=5, metadata={"primary_actuator": "actuator_1"})
    exec_plan = ExecutionPlan.EXECUTE(trajectory=None, confidence=0.8, envelope=envelope)
    assert await handler.dispatch_execution(exec_plan) is True
    assert client.commands_sent[-1].setpoint == 60.0

    # 4. Emergency flush
    flush_plan = ExecutionPlan.EMERGENCY_FLUSH(reason="OUT_OF_BOUNDS")
    assert await handler.dispatch_execution(flush_plan) is True
    assert handler.is_emergency_active() is True
    assert handler.get_emergency_reason() == "OUT_OF_BOUNDS"
    assert "ALL" in client.emergency_stops

    handler.clear_emergency()
    assert handler.is_emergency_active() is False


def test_drift_sensor_adapter_protocol_and_monitoring():
    adapter = IoTDriftSensorAdapter(
        name="telemetry_drift",
        channels=["temperature", "pressure"],
        detector_type="page_hinkley",
    )
    assert isinstance(adapter, DriftSensorPort)
    assert DriftSensorAdapter is IoTDriftSensorAdapter

    adapter.update(25.0, 25.0)
    assert adapter.get_drift_score() >= 0.0

    scores = adapter.get_channel_scores()
    assert "temperature" in scores and "pressure" in scores

    # Export & Import roundtrip
    state = adapter.export_state()
    assert state["schema_version"] == 1
    adapter.import_state(state)


@pytest.mark.asyncio
async def test_callback_actuator_client_sync_and_async():
    dispatched_commands = []
    stopped_actuators = []

    async def async_cmd_callback(cmd: ActuatorCommand):
        dispatched_commands.append(cmd)
        return True

    def sync_emergency_callback(target: str):
        stopped_actuators.append(target)
        return True

    client = CallbackActuatorClient(
        command_callback=async_cmd_callback,
        emergency_callback=sync_emergency_callback,
    )

    cmd = ActuatorCommand(actuator_id="pump_1", setpoint=42.0)
    assert await client.send_command(cmd) is True
    assert len(dispatched_commands) == 1
    assert dispatched_commands[0].actuator_id == "pump_1"

    assert await client.emergency_stop("pump_1") is True
    assert stopped_actuators == ["pump_1"]

    assert await client.emergency_stop_all() is True
    assert stopped_actuators == ["pump_1", "ALL"]
