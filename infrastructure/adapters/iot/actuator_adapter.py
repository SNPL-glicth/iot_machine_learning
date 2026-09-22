"""Consolidated IoT Actuator Adapter.

Pure and lightweight bridge between Rosa Roja decisions and IoT actuators.
Implements ExecutionPort and ActuatorCommandPort without domain hardware coupling.
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from domain.entities.rosa_roja.execution import ExecutionPlan
from domain.ports.rosa_roja.execution_port import ExecutionPort
from domain.ports.iot_ports import ActuatorCommandPort

logger = logging.getLogger(__name__)


@dataclass
class ActuatorCommand:
    """Setpoint command dispatched to an actuator."""
    actuator_id: str
    setpoint: float
    timestamp: float = 0.0
    source: str = "rosa_roja"


@dataclass
class ActuatorConfig:
    """Minimal configuration for an actuator."""
    actuator_id: str
    device_id: str = "default"
    min_setpoint: float = 0.0
    max_setpoint: float = 100.0
    mqtt_topic: Optional[str] = None


class MockActuatorClient:
    """Mock client for offline and test execution."""

    def __init__(self) -> None:
        self.commands_sent: List[ActuatorCommand] = []
        self.emergency_stops: List[str] = []

    async def send_command(self, command: ActuatorCommand) -> bool:
        self.commands_sent.append(command)
        logger.info("[ACTUATOR] %s = %.2f", command.actuator_id, command.setpoint)
        return True

    async def emergency_stop(self, actuator_id: str) -> bool:
        self.emergency_stops.append(actuator_id)
        logger.warning("[ACTUATOR] Emergency stop: %s", actuator_id)
        return True

    async def emergency_stop_all(self) -> bool:
        self.emergency_stops.append("ALL")
        logger.warning("[ACTUATOR] Emergency stop ALL")
        return True


class CallbackActuatorClient:
    """Dispatches actuator commands and emergency events via user-provided callbacks.

    Enables seamless real-world integration (e.g. MQTT publishing, Redis streams,
    or HTTP webhooks) without coupled hardware libraries in core adapter layers.
    """

    def __init__(
        self,
        command_callback: Optional[Callable[[ActuatorCommand], Any]] = None,
        emergency_callback: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._command_callback = command_callback
        self._emergency_callback = emergency_callback
        self.commands_sent: List[ActuatorCommand] = []
        self.emergency_stops: List[str] = []

    async def send_command(self, command: ActuatorCommand) -> bool:
        self.commands_sent.append(command)
        if self._command_callback is not None:
            res = self._command_callback(command)
            if inspect.isawaitable(res):
                res = await res
            return bool(res) if res is not None else True
        logger.info("[CALLBACK_ACTUATOR] Dispatched %s = %.2f", command.actuator_id, command.setpoint)
        return True

    async def emergency_stop(self, actuator_id: str) -> bool:
        self.emergency_stops.append(actuator_id)
        if self._emergency_callback is not None:
            res = self._emergency_callback(actuator_id)
            if inspect.isawaitable(res):
                res = await res
            return bool(res) if res is not None else True
        logger.warning("[CALLBACK_ACTUATOR] Emergency stop %s", actuator_id)
        return True

    async def emergency_stop_all(self) -> bool:
        self.emergency_stops.append("ALL")
        if self._emergency_callback is not None:
            res = self._emergency_callback("ALL")
            if inspect.isawaitable(res):
                res = await res
            return bool(res) if res is not None else True
        logger.warning("[CALLBACK_ACTUATOR] Emergency stop ALL")
        return True


class IoTActuatorHandler(ExecutionPort, ActuatorCommandPort):
    """Orchestrates setpoints and emergency stops for IoT actuators."""

    def __init__(
        self,
        actuator_client: Optional[Any] = None,
        actuators: Optional[Dict[str, ActuatorConfig]] = None,
        safety_limits: Optional[Dict[str, float]] = None,
        default_rate_limit: float = 10.0,
        device_id: str = "iot_gateway",
    ) -> None:
        self._client = actuator_client or MockActuatorClient()
        self._actuators = actuators or {}
        self._device_id = device_id
        self._last_setpoints: Dict[str, float] = {}
        self._emergency_active = False
        self._emergency_reason: Optional[str] = None

    async def dispatch_execution(self, plan: ExecutionPlan) -> bool:
        """Dispatches an orchestrated decision plan to the actuation layer."""
        if plan.action == "HOLD":
            logger.info("HOLD: Maintaining current setpoints")
            return True
        if plan.action == "EMERGENCY_FLUSH":
            reason = plan.veto_details.get("reason", "unknown") if plan.veto_details else "unknown"
            await self.trigger_emergency_flush(reason)
            return True
        if plan.action == "EXECUTE":
            return await self._handle_execute(plan)
        return False

    async def _handle_execute(self, plan: ExecutionPlan) -> bool:
        if plan.envelope is None:
            return False
        primary_id = plan.envelope.metadata.get("primary_actuator") or next(iter(self._actuators), "actuator_1")
        cfg = self._actuators.get(primary_id, ActuatorConfig(primary_id))
        sp = cfg.min_setpoint + (plan.envelope.magnitude * (cfg.max_setpoint - cfg.min_setpoint))
        sp = max(cfg.min_setpoint, min(cfg.max_setpoint, sp))
        return await self.dispatch_command(primary_id, sp)

    async def dispatch_command(
        self, actuator_id: str, setpoint: float, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Dispatches setpoint float directly to the specified actuator."""
        cmd = ActuatorCommand(actuator_id=actuator_id, setpoint=setpoint, timestamp=time.time())
        ok = await self._client.send_command(cmd)
        if ok:
            self._last_setpoints[actuator_id] = setpoint
        return ok

    async def trigger_emergency_flush(self, reason: str) -> None:
        """Fires emergency safety shutdown across all connected actuators."""
        logger.critical("EMERGENCY_FLUSH: %s", reason)
        self._emergency_active = True
        self._emergency_reason = reason
        await self._client.emergency_stop_all()
        self._last_setpoints.clear()

    def is_emergency_active(self) -> bool:
        return self._emergency_active

    def get_emergency_reason(self) -> Optional[str]:
        return self._emergency_reason

    def clear_emergency(self) -> None:
        self._emergency_active = False
        self._emergency_reason = None

    def get_current_setpoints(self) -> Dict[str, float]:
        return self._last_setpoints.copy()

    def get_actuator_status(self, actuator_id: str) -> Optional[Dict[str, Any]]:
        return {
            "actuator_id": actuator_id,
            "current_setpoint": self._last_setpoints.get(actuator_id),
            "emergency_active": self._emergency_active,
        }
