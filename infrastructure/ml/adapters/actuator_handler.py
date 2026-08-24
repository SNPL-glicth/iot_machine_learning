"""IoT Actuator Handler for Rosa Roja - Implements ExecutionPort.

Maps ExecutionPlan/ActionEnvelope to IoT actuator commands (PLC setpoints, MQTT, etc.).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal
from enum import Enum

from core.orchestration.rosa_roja.ports.execution_port import ExecutionPort
from core.orchestration.rosa_roja.domain.execution import ExecutionPlan, ActionEnvelope

logger = logging.getLogger(__name__)


class ActuatorType(Enum):
    """Types of IoT actuators."""
    COMPRESSOR_VFD = "compressor_vfd"           # Variable Frequency Drive - speed control
    CHILLED_WATER_VALVE = "chilled_water_valve"  # Modulating valve 0-100%
    CONDENSER_FAN = "condenser_fan"              # Fan speed control
    EXPANSION_VALVE = "expansion_valve"          # Electronic expansion valve
    HEATER = "heater"                            # Electric heater
    GENERIC = "generic"                          # Generic setpoint


@dataclass
class ActuatorConfig:
    """Configuration for a single actuator."""
    actuator_id: str
    actuator_type: ActuatorType
    device_id: str
    min_setpoint: float = 0.0
    max_setpoint: float = 100.0
    unit: str = "%"
    safety_min: float = 0.0
    safety_max: float = 100.0
    rate_limit: float = 10.0          # Max change per second (%/s)
    mqtt_topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActuatorCommand:
    """Command to send to an actuator."""
    actuator_id: str
    setpoint: float
    timestamp: float
    source: str = "rosa_roja"
    priority: int = 100  # Lower = higher priority


class ActuatorClient:
    """Abstract actuator client - implement for MQTT, OPC-UA, Modbus, etc."""
    
    async def send_command(self, command: ActuatorCommand) -> bool:
        """Send command to actuator. Returns True if successful."""
        raise NotImplementedError
    
    async def emergency_stop(self, actuator_id: str) -> bool:
        """Emergency stop for specific actuator."""
        raise NotImplementedError
    
    async def emergency_stop_all(self) -> bool:
        """Emergency stop all actuators."""
        raise NotImplementedError


class MockActuatorClient(ActuatorClient):
    """Mock actuator client for testing/development."""
    
    def __init__(self):
        self.commands_sent = []
        self.emergency_stops = []
    
    async def send_command(self, command: ActuatorCommand) -> bool:
        self.commands_sent.append(command)
        logger.info(f"[MOCK] Actuator command: {command.actuator_id} = {command.setpoint:.2f}{command.unit if hasattr(command, 'unit') else '%'}")
        return True
    
    async def emergency_stop(self, actuator_id: str) -> bool:
        self.emergency_stops.append(actuator_id)
        logger.warning(f"[MOCK] Emergency stop: {actuator_id}")
        return True
    
    async def emergency_stop_all(self) -> bool:
        self.emergency_stops.append("ALL")
        logger.warning("[MOCK] Emergency stop ALL actuators")
        return True


class IoTActuatorHandler(ExecutionPort):
    """IoT Actuator Handler implementing ExecutionPort.
    
    Translates Rosa Roja ExecutionPlan into actuator commands for
    industrial IoT equipment (chillers, compressors, HVAC, etc.).
    
    Action Mapping:
    - EXECUTE: Apply ActionEnvelope magnitude/bounds as actuator setpoints
    - HOLD: Maintain current setpoints (no-op)
    - EMERGENCY_FLUSH: Trigger safety shutdown sequence
    
    Example:
        handler = IoTActuatorHandler(
            actuator_client=mqtt_client,
            actuators={
                "comp_vfd": ActuatorConfig("comp_vfd", ActuatorType.COMPRESSOR_VFD, "chiller_01", 0, 100, "%", mqtt_topic="chiller/01/vfd/set"),
                "chw_valve": ActuatorConfig("chw_valve", ActuatorType.CHILLED_WATER_VALVE, "chiller_01", 0, 100, "%", mqtt_topic="chiller/01/valve/set"),
            },
            safety_limits={"temp_max": 95, "pressure_max": 30}
        )
        await handler.dispatch_execution(plan)
    """
    
    def __init__(
        self,
        actuator_client: ActuatorClient,
        actuators: Dict[str, ActuatorConfig],
        safety_limits: Optional[Dict[str, float]] = None,
        default_rate_limit: float = 10.0,
        device_id: str = "iot_gateway",
    ):
        """
        Args:
            actuator_client: Client for sending commands (MQTT, OPC-UA, etc.)
            actuators: Dict of actuator_id -> ActuatorConfig
            safety_limits: Hard safety limits (temp_max, pressure_max, etc.)
            default_rate_limit: Default rate limit for setpoint changes (%/s)
            device_id: Identifier for this gateway/handler
        """
        self._client = actuator_client
        self._actuators = actuators
        self._safety_limits = safety_limits or {}
        self._default_rate_limit = default_rate_limit
        self._device_id = device_id
        
        # Track last setpoints for rate limiting
        self._last_setpoints: Dict[str, float] = {}
        self._last_update_time: Dict[str, float] = {}
        
        # Emergency state
        self._emergency_active = False
        self._emergency_reason: Optional[str] = None
        
        logger.info(f"IoTActuatorHandler initialized for {device_id} with {len(actuators)} actuators")
    
    async def dispatch_execution(self, plan: ExecutionPlan) -> bool:
        """
        Process ExecutionPlan into actuator commands.
        
        Args:
            plan: ExecutionPlan from Rosa Roja Engine
            
        Returns:
            True if all commands dispatched successfully
        """
        logger.debug(f"Dispatching execution: action={plan.action}, confidence={plan.global_confidence:.3f}")
        
        if plan.action == "HOLD":
            return await self._handle_hold(plan)
        
        elif plan.action == "EMERGENCY_FLUSH":
            return await self._handle_emergency_flush(plan)
        
        elif plan.action == "EXECUTE":
            return await self._handle_execute(plan)
        
        else:
            logger.warning(f"Unknown action: {plan.action}")
            return False
    
    async def _handle_hold(self, plan: ExecutionPlan) -> bool:
        """HOLD - maintain current setpoints, log reason."""
        reason = plan.veto_details.get("reason", "unknown") if plan.veto_details else "unknown"
        logger.info(f"HOLD: Maintaining current setpoints (reason: {reason})")
        return True
    
    async def _handle_emergency_flush(self, plan: ExecutionPlan) -> bool:
        """EMERGENCY_FLUSH - trigger safety shutdown sequence."""
        reason = plan.veto_details.get("reason", "unknown") if plan.veto_details else "unknown"
        logger.critical(f"EMERGENCY_FLUSH triggered: {reason}")
        
        self._emergency_active = True
        self._emergency_reason = reason
        
        # 1. Emergency stop all actuators
        success = await self._client.emergency_stop_all()
        
        # 2. Reset last setpoints tracking
        self._last_setpoints.clear()
        self._last_update_time.clear()
        
        if success:
            logger.info("Emergency shutdown completed successfully")
        else:
            logger.error("Emergency shutdown partially failed")
        
        return success
    
    async def _handle_execute(self, plan: ExecutionPlan) -> bool:
        """EXECUTE - apply ActionEnvelope as actuator setpoints."""
        if plan.envelope is None:
            logger.warning("EXECUTE plan missing ActionEnvelope")
            return False
        
        envelope = plan.envelope
        magnitude = envelope.magnitude  # 0.0 - 1.0
        bounds = envelope.bounds        # e.g., {"stop_pct": 0.02, "target_pct": 0.05}
        max_steps = envelope.max_steps
        
        logger.info(f"EXECUTE: magnitude={magnitude:.2f}, bounds={bounds}, max_steps={max_steps}")
        
        # Generate actuator commands from envelope
        commands = self._envelope_to_commands(envelope, plan)
        
        if not commands:
            logger.warning("No actuator commands generated from envelope")
            return False
        
        # Apply safety checks and rate limits
        safe_commands = self._apply_safety_and_rate_limits(commands)
        
        # Dispatch all commands
        all_success = True
        for cmd in safe_commands:
            success = await self._client.send_command(cmd)
            all_success = all_success and success
            
            # Update tracking
            self._last_setpoints[cmd.actuator_id] = cmd.setpoint
            self._last_update_time[cmd.actuator_id] = cmd.timestamp
        
        return all_success
    
    def _envelope_to_commands(self, envelope: ActionEnvelope, plan: ExecutionPlan) -> list[ActuatorCommand]:
        """Convert ActionEnvelope to actuator commands.
        
        This is the key mapping logic - customize per application.
        Default implementation maps magnitude to primary actuator.
        """
        commands = []
        timestamp = time.time()
        
        # Get primary actuator (first one by default, or from metadata)
        primary_actuator_id = envelope.metadata.get("primary_actuator")
        if not primary_actuator_id and self._actuators:
            primary_actuator_id = next(iter(self._actuators))
        
        if not primary_actuator_id:
            logger.warning("No actuators configured")
            return []
        
        primary_config = self._actuators[primary_actuator_id]
        
        # Map magnitude (0-1) to actuator range
        magnitude = envelope.magnitude
        setpoint_range = primary_config.max_setpoint - primary_config.min_setpoint
        base_setpoint = primary_config.min_setpoint + (magnitude * setpoint_range)
        
        # Apply bounds if present (e.g., stop_pct, target_pct as safety margins)
        bounds = envelope.bounds
        if "target_pct" in bounds:
            # Use target_pct as max allowed setpoint change
            max_change = bounds["target_pct"] * setpoint_range
            last_sp = self._last_setpoints.get(primary_actuator_id, base_setpoint)
            base_setpoint = max(base_setpoint, last_sp - max_change)
            base_setpoint = min(base_setpoint, last_sp + max_change)
        
        # Clamp to safety limits
        base_setpoint = max(primary_config.safety_min, min(primary_config.safety_max, base_setpoint))
        
        # Create command
        cmd = ActuatorCommand(
            actuator_id=primary_actuator_id,
            setpoint=base_setpoint,
            timestamp=timestamp,
            source="rosa_roja",
            priority=100 - int(magnitude * 50),  # Higher magnitude = higher priority
        )
        commands.append(cmd)
        
        # Add secondary actuators from metadata
        secondary_actuators = envelope.metadata.get("secondary_actuators", {})
        for actuator_id, config in secondary_actuators.items():
            if actuator_id in self._actuators:
                act_config = self._actuators[actuator_id]
                setpoint = act_config.min_setpoint + (config.get("magnitude", magnitude) * (act_config.max_setpoint - act_config.min_setpoint))
                setpoint = max(act_config.safety_min, min(act_config.safety_max, setpoint))
                
                sec_cmd = ActuatorCommand(
                    actuator_id=actuator_id,
                    setpoint=setpoint,
                    timestamp=timestamp,
                    source="rosa_roja",
                    priority=100,
                )
                commands.append(sec_cmd)
        
        return commands
    
    def _apply_safety_and_rate_limits(self, commands: list[ActuatorCommand]) -> list[ActuatorCommand]:
        """Apply safety limits and rate limiting to commands."""
        safe_commands = []
        now = time.time()
        
        for cmd in commands:
            config = self._actuators.get(cmd.actuator_id)
            if not config:
                logger.warning(f"No config for actuator {cmd.actuator_id}, skipping")
                continue
            
            # Rate limiting
            if cmd.actuator_id in self._last_setpoints:
                last_sp = self._last_setpoints[cmd.actuator_id]
                last_time = self._last_update_time.get(cmd.actuator_id, now)
                dt = now - last_time
                rate_limit = config.rate_limit if config.rate_limit > 0 else self._default_rate_limit
                max_change = rate_limit * dt
                
                if abs(cmd.setpoint - last_sp) > max_change:
                    # Clamp to rate limit
                    direction = 1 if cmd.setpoint > last_sp else -1
                    cmd.setpoint = last_sp + direction * max_change
                    logger.debug(f"Rate limited {cmd.actuator_id}: setpoint clamped to {cmd.setpoint:.2f}")
            
            # Safety clamping
            if cmd.setpoint < config.safety_min:
                logger.warning(f"Safety min violated for {cmd.actuator_id}: {cmd.setpoint:.2f} < {config.safety_min}")
                cmd.setpoint = config.safety_min
            elif cmd.setpoint > config.safety_max:
                logger.warning(f"Safety max violated for {cmd.actuator_id}: {cmd.setpoint:.2f} > {config.safety_max}")
                cmd.setpoint = config.safety_max
            
            # Hard safety limits
            for limit_name, limit_value in self._safety_limits.items():
                if limit_name.endswith("_max") and cmd.setpoint > limit_value:
                    logger.warning(f"Hard safety limit {limit_name}={limit_value} exceeded, clamping")
                    cmd.setpoint = min(cmd.setpoint, limit_value)
                elif limit_name.endswith("_min") and cmd.setpoint < limit_value:
                    logger.warning(f"Hard safety limit {limit_name}={limit_value} violated, clamping")
                    cmd.setpoint = max(cmd.setpoint, limit_value)
            
            safe_commands.append(cmd)
        
        return safe_commands
    
    async def trigger_emergency_flush(self, reason: str) -> None:
        """Emergency flush - immediate safety shutdown."""
        logger.critical(f"trigger_emergency_flush called: {reason}")
        self._emergency_active = True
        self._emergency_reason = reason
        await self._client.emergency_stop_all()
    
    def is_emergency_active(self) -> bool:
        """Check if emergency state is active."""
        return self._emergency_active
    
    def get_emergency_reason(self) -> Optional[str]:
        """Get reason for current emergency state."""
        return self._emergency_reason
    
    def clear_emergency(self) -> None:
        """Clear emergency state (after manual verification)."""
        self._emergency_active = False
        self._emergency_reason = None
        logger.info("Emergency state cleared")
    
    def get_current_setpoints(self) -> Dict[str, float]:
        """Get last known setpoints for all actuators."""
        return self._last_setpoints.copy()
    
    def get_actuator_config(self, actuator_id: str) -> Optional[ActuatorConfig]:
        """Get actuator configuration."""
        return self._actuators.get(actuator_id)