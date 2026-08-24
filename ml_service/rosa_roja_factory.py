"""Rosa Roja Engine Factory for IoT ML Service.

Creates and configures the complete Rosa Roja engine with IoT-specific
adapters for experts, drift sensors, and actuators.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from core.orchestration.rosa_roja.engine import RosaRojaEngine
from core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
from core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator
from core.orchestration.rosa_roja.modules.module3_moe_gating import MultiplicativeMoEGating
from infrastructure.ml.adapters import (
    TaylorExpertAdapter,
    KalmanExpertAdapter,
    StatisticalExpertAdapter,
    IoTDriftSensorAdapter,
    IoTActuatorHandler,
    ActuatorConfig,
    ActuatorType,
    MockActuatorClient,
)

logger = logging.getLogger(__name__)


class RosaRojaEngineFactory:
    """Factory for creating and configuring Rosa Roja Engine instances."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._engine: Optional[RosaRojaEngine] = None
        self._actuator_handler: Optional[IoTActuatorHandler] = None
        self._experts = []
        self._drift_sensors = []
    
    def create_engine(self) -> RosaRojaEngine:
        """Create the complete Rosa Roja engine with all components."""
        
        # 1. Create ML expert adapters
        experts = self._create_experts()
        
        # 2. Create IoT drift sensors
        drift_sensors = self._create_drift_sensors()
        
        # 3. Create core modules
        ingestion = self._create_ingestion_filter()
        rhythm = self._create_rhythm_generator()
        gating = self._create_moe_gating()
        
        # 4. Create actuator handler
        actuator_handler = self._create_actuator_handler()

        # 5. Optional ML state store for warm restarts (None = cold-start only)
        state_store = None
        store_config = self.config.get("state_store")
        if store_config:
            try:
                from infrastructure.ml.adapters.ml_state_store import create_state_store
                state_store = create_state_store(store_config)
            except Exception as e:
                logger.warning(f"Failed to create ML state store, persistence disabled: {e}")

        # 6. Assemble engine
        engine = RosaRojaEngine(
            ingestion_filter=ingestion,
            rhythm_generator=rhythm,
            moe_gating=gating,
            expert_jury=experts,
            drift_sensors=drift_sensors,
            outlier_reset_threshold=self.config.get("outlier_reset_threshold", 3),
            exploration_boost_events=self.config.get("exploration_boost_events", 5),
            state_store=state_store,
            engine_id=self.config.get("engine_id", "default"),
            checkpoint_interval=self.config.get("checkpoint_interval", 100),
        )

        # Warm start: restore prior learning state when available.
        if state_store is not None and self.config.get("restore_on_create", True):
            restored = engine.restore()
            logger.info("ML state restore attempted: restored=%s", restored)
        
        self._engine = engine
        self._actuator_handler = actuator_handler
        self._experts = experts
        self._drift_sensors = drift_sensors
        
        logger.info(
            "Rosa Roja Engine created",
            extra={
                "n_experts": len(experts),
                "n_drift_sensors": len(drift_sensors),
                "n_actuators": len(actuator_handler._actuators) if actuator_handler else 0,
            }
        )
        
        return engine
    
    def _create_experts(self):
        """Create expert adapters wrapping ML prediction engines."""
        experts = []
        
        # Taylor Expert (critical, trend detection)
        try:
            from iot_machine_learning.infrastructure.ml.engines.taylor.engine import TaylorPredictionEngine
            taylor_engine = TaylorPredictionEngine()
            taylor_expert = TaylorExpertAdapter(
                engine=taylor_engine,
                name="taylor",
                is_critical=self.config.get("taylor_critical", True),
                threshold=self.config.get("taylor_threshold", 0.65),
                weight=self.config.get("taylor_weight", 1.2),
            )
            experts.append(taylor_expert)
            logger.info("Taylor expert adapter created")
        except Exception as e:
            logger.warning(f"Failed to create Taylor expert: {e}")
        
        # Kalman Expert (critical, state estimation)
        try:
            from iot_machine_learning.infrastructure.ml.engines.kalman.engine import KalmanPredictionEngine
            kalman_engine = KalmanPredictionEngine()
            kalman_expert = KalmanExpertAdapter(
                engine=kalman_engine,
                name="kalman",
                is_critical=self.config.get("kalman_critical", True),
                threshold=self.config.get("kalman_threshold", 0.60),
                weight=self.config.get("kalman_weight", 1.0),
            )
            experts.append(kalman_expert)
            logger.info("Kalman expert adapter created")
        except Exception as e:
            logger.warning(f"Failed to create Kalman expert: {e}")
        
        # Statistical Expert (non-critical, baseline)
        try:
            from iot_machine_learning.infrastructure.ml.engines.statistical.engine import StatisticalPredictionEngine
            stat_engine = StatisticalPredictionEngine()
            stat_expert = StatisticalExpertAdapter(
                engine=stat_engine,
                name="statistical",
                is_critical=self.config.get("statistical_critical", False),
                threshold=self.config.get("statistical_threshold", 0.55),
                weight=self.config.get("statistical_weight", 0.8),
            )
            experts.append(stat_expert)
            logger.info("Statistical expert adapter created")
        except Exception as e:
            logger.warning(f"Failed to create Statistical expert: {e}")
        
        # Cognitive Expert (optional, if available)
        try:
            from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import MetaCognitiveOrchestrator
            from iot_machine_learning.infrastructure.ml.engines.taylor.engine import TaylorPredictionEngine
            from iot_machine_learning.infrastructure.ml.engines.statistical.engine import StatisticalPredictionEngine
            from iot_machine_learning.ml_service.config.feature_flags import get_feature_flags
            
            cognitive_orch = MetaCognitiveOrchestrator(
                engines=[TaylorPredictionEngine(), StatisticalPredictionEngine()]
            )
            cognitive_expert = TaylorExpertAdapter(
                engine=cognitive_orch,
                name="cognitive",
                is_critical=self.config.get("cognitive_critical", False),
                threshold=self.config.get("cognitive_threshold", 0.70),
                weight=self.config.get("cognitive_weight", 1.5),
            )
            experts.append(cognitive_expert)
            logger.info("Cognitive expert adapter created")
        except Exception as e:
            logger.debug(f"Cognitive expert not available: {e}")
        
        if not experts:
            logger.warning("No experts created - Rosa Roja will not function properly")
        
        return experts
    
    def _create_drift_sensors(self):
        """Create drift sensors for telemetry channels."""
        drift_sensors = []
        
        # Channel groups - should be provided via config
        chiller_channels = self.config.get("chiller_channels", [])
        ca_channels = self.config.get("ca_channels", [])
        
        # Chiller drift sensor (only if channels configured)
        if chiller_channels:
            chiller_drift = IoTDriftSensorAdapter(
                name="chiller_drift",
                channels=chiller_channels,
                detector_type=self.config.get("drift_detector_type", "page_hinkley"),
                aggregation=self.config.get("drift_aggregation", "max"),
                channel_weights=self.config.get("chiller_channel_weights"),
                ph_delta=self.config.get("ph_delta", 0.005),
                ph_lambda=self.config.get("ph_lambda", 50.0),
                ph_alpha=self.config.get("ph_alpha", 0.9999),
            )
            drift_sensors.append(chiller_drift)
            logger.info(f"Chiller drift sensor created with {len(chiller_channels)} channels")
        
        # CA drift sensor (only if channels configured)
        if ca_channels:
            ca_drift = IoTDriftSensorAdapter(
                name="ca_drift",
                channels=ca_channels,
                detector_type=self.config.get("drift_detector_type", "page_hinkley"),
                aggregation=self.config.get("drift_aggregation", "max"),
                channel_weights=self.config.get("ca_channel_weights"),
                ph_delta=self.config.get("ph_delta", 0.005),
                ph_lambda=self.config.get("ph_lambda", 50.0),
                ph_alpha=self.config.get("ph_alpha", 0.9999),
            )
            drift_sensors.append(ca_drift)
            logger.info(f"CA drift sensor created with {len(ca_channels)} channels")
        
        # Generic error drift detector (uses prediction errors)
        error_drift = IoTDriftSensorAdapter(
            name="error_drift",
            channels=["prediction_error"],
            detector_type="error_drift",
            aggregation="max",
            window_size=self.config.get("error_drift_window", 100),
            error_detector_type=self.config.get("error_drift_detector_type", "page_hinkley"),
            ph_delta=self.config.get("ph_delta", 0.005),
            ph_lambda=self.config.get("ph_lambda", 50.0),
            ph_alpha=self.config.get("ph_alpha", 0.9999),
        )
        drift_sensors.append(error_drift)
        logger.info("Error drift sensor created")
        
        return drift_sensors
    
    def _create_ingestion_filter(self) -> MahalanobisFilter:
        """Create Module 1: Mahalanobis Anti-Contamination Filter."""
        return MahalanobisFilter(
            noise_threshold=self.config.get("mahalanobis_threshold", 3.0),
            history_window=self.config.get("mahalanobis_history_window", 100),
            min_samples_for_cov=self.config.get("mahalanobis_min_samples", 20),
        )
    
    def _create_rhythm_generator(self) -> RhythmTrajectoryGenerator:
        """Create Module 2: Rhythm Trajectory Generator."""
        return RhythmTrajectoryGenerator(
            min_trajectory_len=self.config.get("min_trajectory_len", 11),
            max_trajectory_len=self.config.get("max_trajectory_len", 15),
            top_k=self.config.get("top_k", 3),
            rhythm_weight=self.config.get("rhythm_weight", 0.5),
            max_entropy=self.config.get("max_entropy", 1.0),
            oversample_factor=self.config.get("oversample_factor", 2),
            max_random_walk_steps=self.config.get("max_random_walk_steps", 50),  # Reduced for latency
            invalidation_threshold=self.config.get("invalidation_threshold", 0.5),
            theta_alpha=self.config.get("theta_alpha", 0.95),
            quantization_decimals=self.config.get("quantization_decimals", 3),
        )
    
    def _create_moe_gating(self) -> MultiplicativeMoEGating:
        """Create Module 3: MoE Gating with Hard Veto."""
        return MultiplicativeMoEGating(
            variance_penalty=self.config.get("variance_penalty", 0.5),
        )
    
    def _create_actuator_handler(self) -> IoTActuatorHandler:
        """Create IoT Actuator Handler (ExecutionPort)."""
        
        # Actuator configurations from config or defaults
        actuator_configs = self.config.get("actuators", {})
        
        if not actuator_configs:
            # Default generic actuators (should be overridden by config in production)
            device_id = self.config.get("device_id", "device_01")
            actuator_configs = {
                "actuator_1": ActuatorConfig(
                    actuator_id="actuator_1",
                    actuator_type=ActuatorType.GENERIC,
                    device_id=device_id,
                    min_setpoint=0.0,
                    max_setpoint=100.0,
                    unit="%",
                    safety_min=0.0,
                    safety_max=100.0,
                    rate_limit=5.0,
                    mqtt_topic=f"{device_id}/actuator_1/set",
                ),
                "actuator_2": ActuatorConfig(
                    actuator_id="actuator_2",
                    actuator_type=ActuatorType.GENERIC,
                    device_id=device_id,
                    min_setpoint=0.0,
                    max_setpoint=100.0,
                    unit="%",
                    safety_min=0.0,
                    safety_max=100.0,
                    rate_limit=10.0,
                    mqtt_topic=f"{device_id}/actuator_2/set",
                ),
            }
        
        # Use mock client if no real client provided
        actuator_client = self.config.get("actuator_client")
        if actuator_client is None:
            actuator_client = MockActuatorClient()
            logger.warning("Using MockActuatorClient - replace with real MQTT/OPC-UA client in production")
        
        handler = IoTActuatorHandler(
            actuator_client=actuator_client,
            actuators=actuator_configs,
            safety_limits=self.config.get("safety_limits", {}),
            default_rate_limit=self.config.get("default_rate_limit", 10.0),
            device_id=self.config.get("device_id", "device_01"),
        )
        
        return handler
    
    def get_engine(self) -> Optional[RosaRojaEngine]:
        """Get the created engine instance."""
        return self._engine
    
    def get_actuator_handler(self) -> Optional[IoTActuatorHandler]:
        """Get the actuator handler instance."""
        return self._actuator_handler
    
    def get_experts(self) -> list:
        """Get expert adapters."""
        return self._experts
    
    def get_drift_sensors(self) -> list:
        """Get drift sensors."""
        return self._drift_sensors


def create_rosa_roja_engine(config: Optional[Dict[str, Any]] = None) -> RosaRojaEngine:
    """Convenience function to create a Rosa Roja engine."""
    factory = RosaRojaEngineFactory(config)
    return factory.create_engine()


def init_rosa_roja_in_app(app, config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize Rosa Roja engine and attach to FastAPI app state.
    
    Called during application lifespan startup.
    
    Args:
        app: FastAPI application instance
        config: Optional configuration dictionary
    """
    factory = RosaRojaEngineFactory(config)
    engine = factory.create_engine()
    actuator_handler = factory.get_actuator_handler()
    
    app.state.rosa_roja = engine
    app.state.actuator = actuator_handler
    app.state.rosa_roja_factory = factory
    
    logger.info("Rosa Roja engine initialized and attached to app.state")


def get_rosa_roja_engine(app) -> Optional[RosaRojaEngine]:
    """Get Rosa Roja engine from app state."""
    return getattr(app.state, "rosa_roja", None)


def get_actuator_handler(app) -> Optional[IoTActuatorHandler]:
    """Get actuator handler from app state."""
    return getattr(app.state, "actuator", None)