"""DI Container para batch runner enterprise bridge.

Responsabilidades:
- Crear instancias de use cases con dependencias inyectadas
- Lazy initialization (crear solo cuando se usa)
- Lifecycle management (close connections)

Restricción: < 180 líneas.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from sqlalchemy.engine import Connection, Engine

from iot_machine_learning.domain.ports.audit_port import AuditPort
from iot_machine_learning.domain.ports.storage_port import StoragePort
from iot_machine_learning.domain.services.prediction_domain_service import (
    PredictionDomainService,
)
from iot_machine_learning.infrastructure.persistence.sql.storage import (
    SqlServerStorageAdapter,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_ml_storage import (
    ZeninMLStorageAdapter,
)
from iot_machine_learning.infrastructure.persistence.sql.dual_write_storage import (
    DualWriteStorageAdapter,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_ml_only_storage import (
    ZeninMLOnlyStorageAdapter,
)
from iot_machine_learning.infrastructure.security.audit_logger import (
    FileAuditLogger,
    NullAuditLogger,
)
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags

from ..adapters.enterprise_prediction import EnterprisePredictionAdapter
from ..adapters.orchestrator_prediction import OrchestratorPredictionAdapter

logger = logging.getLogger(__name__)


class BatchEnterpriseContainer:
    """DI container para componentes enterprise del batch runner.

    Patrón: Lazy initialization — crea singletons al primer acceso.
    Cada instancia está ligada a un ``engine`` de SQLAlchemy.

    Attributes:
        _engine: SQLAlchemy engine para crear conexiones.
        _flags: Feature flags globales.
        _audit_log_path: Ruta al archivo de audit log.
    """

    def __init__(
        self,
        engine: Engine,
        flags: FeatureFlags,
        audit_log_path: str = "./logs/batch_audit.log",
    ) -> None:
        self._engine = engine
        self._flags = flags
        self._audit_log_path = audit_log_path

        # Thread-local storage — one connection + storage per thread
        self._thread_local = threading.local()

        # Lazy singletons (thread-safe container-level)
        self._audit: Optional[AuditPort] = None
        self._prediction_adapter: Optional[EnterprisePredictionAdapter] = None
        self._cognitive_adapter: Optional[OrchestratorPredictionAdapter] = None
        self._dynamic_tuner: Optional[object] = None

    @property
    def flags(self) -> FeatureFlags:
        return self._flags

    def get_connection(self) -> Connection:
        """Conexión SQLAlchemy (singleton por container, auto-reconnect)."""
        if not hasattr(self._thread_local, "connection") or \
           self._thread_local.connection is None or \
           self._thread_local.connection.closed:
            self._thread_local.connection = self._engine.connect()
        return self._thread_local.connection

    def get_storage(self) -> StoragePort:
        """StoragePort (singleton por container, refreshed on reconnect).
        
        Uses ZeninMLOnlyStorageAdapter when enterprise mode is enabled:
        - Reads from legacy dbo (SqlServerStorageAdapter)
        - Writes ONLY to zenin_ml.predictions (ZeninMLStorageAdapter)
        
        Falls back to legacy SqlServerStorageAdapter (dbo.predictions only)
        when ML_BATCH_USE_ENTERPRISE is False.
        """
        conn = self.get_connection()
        if not hasattr(self._thread_local, "storage") or \
           self._thread_local.storage is None or \
           self._thread_local.storage._conn is not conn:
            if self._flags.ML_BATCH_USE_ENTERPRISE:
                self._thread_local.storage = ZeninMLOnlyStorageAdapter(conn)
                logger.info("[BATCH_CONTAINER] Using ZeninMLOnlyStorageAdapter (reads legacy, writes ONLY to zenin_ml)")
            else:
                self._thread_local.storage = SqlServerStorageAdapter(conn)
                logger.info("[BATCH_CONTAINER] Using SqlServerStorageAdapter (legacy dbo.predictions only)")
        return self._thread_local.storage

    def get_audit(self) -> AuditPort:
        """AuditPort (singleton por container)."""
        if self._audit is None:
            if self._flags.ML_ENABLE_AUDIT_LOGGING:
                self._audit = FileAuditLogger(
                    log_file=Path(self._audit_log_path),
                )
            else:
                self._audit = NullAuditLogger()
        return self._audit

    def _build_prediction_engines(self):
        """Build the canonical list of PredictionEngine instances.

        Returns:
            List[PredictionEngine]: Canonical engine list.
        """
        from iot_machine_learning.infrastructure.ml.engines.core import (
            EngineFactory,
        )

        engines = []

        # Baseline as fallback
        baseline_engine = EngineFactory.create("baseline_moving_average")
        engines.append(baseline_engine)

        # Kalman
        try:
            kalman_engine = EngineFactory.create("kalman")
            engines.insert(0, kalman_engine)
        except Exception as exc:
            logger.warning(
                "container_kalman_init_failed",
                extra={"error": str(exc)},
            )

        # Taylor if enabled
        if self._flags.ML_USE_TAYLOR_PREDICTOR:
            try:
                taylor_engine = EngineFactory.create("taylor")
                engines.insert(0, taylor_engine)
            except Exception as exc:
                logger.warning(
                    "container_taylor_init_failed",
                    extra={"error": str(exc)},
                )

        # MoE as engine within pipeline (no reemplaza, agrega)
        moe_as_engine = getattr(self._flags, 'ML_MOE_AS_ENGINE', False)
        if isinstance(moe_as_engine, str):
            moe_as_engine = moe_as_engine.lower() == 'true'
        if moe_as_engine:
            try:
                moe_engine = self._create_moe_prediction_engine(baseline_engine)
                if moe_engine:
                    engines.insert(0, moe_engine)
                    logger.info(
                        "moe_as_engine_enabled",
                        extra={"engine": "moe_engine", "fallback": "baseline"},
                    )
            except Exception as exc:
                logger.warning(
                    "moe_as_engine_init_failed",
                    extra={"error": str(exc)},
                )

        return engines

    def _create_moe_prediction_engine(self, fallback_engine):
        """Crea MoEPredictionEngine con expertos registrados."""
        from iot_machine_learning.infrastructure.config.moe_factory import create_moe_engine
        return create_moe_engine(
            fallback_engine=fallback_engine.as_port(),
            sparsity_k=2,
            enable_shadow_gating=True,
            ab_cell="B",
        )

    def get_prediction_adapter(self) -> EnterprisePredictionAdapter:
        """Enterprise prediction adapter (singleton por container)."""
        if self._prediction_adapter is None:
            storage = self.get_storage()
            audit = self.get_audit()

            engines = self._build_prediction_engines()
            # Convert PredictionEngine -> PredictionPort for PredictionDomainService
            ports = [e.as_port() for e in engines]

            # Rollout gradual: envolver port MoE con RolloutPredictionPortBridge
            moe_rollout = getattr(self._flags, 'ML_MOE_AS_ENGINE', False)
            if isinstance(moe_rollout, str):
                moe_rollout = moe_rollout.lower() == 'true'
            if moe_rollout:
                try:
                    from iot_machine_learning.infrastructure.ml.moe.rollout.rollout_bridge import (
                        RolloutPredictionPortBridge,
                    )
                    from iot_machine_learning.infrastructure.ml.moe.rollout.rollout_decider import (
                        RolloutDecider,
                    )
                    # Encontrar port MoE y fallback (baseline)
                    moe_port = None
                    fallback_port = None
                    for p in ports:
                        if p.name == "moe_engine":
                            moe_port = p
                        if "baseline" in p.name.lower():
                            fallback_port = p
                    if moe_port and fallback_port:
                        decider = RolloutDecider()
                        bridge = RolloutPredictionPortBridge(
                            engine=moe_port,
                            fallback=fallback_port,
                            decider=decider,
                        )
                        # Reemplazar port MoE con bridge
                        ports = [bridge if p is moe_port else p for p in ports]
                        logger.info(
                            "moe_rollout_applied",
                            extra={"percent": decider.percent},
                        )
                except Exception as exc:
                    logger.warning("moe_rollout_failed", extra={"error": str(exc)})

            domain_service = PredictionDomainService(
                engines=ports,
                audit=audit,
            )

            from iot_machine_learning.application.use_cases.predict_sensor_value import (
                PredictSensorValueUseCase,
            )

            use_case = PredictSensorValueUseCase(
                prediction_service=domain_service,
                storage=storage,
                audit=audit,
                window_size=500,
                flags=self._flags,
            )

            self._prediction_adapter = EnterprisePredictionAdapter(
                storage=storage,
                use_case=use_case,
                audit=audit,
            )

            logger.info(
                "container_enterprise_adapter_created",
                extra={
                    "engines": [e.name for e in engines],
                    "audit_enabled": self._flags.ML_ENABLE_AUDIT_LOGGING,
                    "moe_as_engine": getattr(self._flags, 'ML_MOE_AS_ENGINE', False),
                },
            )
            # OBSERVABILITY: Track engine usage distribution
            from ...metrics.observability import get_observability
            obs = get_observability()
            for e in engines:
                obs.engine_usage.record(e.name, -1)

        return self._prediction_adapter

    def get_cognitive_adapter(self) -> OrchestratorPredictionAdapter:
        """Cognitive orchestrator adapter (singleton por container)."""
        if self._cognitive_adapter is None:
            storage = self.get_storage()
            audit = self.get_audit()

            engines = self._build_prediction_engines()

            from iot_machine_learning.domain.repositories.sensor_profile_repository import (
                SensorProfileRepository,
            )
            from iot_machine_learning.infrastructure.repositories.sql_sensor_profile_repository import (
                SqlSensorProfileRepository,
            )
            from iot_machine_learning.infrastructure.ml.moe.engine_weight_initializer import (
                EquipmentTypeWeightInitializer,
                StructuralEngineFilter,
            )
            from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import (
                MetaCognitiveOrchestrator,
            )

            # Fase 2: inyectar SensorProfileRepository al orchestrator
            profile_repo = SqlSensorProfileRepository(self.get_connection())
            weight_initializer = EquipmentTypeWeightInitializer()
            engine_filter = StructuralEngineFilter()

            orchestrator = MetaCognitiveOrchestrator(
                engines=engines,
                budget_ms=500.0,
                enable_plasticity=True,
                enable_advanced_plasticity=False,
                enable_iterative=False,
                sensor_profile_repository=profile_repo,
                weight_initializer=weight_initializer,
                engine_filter=engine_filter,
            )

            self._cognitive_adapter = OrchestratorPredictionAdapter(
                orchestrator=orchestrator,
                storage=storage,
                audit=audit,
                flags=self._flags,
            )

            logger.info(
                "container_cognitive_adapter_created",
                extra={
                    "engines": [e.name for e in engines],
                    "plasticity": True,
                },
            )

        return self._cognitive_adapter

    def get_dynamic_tuner(self) -> Optional[object]:
        """DynamicTuner (singleton por container)."""
        if self._dynamic_tuner is None:
            from core.tuning.dynamic_tuning import DynamicTuner
            from core.parameters.parameter_bounds import ParameterBoundsEnforcer
            self._dynamic_tuner = DynamicTuner(
                bounds_enforcer=ParameterBoundsEnforcer(),
                convergence_window=20,
            )
            logger.info("container_dynamic_tuner_created")
        return self._dynamic_tuner

    def close(self) -> None:
        """Cierra recursos abiertos."""
        conn = getattr(self._thread_local, "connection", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._thread_local.connection = None
        self._thread_local.storage = None
        self._prediction_adapter = None
        self._cognitive_adapter = None
        self._dynamic_tuner = None
