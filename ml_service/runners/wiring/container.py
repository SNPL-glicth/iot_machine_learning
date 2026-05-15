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
from iot_machine_learning.infrastructure.config.moe_factory import (
    create_moe_gateway_safe,
)
from iot_machine_learning.infrastructure.ml.moe import MoEGateway

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
        self._moe_gateway: Optional[object] = None  # MoE gateway (lazy, solo si ML_MOE_ENABLED)

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

    def _get_or_create_moe_gateway(self) -> Optional[MoEGateway]:
        """Lazy initialization de MoE gateway (solo si ML_MOE_ENABLED)."""
        if self._moe_gateway is None:
            # Verificar si MoE está habilitado (string "true" o boolean True)
            moe_enabled = getattr(self._flags, 'ML_MOE_ENABLED', False)
            if isinstance(moe_enabled, str):
                moe_enabled = moe_enabled.lower() == 'true'
            
            if moe_enabled:
                # CONTROLLED ACTIVATION: Log whitelist status for safety audit
                whitelist_str = getattr(self._flags, 'ML_BATCH_ENTERPRISE_SENSORS', None)
                mode = "whitelist" if whitelist_str else "global"
                logger.info("MoE feature flag activo", extra={"flag": "ML_MOE_ENABLED", "mode": mode})
                try:
                    self._moe_gateway = create_moe_gateway_safe(sparsity_k=2)
                    if self._moe_gateway:
                        logger.info("MoE gateway activo con k=2")
                    else:
                        logger.warning("MoE gateway falló, usando fallback estándar")
                except Exception as exc:
                    logger.error(f"Error inicializando MoE: {exc}")
                    self._moe_gateway = None
            else:
                logger.debug("MoE deshabilitado")
        return self._moe_gateway

    def _build_prediction_engines(self):
        """Build the canonical list of PredictionEngine instances.

        Returns:
            Tuple[List[PredictionEngine], Optional[MoEGateway]]:
                Engines (PredictionEngine) and optional MoE gateway.
        """
        from iot_machine_learning.infrastructure.ml.engines.core import (
            EngineFactory,
        )

        moe_gateway = self._get_or_create_moe_gateway()
        if moe_gateway:
            logger.info(
                "prediction_adapter_moe_mode",
                extra={"mode": "moe", "sparsity_k": 2},
            )
            return [moe_gateway], moe_gateway

        logger.info(
            "prediction_adapter_standard_mode",
            extra={"mode": "standard"},
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

        return engines, None

    def get_prediction_adapter(self) -> EnterprisePredictionAdapter:
        """Enterprise prediction adapter (singleton por container)."""
        if self._prediction_adapter is None:
            storage = self.get_storage()
            audit = self.get_audit()

            engines, moe_gateway = self._build_prediction_engines()
            # Convert PredictionEngine -> PredictionPort for PredictionDomainService
            ports = [e.as_port() for e in engines]

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
                    "moe_enabled": moe_gateway is not None,
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

            engines, _ = self._build_prediction_engines()

            from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import (
                MetaCognitiveOrchestrator,
            )

            orchestrator = MetaCognitiveOrchestrator(
                engines=engines,
                budget_ms=500.0,
                enable_plasticity=True,
                enable_advanced_plasticity=False,
                enable_iterative=False,
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
