"""DI Container para batch runner enterprise bridge.

Responsabilidades:
- Crear instancias de use cases con dependencias inyectadas
- Lazy initialization (crear solo cuando se usa)
- Lifecycle management (close connections)

Restricción: < 180 líneas.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy.engine import Connection, Engine

from iot_machine_learning.domain.ports.audit_port import AuditPort
from iot_machine_learning.domain.ports.storage_port import StoragePort
from iot_machine_learning.domain.services.prediction_domain_service import (
    PredictionDomainService,
)
from iot_machine_learning.infrastructure.adapters.sqlserver_storage import (
    SqlServerStorageAdapter,
)
from iot_machine_learning.infrastructure.security.audit_logger import (
    FileAuditLogger,
    NullAuditLogger,
)
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags

from ..adapters.enterprise_prediction import EnterprisePredictionAdapter

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

        # Lazy singletons
        self._conn: Optional[Connection] = None
        self._storage: Optional[SqlServerStorageAdapter] = None
        self._audit: Optional[AuditPort] = None
        self._prediction_adapter: Optional[EnterprisePredictionAdapter] = None

    @property
    def flags(self) -> FeatureFlags:
        return self._flags

    def get_connection(self) -> Connection:
        """Conexión SQLAlchemy (singleton por container, auto-reconnect)."""
        if self._conn is None or self._conn.closed:
            self._conn = self._engine.connect()
        return self._conn

    def get_storage(self) -> StoragePort:
        """StoragePort (singleton por container, refreshed on reconnect)."""
        conn = self.get_connection()
        if self._storage is None or self._storage._conn is not conn:
            self._storage = SqlServerStorageAdapter(conn)
        return self._storage

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

    def get_prediction_adapter(self) -> EnterprisePredictionAdapter:
        """Enterprise prediction adapter (singleton por container)."""
        if self._prediction_adapter is None:
            storage = self.get_storage()
            audit = self.get_audit()

            # Build engine list: baseline as fallback
            from iot_machine_learning.infrastructure.ml.engines.baseline_adapter import (
                BaselinePredictionAdapter,
            )

            engines = [BaselinePredictionAdapter(window=60)]

            # Add Taylor if enabled
            if self._flags.ML_USE_TAYLOR_PREDICTOR:
                try:
                    from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
                        EngineFactory,
                    )

                    taylor_engine = EngineFactory.create("taylor")
                    taylor_port = taylor_engine.as_port()
                    engines.insert(0, taylor_port)
                except Exception as exc:
                    logger.warning(
                        "container_taylor_init_failed",
                        extra={"error": str(exc)},
                    )

            domain_service = PredictionDomainService(
                engines=engines,
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
                },
            )

        return self._prediction_adapter

    def close(self) -> None:
        """Cierra recursos abiertos."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._storage = None
        self._prediction_adapter = None
