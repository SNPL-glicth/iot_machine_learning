"""IoT ML Service - FastAPI Application.

REFACTORIZADO 2026-01-29:
- Modularizado en api/services/, api/routes.py, api/schemas.py
- Integración con Redis Streams broker
- Logs estructurados
- Métricas de performance

Este archivo ahora es solo el punto de entrada de la aplicación.
La lógica de negocio está en api/services/.
"""

from __future__ import annotations

# FIX: Add project root to PYTHONPATH for proper module resolution
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# FIX: Load env vars BEFORE any other imports that might use them
from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

_zenin_poller = None      # Keep reference for stats/health
_poller_thread = None     # Keep reference for healthcheck


def _validate_prediction_paths() -> None:
    """INF-5: Guard against dual prediction paths (stream + batch runner).

    Stream predictions and batch runner are MUTUALLY EXCLUSIVE.
    Enabling both guarantees duplicate rows in dbo.predictions and
    dbo.ml_events. This guard fails fast at startup.
    """
    from .config.loader import get_feature_flags

    flags = get_feature_flags()
    if flags.ML_STREAM_PREDICTIONS_ENABLED:
        batch_active = (
            os.environ.get("ZENIN_QUEUE_POLLER_ENABLED", "false").lower() == "true"
            or bool(os.environ.get("ML_BATCH_ENTERPRISE_SENSORS"))
        )
        if batch_active:
            logger.error(
                json.dumps({
                    "event": "startup_config_conflict",
                    "component": "prediction_path_guard",
                    "error": "ML_STREAM_PREDICTIONS_ENABLED y batch runner activos simultáneamente — duplicación de predicciones garantizada",
                    "action": "Deshabilitar uno de los dos paths antes de iniciar"
                })
            )
            raise RuntimeError(
                "INF-5: ML_STREAM_PREDICTIONS_ENABLED=true es incompatible "
                "con batch runner activo. Ver logs para detalle."
            )


def validate_compliance_path(export_path: str, base_dir: str) -> Path:
    """Validate compliance export path against directory traversal.

    Args:
        export_path: requested sink path (may be relative).
        base_dir: allowed base directory.

    Returns:
        Resolved Path if valid.

    Raises:
        ValueError: if resolved path escapes base_dir.
    """
    allowed = Path(base_dir).resolve()
    resolved = Path(export_path).resolve()
    if not str(resolved).startswith(str(allowed)):
        raise ValueError(
            f"compliance_path_traversal: {resolved} not under {allowed}"
        )
    return resolved


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    global _zenin_poller

    # Startup
    logger.info("[ML-SERVICE] Starting up...")

    # IMP-5: Resolve compliance exporter once at startup (PIPE-2 fix).
    # Path traversal validation: sink must live under ALLOWED_BASE_DIR.
    _compliance_export_path = os.environ.get("ML_COMPLIANCE_EXPORT_PATH")
    if _compliance_export_path:
        try:
            _resolved_sink = validate_compliance_path(
                _compliance_export_path,
                os.environ.get("ML_COMPLIANCE_BASE_DIR", "/var/lib/zenin/compliance"),
            )
            from iot_machine_learning.infrastructure.ml.cognitive.compliance.compliance_exporter import (
                ComplianceExporter,
                load_hmac_key_from_env,
            )
            app.state.compliance_exporter = ComplianceExporter(
                sink_path=_resolved_sink,
                hmac_key=load_hmac_key_from_env(),
            )
            logger.info(
                "compliance_exporter_ready",
                extra={"sink": str(_resolved_sink)},
            )
        except ValueError as exc:
            logger.error(
                "compliance_path_traversal_blocked",
                extra={"error": str(exc)},
            )
            app.state.compliance_exporter = None
        except Exception as exc:
            logger.error(
                "compliance_exporter_init_failed",
                extra={"error": str(exc)},
            )
            app.state.compliance_exporter = None
    else:
        app.state.compliance_exporter = None

    # FIX: Reset circuit breakers to ensure fresh connection state
    try:
        from iot_machine_learning.infrastructure.persistence.redis import reset_all_circuits
        reset_all_circuits()
        logger.info("[ML-SERVICE] Circuit breakers reset")
    except Exception as e:
        logger.warning(
            "[ML-SERVICE] Circuit breakers reset failed — continuing without resilience cache",
            extra={"degraded_feature": "circuit_breaker", "error": str(e)},
        )
    
    # Initialize broker (CRITICAL — app cannot function without messaging)
    from .broker import get_broker, get_broker_health
    broker = get_broker()
    health = get_broker_health()
    logger.info(
        "[ML-SERVICE] Broker initialized: type=%s connected=%s",
        health.get("type", "unknown"),
        health.get("connected", False),
    )

    # INF-5: Validate mutually exclusive prediction paths
    _validate_prediction_paths()

    # Initialize Zenin Queue Poller if enabled (OPTIONAL background worker)
    _poller_flag = os.environ.get("ZENIN_QUEUE_POLLER_ENABLED", "false")
    logger.info("[ML-SERVICE] ZENIN_QUEUE_POLLER_ENABLED=%s (from env)", _poller_flag)
    if _poller_flag.lower() == "true":
        try:
            from .workers.zenin_queue_poller import ZeninQueuePoller
            _zenin_poller = ZeninQueuePoller()
            _poller_thread = threading.Thread(
                target=_zenin_poller.start,
                name="zenin-queue-poller",
                daemon=True,
            )
            _poller_thread.start()
            logger.info("[ML-SERVICE] Zenin Queue Poller started (daemon thread)")
        except Exception as e:
            logger.warning(
                "[ML-SERVICE] Zenin Queue Poller failed to start — continuing without background polling",
                extra={"degraded_feature": "zenin_queue_poller", "error": str(e)},
            )
    else:
        logger.info("[ML-SERVICE] Zenin Queue Poller disabled (ZENIN_QUEUE_POLLER_ENABLED != true)")

    app.state.zenin_poller = _zenin_poller
    app.state.zenin_poller_thread = _poller_thread

    yield
    
    # Shutdown
    logger.info("[ML-SERVICE] Shutting down...")
    try:
        from .broker import reset_broker
        reset_broker()
    except Exception:
        pass
    try:
        from ..infrastructure.persistence.sql.zenin_db_connection import ZeninDbConnection
        ZeninDbConnection.dispose()
    except Exception:
        pass


# API-4: Disable /docs and /redoc in production
is_production = os.getenv("ML_ENV", "production") == "production"

app = FastAPI(
    title="IoT ML Service",
    version="0.2.0",
    description="Machine Learning service for IoT sensor predictions",
    lifespan=lifespan,
    docs_url=None if is_production else "/docs",
    redoc_url=None if is_production else "/redoc",
)

# CORS configuration - load from FeatureFlags
from .config.loader import get_feature_flags
_flags = get_feature_flags()
ALLOWED_ORIGINS = os.getenv(
    "ML_CORS_ORIGINS",
    _flags.ML_CORS_ORIGINS
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Import and include router after app creation to avoid circular imports
from .api.routes import router
from .api.routes_cognitive import router as cognitive_router

app.include_router(router)
app.include_router(cognitive_router)

# Health endpoint at root level for backwards compatibility
@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "iot-ml-service", "version": "0.2.0", "status": "ok"}
