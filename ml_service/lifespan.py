"""Application lifespan manager — extracted from main.py for ≤180 lines."""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from iot_machine_learning.infrastructure.security.secret_redactor import SecretRedactor

logger = logging.getLogger(__name__)

_zenin_poller = None
_governance_initializer = None
_broker = None

def _validate_prediction_paths() -> None:
    """INF-5: Guard against dual prediction paths (stream + batch runner)."""
    from .config.loader import get_feature_flags

    flags = get_feature_flags()
    if flags.ML_STREAM_PREDICTIONS_ENABLED:
        batch_active = (
            os.environ.get("ZENIN_QUEUE_POLLER_ENABLED", "false").lower() == "true"
            or bool(os.environ.get("ML_BATCH_ENTERPRISE_SENSORS"))
        )
        if batch_active:
            logger.error(json.dumps({
                "event": "startup_config_conflict",
                "component": "prediction_path_guard",
                "error": "ML_STREAM_PREDICTIONS_ENABLED y batch runner activos simultáneamente",
            }))
            raise RuntimeError(
                "INF-5: ML_STREAM_PREDICTIONS_ENABLED=true es incompatible "
                "con batch runner activo."
            )

def validate_compliance_path(export_path: str, base_dir: str) -> Path:
    """Validate compliance export path against directory traversal."""
    allowed = Path(base_dir).resolve()
    resolved = Path(export_path).resolve()
    if not str(resolved).startswith(str(allowed)):
        raise ValueError(f"compliance_path_traversal: {resolved} not under {allowed}")
    return resolved


def _init_compliance(app: FastAPI) -> None:
    """Initialize compliance exporter if configured."""
    _compliance_export_path = os.environ.get("ML_COMPLIANCE_EXPORT_PATH")
    logger.info("startup_config_loaded", extra=SecretRedactor.redact({
        "ML_COMPLIANCE_EXPORT_PATH": _compliance_export_path,
        "ML_COMPLIANCE_BASE_DIR": os.environ.get("ML_COMPLIANCE_BASE_DIR", "/var/lib/zenin/compliance"),
        "ZENIN_QUEUE_POLLER_ENABLED": os.environ.get("ZENIN_QUEUE_POLLER_ENABLED", "false"),
        "ML_ENV": os.environ.get("ML_ENV", "production"),
    }))
    if _compliance_export_path:
        try:
            _resolved_sink = validate_compliance_path(
                _compliance_export_path,
                os.environ.get("ML_COMPLIANCE_BASE_DIR", "/var/lib/zenin/compliance"),
            )
            from iot_machine_learning.infrastructure.ml.cognitive.compliance.compliance_exporter import (
                ComplianceExporter, load_hmac_key_from_env,
            )
            app.state.compliance_exporter = ComplianceExporter(
                sink_path=_resolved_sink, hmac_key=load_hmac_key_from_env(),
            )
            logger.info("compliance_exporter_ready", extra={"sink": str(_resolved_sink)})
        except ValueError as exc:
            logger.error("compliance_path_traversal_blocked", extra={"error": str(exc)})
            app.state.compliance_exporter = None
        except Exception as exc:
            logger.error("compliance_exporter_init_failed", extra={"error": str(exc)})
            app.state.compliance_exporter = None
    else:
        app.state.compliance_exporter = None


def _init_poller() -> object | None:
    """Start Zenin Queue Poller if enabled."""
    _poller_flag = os.environ.get("ZENIN_QUEUE_POLLER_ENABLED", "false")
    logger.info("[ML-SERVICE] ZENIN_QUEUE_POLLER_ENABLED=%s", _poller_flag)
    if _poller_flag.lower() == "true":
        try:
            from .workers.zenin_queue_poller import ZeninQueuePoller
            poller = ZeninQueuePoller()
            poller.start()
            logger.info("[ML-SERVICE] Zenin Queue Poller started (watchdog)")
            return poller
        except Exception as e:
            logger.warning("[ML-SERVICE] Zenin Queue Poller failed to start",
                           extra={"error": str(e)})
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _zenin_poller, _governance_initializer, _broker

    _api_workers = os.environ.get("ML_API_WORKERS", "4")
    logger.info("ml_service_startup", extra={"ml_api_workers": _api_workers})

    # FASE-9: Governance
    try:
        from .governance_initializer import GovernanceInitializer
        _governance_initializer = GovernanceInitializer(logger=logger)
        app.state.governance = _governance_initializer.initialize()
        logger.info("[ML-SERVICE] Governance system initialized")
    except Exception as e:
        logger.error("[ML-SERVICE] Governance initialization failed", extra={"error": str(e)})
        app.state.governance = None

    _init_compliance(app)

    # Circuit breakers
    try:
        from iot_machine_learning.infrastructure.persistence.redis import reset_all_circuits
        reset_all_circuits()
    except Exception as e:
        logger.warning("[ML-SERVICE] Circuit breakers reset failed", extra={"error": str(e)})

    # Broker
    from .broker import get_broker, get_broker_health
    _broker = get_broker()
    health = get_broker_health()
    logger.info("[ML-SERVICE] Broker: type=%s connected=%s",
                health.get("type", "unknown"), health.get("connected", False))

    _validate_prediction_paths()

    _zenin_poller = _init_poller()
    app.state.zenin_poller = _zenin_poller
    app.state.broker = _broker

    # FIX P3: PredictionWorker + ResultStore + StreamConsumer
    try:
        from .consumers.prediction_lifecycle import init_prediction_worker
        init_prediction_worker(app)
    except Exception as e:
        logger.warning("[ML-SERVICE] P3 init failed: %s", e)
        app.state.prediction_worker = None
        app.state.result_store = None

    yield

    # Shutdown
    logger.info("[ML-SERVICE] Shutting down...")
    if _governance_initializer:
        try:
            _governance_initializer.shutdown()
        except Exception as e:
            logger.warning("[ML-SERVICE] Governance shutdown failed", extra={"error": str(e)})

    if _zenin_poller is not None:
        try:
            _zenin_poller.stop()
        except Exception:
            pass

    # FIX P3: stop PredictionWorker
    try:
        from .consumers.prediction_lifecycle import stop_prediction_worker
        stop_prediction_worker(app)
    except Exception:
        pass

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
