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

import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Load .env from the iot_machine_learning root directory
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=False)

logger = logging.getLogger(__name__)

_zenin_poller = None  # Keep reference for stats/health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    global _zenin_poller

    # Startup
    logger.info("[ML-SERVICE] Starting up...")
    
    # Initialize broker if Redis is enabled
    try:
        from .broker import get_broker, get_broker_health
        broker = get_broker()
        health = get_broker_health()
        logger.info(
            "[ML-SERVICE] Broker initialized: type=%s connected=%s",
            health.get("type", "unknown"),
            health.get("connected", False),
        )
    except Exception as e:
        logger.warning("[ML-SERVICE] Broker initialization failed: %s", str(e))
    
    # Initialize Zenin Queue Poller if enabled
    _poller_flag = os.environ.get("ZENIN_QUEUE_POLLER_ENABLED", "false")
    logger.info("[ML-SERVICE] ZENIN_QUEUE_POLLER_ENABLED=%s (from env)", _poller_flag)
    if _poller_flag.lower() == "true":
        try:
            from .workers.zenin_queue_poller import ZeninQueuePoller
            _zenin_poller = ZeninQueuePoller()
            poller_thread = threading.Thread(
                target=_zenin_poller.start,
                name="zenin-queue-poller",
                daemon=True,
            )
            poller_thread.start()
            logger.info("[ML-SERVICE] Zenin Queue Poller started (daemon thread)")
        except Exception as e:
            logger.warning("[ML-SERVICE] Zenin Queue Poller failed to start: %s", str(e))
    else:
        logger.info("[ML-SERVICE] Zenin Queue Poller disabled (ZENIN_QUEUE_POLLER_ENABLED != true)")
    
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


app = FastAPI(
    title="IoT ML Service",
    version="0.2.0",
    description="Machine Learning service for IoT sensor predictions",
    lifespan=lifespan,
)


# Import and include router after app creation to avoid circular imports
from .api.routes import router
app.include_router(router)


# Health endpoint at root level for backwards compatibility
@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "iot-ml-service", "version": "0.2.0", "status": "ok"}
