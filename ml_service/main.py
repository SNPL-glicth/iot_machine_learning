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

from fastapi import FastAPI
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown events.
    """
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
    
    yield
    
    # Shutdown
    logger.info("[ML-SERVICE] Shutting down...")
    try:
        from .broker import reset_broker
        reset_broker()
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
