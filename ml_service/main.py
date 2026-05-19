"""IoT ML Service — FastAPI Application entry point.

Lifespan logic lives in lifespan.py.
API routes live in api/routes*.py.
This file is only the app creation and middleware setup.
"""
from __future__ import annotations

import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv()

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .lifespan import lifespan

logger = logging.getLogger(__name__)

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

# CORS configuration
from .config.loader import get_feature_flags
_flags = get_feature_flags()
ALLOWED_ORIGINS = os.getenv("ML_CORS_ORIGINS", _flags.ML_CORS_ORIGINS).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Include routers
from .api.routes import router
from .api.routes_cognitive import router as cognitive_router

app.include_router(router)
app.include_router(cognitive_router)

try:
    from .api.routes_governance import router as governance_router
    app.include_router(governance_router)
except ImportError:
    pass


@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "iot-ml-service", "version": "0.2.0", "status": "ok"}
