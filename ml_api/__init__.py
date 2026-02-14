"""ML API facade (E-13).

Re-exports from ml_service.api for independent deployment.
Existing imports via ml_service.api continue to work unchanged.

Usage:
    from ml_api import create_app
    app = create_app()
"""

from __future__ import annotations


def create_app():
    """Create the FastAPI application for ML predictions."""
    from iot_machine_learning.ml_service.api.routes import router
    from fastapi import FastAPI

    app = FastAPI(
        title="ML Prediction API",
        description="On-demand sensor predictions",
        version="1.0.0",
    )
    app.include_router(router)
    return app
