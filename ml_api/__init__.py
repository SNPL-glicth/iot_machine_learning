"""ml_api — Facade de entrada HTTP para el servicio ML.

Expone create_app() que construye la aplicación FastAPI con las rutas
del servicio ML (/health, /ml/predict, etc.).
"""

from __future__ import annotations


def create_app():
    """Crea y configura la aplicación FastAPI del servicio ML.

    Returns:
        FastAPI application con rutas /health y /ml/predict.
    """
    from fastapi import FastAPI

    app = FastAPI(title="IoT ML Service", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    try:
        from iot_machine_learning.ml_service.api import router as ml_router
        app.include_router(ml_router, prefix="/ml")
    except Exception:
        pass

    return app


__all__ = ["create_app"]
