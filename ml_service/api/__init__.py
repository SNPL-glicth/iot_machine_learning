"""API module for ML service.

Modular FastAPI endpoints and dependencies.
"""

from .dependencies import get_db_conn, DbConnDep
from .schemas import PredictRequest, PredictResponse
from .routes import router

__all__ = [
    "get_db_conn",
    "DbConnDep",
    "PredictRequest",
    "PredictResponse",
    "router",
]
