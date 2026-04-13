"""FastAPI dependencies for ML service.

Provides database connections and other shared dependencies.
"""

from __future__ import annotations

from typing import Annotated, Generator

from fastapi import Depends, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from sqlalchemy.engine import Connection

from iot_ingest_services.common.db import get_engine
from ..config.loader import get_feature_flags


def get_db_conn() -> Generator[Connection, None, None]:
    """Dependencia FastAPI para obtener una conexión SQLAlchemy.

    Usa el mismo engine que iot_ingest_services.common.db.
    """
    engine = get_engine()
    with engine.begin() as conn:
        yield conn


DbConnDep = Annotated[Connection, Depends(get_db_conn)]

# API Key security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key from X-API-Key header.

    In production, rotate keys via ML_API_KEY env var.
    This is a minimal gate — not a substitute for full auth (see Phase 2).
    """
    flags = get_feature_flags()
    expected = getattr(flags, "ML_API_KEY", None)

    if not expected:
        # If flag not set, auth is disabled (dev mode)
        return "dev-mode"

    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
