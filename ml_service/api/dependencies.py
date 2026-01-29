"""FastAPI dependencies for ML service.

Provides database connections and other shared dependencies.
"""

from __future__ import annotations

from typing import Annotated, Generator

from fastapi import Depends
from sqlalchemy.engine import Connection

from iot_ingest_services.common.db import get_engine


def get_db_conn() -> Generator[Connection, None, None]:
    """Dependencia FastAPI para obtener una conexión SQLAlchemy.

    Usa el mismo engine que iot_ingest_services.common.db.
    """
    engine = get_engine()
    with engine.begin() as conn:
        yield conn


DbConnDep = Annotated[Connection, Depends(get_db_conn)]
