"""Conexión SEPARADA a zenin_db (NO tocar la conexión IoT).

This module provides a singleton SQLAlchemy engine for zenin_db,
used exclusively by the Zenin Queue Poller to read ingestion_queue
and write analysis_results.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Read int from env; log warning and use default on invalid value."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "zenin_db_config_env_invalid",
            extra={"env_var": name, "raw_value": raw, "using_default": default},
        )
        return default


class ZeninDbConnection:
    """Manages connection to zenin_db (separate from iot_monitoring_system)."""

    _engine: Optional[Engine] = None

    @classmethod
    def get_engine(cls) -> Engine:
        """Return singleton engine for zenin_db."""
        if cls._engine is None:
            cls._engine = cls._create_engine()
        return cls._engine

    @classmethod
    def _create_engine(cls) -> Engine:
        """Create engine with connection string to zenin_db (pymssql)."""
        host = os.environ.get("ZENIN_DB_HOST", "localhost")
        port = os.environ.get("ZENIN_DB_PORT", "1434")
        database = os.environ.get("ZENIN_DB_NAME", "zenin_db")
        user = os.environ.get("ZENIN_DB_USER", "sa")
        password = os.environ.get("ZENIN_DB_PASSWORD", "")

        conn_str = (
            f"mssql+pymssql://{user}:{password}@{host}:{port}/{database}"
        )

        # PERF-P0: Pool sizes configurable for 1000-sensor scale.
        # Rule: pool_size >= total_workers * 2 (batch + API + MQTT)
        pool_size = _env_int("ZENIN_DB_POOL_SIZE", 20)
        max_overflow = _env_int("ZENIN_DB_MAX_OVERFLOW", 30)
        pool_timeout = _env_int("ZENIN_DB_POOL_TIMEOUT", 30)
        pool_recycle = _env_int("ZENIN_DB_POOL_RECYCLE", 300)
        connect_timeout = _env_int("ZENIN_DB_CONNECT_TIMEOUT", 10)

        engine = create_engine(
            conn_str,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_pre_ping=True,
            pool_recycle=pool_recycle,
            connect_args={"timeout": connect_timeout},
        )

        logger.info(
            "zenin_db_engine_created",
            extra={
                "host": host,
                "port": port,
                "database": database,
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_timeout": pool_timeout,
                "pool_recycle": pool_recycle,
                "connect_timeout": connect_timeout,
                "max_capacity": pool_size + max_overflow,
            },
        )
        return engine

    @classmethod
    @contextmanager
    def get_connection(cls):
        """Context manager for a transactional connection."""
        engine = cls.get_engine()
        conn = engine.connect()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("[ZENIN_DB] Transaction rolled back: %s", e)
            raise
        finally:
            conn.close()

    @classmethod
    def health_check(cls) -> bool:
        """Quick connectivity check."""
        try:
            with cls.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning("[ZENIN_DB] Health check failed: %s", e)
            return False

    @classmethod
    def dispose(cls) -> None:
        """Dispose the engine pool."""
        if cls._engine is not None:
            cls._engine.dispose()
            cls._engine = None
            logger.info("[ZENIN_DB] Engine disposed")
