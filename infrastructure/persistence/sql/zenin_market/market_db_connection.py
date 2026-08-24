"""Conexión SEPARADA a zenin_market (MySQL) — NO tocar la conexión IoT/mssql.

Módulo hermano de zenin_db_connection.py: replica el mismo patrón
(singleton SQLAlchemy, pool configurable, pre_ping, health check SELECT 1,
contextmanager transaccional) pero con driver MySQL/PyMySQL para la base
zenin_market del proyecto ZENIN Market.

Variables de entorno (todas con prefijo MYSQL_, ver .env.example):
- MYSQL_HOST          (default: localhost)
- MYSQL_PORT          (default: 3306)
- MYSQL_DATABASE      (default: zenin_market)
- MYSQL_USER          (default: zenin)
- MYSQL_PASSWORD      (REQUERIDA, sin default seguro)
- MYSQL_POOL_SIZE / MYSQL_MAX_OVERFLOW / MYSQL_POOL_TIMEOUT /
  MYSQL_POOL_RECYCLE / MYSQL_CONNECT_TIMEOUT
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

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
            "zenin_market_env_invalid",
            extra={"env_var": name, "raw_value": raw, "using_default": default},
        )
        return default


class ZeninMarketDbConnection:
    """Manages connection to zenin_market (MySQL, separate from IoT stacks)."""

    _engine: Engine | None = None

    @classmethod
    def get_engine(cls) -> Engine:
        """Return singleton engine for zenin_market."""
        if cls._engine is None:
            cls._engine = cls._create_engine()
        return cls._engine

    @classmethod
    def _create_engine(cls) -> Engine:
        """Create engine with connection string to zenin_market (pymysql)."""
        host = os.environ.get("MYSQL_HOST", "localhost")
        port = os.environ.get("MYSQL_PORT", "3306")
        database = os.environ.get("MYSQL_DATABASE", "zenin_market")
        user = os.environ.get("MYSQL_USER", "zenin")
        password = os.environ.get("MYSQL_PASSWORD", "")

        conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"

        # PERF: Pool sizes configurable, mismo criterio que ZeninDbConnection.
        pool_size = _env_int("MYSQL_POOL_SIZE", 10)
        max_overflow = _env_int("MYSQL_MAX_OVERFLOW", 20)
        pool_timeout = _env_int("MYSQL_POOL_TIMEOUT", 30)
        pool_recycle = _env_int("MYSQL_POOL_RECYCLE", 300)
        connect_timeout = _env_int("MYSQL_CONNECT_TIMEOUT", 10)

        engine = create_engine(
            conn_str,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_pre_ping=True,
            pool_recycle=pool_recycle,
            connect_args={"connect_timeout": connect_timeout},
        )

        logger.info(
            "zenin_market_engine_created",
            extra={
                "host": host,
                "port": port,
                "database": database,
                "user": user,
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
            logger.error("[ZENIN_MARKET] Transaction rolled back: %s", e)
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
            logger.warning("[ZENIN_MARKET] Health check failed: %s", e)
            return False

    @classmethod
    def dispose(cls) -> None:
        """Dispose the engine pool."""
        if cls._engine is not None:
            cls._engine.dispose()
            cls._engine = None
            logger.info("[ZENIN_MARKET] Engine disposed")
