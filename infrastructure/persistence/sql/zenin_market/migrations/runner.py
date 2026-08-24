"""Aplica migraciones SQL de zenin_market de forma idempotente.

Sin Alembic (convención del proyecto: SQL crudo). Cada archivo
migrations/NNN_*.sql se ejecuta una sola vez, registrado en
schema_migrations. El lote completo es atómico: si una migración
falla, se hace rollback y no se registra avance.

Uso:
    python -m infrastructure.persistence.sql.zenin_market.migrations.runner
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import text

from ..market_db_connection import ZeninMarketDbConnection

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent


def apply_migrations(dry_run: bool = False) -> list[str]:
    """Aplica las migraciones pendientes.

    Returns:
        Lista de versiones aplicadas (vacía si no había pendientes).
    """
    engine = ZeninMarketDbConnection.get_engine()
    applied: list[str] = []

    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version VARCHAR(128) NOT NULL PRIMARY KEY,"
                " applied_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)"
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
            )
        )
        done = {
            row[0]
            for row in conn.execute(text("SELECT version FROM schema_migrations"))
        }

        for path in sorted(_MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql")):
            version = path.stem
            if version in done:
                continue
            script = path.read_text(encoding="utf-8")
            if dry_run:
                logger.info("migration_pending", extra={"version": version})
                continue
            for stmt in _split_statements(script):
                conn.execute(text(stmt))
            conn.execute(
                text("INSERT INTO schema_migrations (version) VALUES (:v)"),
                {"v": version},
            )
            applied.append(version)
            logger.info("migration_applied", extra={"version": version})

    return applied


def _split_statements(script: str) -> list[str]:
    """Divide un script SQL en statements simples (soporta comentarios --)."""
    stmts: list[str] = []
    buf: list[str] = []
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("--"):
            continue
        buf.append(raw_line)
        if line.endswith(";"):
            stmts.append("\n".join(buf))
            buf = []
    if buf:
        stmts.append("\n".join(buf))
    return stmts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    applied = apply_migrations()
    print(f"Applied: {applied if applied else 'nothing (up to date)'}")
