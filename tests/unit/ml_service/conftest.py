"""Conftest para tests de ml_service/.

Instala mocks de dependencias pesadas ANTES de que se importen
los módulos bajo test, para evitar ModuleNotFoundError en entornos
sin esas dependencias (sqlalchemy, sklearn, httpx, etc.).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _ensure_mock(module_name: str, submodules: list[str] | None = None) -> None:
    """Registra un mock en sys.modules si el módulo no existe."""
    if module_name not in sys.modules:
        mock = MagicMock()
        sys.modules[module_name] = mock
        for sub in (submodules or []):
            full = f"{module_name}.{sub}"
            if full not in sys.modules:
                sys.modules[full] = MagicMock()


# --- Dependencias pesadas ---

_ensure_mock("sqlalchemy", ["engine", "engine.Connection"])
_ensure_mock("iot_ingest_services", [
    "common", "common.db",
    "ingest_api", "ingest_api.sensor_state",
])
try:
    import sklearn  # noqa: F401 — only mock if not installed
except ImportError:
    _ensure_mock("sklearn", [
        "linear_model", "ensemble", "neighbors",
    ])
_ensure_mock("httpx")
