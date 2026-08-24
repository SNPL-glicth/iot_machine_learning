"""Configuración de tests de integración ZENIN Market.

Carga el .env del proyecto para que las variables MYSQL_* estén
disponibles en todos los tests del paquete.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config) -> None:  # noqa: ARG001
    _env = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
    load_dotenv(_env, override=True)
