"""Fixtures congelados de payloads reales de Alpaca y Binance (FASE 4).

Sin red ni claves API: los tests importan estos JSON desde disco.
Fuente de las muestras: documentación pública de streams y REST de
Alpaca (trades/quotes/bars) y Binance (aggTrade/bookTicker/kline/depth).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def load_payload() -> Callable[[str], dict[str, object]]:
    def _load(name: str) -> dict[str, object]:
        loaded = json.loads((FIXTURES_DIR / name).read_text("utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError(f"fixture {name!r} no es un objeto JSON")
        return cast(dict[str, object], loaded)

    return _load
