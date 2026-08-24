"""Integration tests — circuito mínimo dominio → persistencia → dominio.

FASES 0–2: verifica que la conexión MySQL (zenin_market), la migración
inicial y el repo ProviderProfileRepository funcionan de extremo a extremo
usando la entidad de dominio real (domain/entities/market/capability.py).
El mapeo fila ↔ entidad vive en el test, no en el dominio (regla: el
dominio no conoce infraestructura).

Requiere MySQL corriendo (contenedor `mysql`, puerto 3306) y las variables
MYSQL_* en el entorno (ver .env).
"""

from __future__ import annotations

import json

import pytest
from iot_machine_learning.domain.entities.market import Capability, ProviderProfile
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_db_connection import (
    ZeninMarketDbConnection,
)

pytestmark = pytest.mark.integration


def _profile_to_row(profile: ProviderProfile) -> dict:
    """Serialización para la tabla provider_profiles (solo en el test)."""
    return {
        "provider": profile.provider,
        "asset_class": profile.asset_class,
        "capabilities": json.dumps(
            sorted(cap.value for cap in profile.capabilities)
        ),
        "max_ws_symbols": profile.max_ws_symbols,
    }


def _profile_from_row(row) -> ProviderProfile:
    """Deserialización desde la tabla provider_profiles (solo en el test)."""
    return ProviderProfile(
        provider=row["provider"],
        asset_class=row["asset_class"],
        capabilities=frozenset(Capability(cap) for cap in json.loads(row["capabilities"])),
        max_ws_symbols=row["max_ws_symbols"],
    )


@pytest.fixture(scope="module")
def engine():
    if not ZeninMarketDbConnection.health_check():
        pytest.skip("MySQL zenin_market no disponible")
    return ZeninMarketDbConnection.get_engine()


class TestProviderProfilesCircuit:
    """Round-trip: ProviderProfile -> MySQL -> ProviderProfile."""

    def test_round_trip(self, engine):
        from sqlalchemy import text

        profile = ProviderProfile(
            provider="alpaca",
            asset_class="equities",
            capabilities=frozenset(
                {
                    Capability.TRADES,
                    Capability.QUOTES,
                    Capability.CANDLES,
                    Capability.HISTORICAL_TICKS,
                    Capability.REALTIME,
                }
            ),
            max_ws_symbols=30,
        )
        row = _profile_to_row(profile)

        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO provider_profiles (
                        provider, asset_class, capabilities, max_ws_symbols
                    ) VALUES (
                        :provider, :asset_class, :capabilities, :max_ws_symbols
                    )
                    ON DUPLICATE KEY UPDATE
                        capabilities = VALUES(capabilities),
                        max_ws_symbols = VALUES(max_ws_symbols)
                    """
                ),
                row,
            )
            result = conn.execute(
                text(
                    """
                    SELECT provider, asset_class, capabilities, max_ws_symbols
                    FROM provider_profiles
                    WHERE provider = :provider AND asset_class = :asset_class
                    """
                ),
                {"provider": "alpaca", "asset_class": "equities"},
            )
            db_row = result.mappings().one()

        restored = _profile_from_row(dict(db_row))
        assert restored == profile
        assert not restored.has_order_book

    def test_health_check(self):
        assert ZeninMarketDbConnection.health_check() is True

    def test_engine_singleton(self):
        assert (
            ZeninMarketDbConnection.get_engine()
            is ZeninMarketDbConnection.get_engine()
        )
