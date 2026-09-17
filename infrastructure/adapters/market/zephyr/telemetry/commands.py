"""Command dispatcher and live config modification handlers for telemetry WebSocket.

Provides secure configuration modification with strict credential segregation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, cast

logger = logging.getLogger(__name__)

SECRET_KEYS = (
    "alpaca_api_key",
    "alpaca_secret_key",
    "binance_api_key",
    "binance_api_secret",
    "weaviate_api_key",
)


def extract_bot_config_dict(config: Any) -> Dict[str, Any]:
    """Extrae todos los campos de configuración del bot en un diccionario sanitizado."""
    if not config:
        return {}
    if hasattr(config, "to_safe_dict"):
        return cast(Dict[str, Any], config.to_safe_dict())

    if hasattr(config, "__dict__"):
        res = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    elif isinstance(config, dict):
        res = dict(config)
    else:
        res = {}

    for sec in SECRET_KEYS:
        if sec in res and res[sec]:
            res[sec] = "***REDACTED***"
    return res


def apply_bot_config_update(runner: Any, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Aplica actualizaciones de configuración en caliente y persiste de forma segregada."""
    if not hasattr(runner, "config") or runner.config is None:
        raise ValueError("Runner no posee un objeto de configuración activo")

    config = runner.config
    applied_changes = {}

    for key, value in updates.items():
        if key.startswith("_"):
            continue
        if hasattr(config, key):
            old_val = getattr(config, key)
            try:
                if isinstance(old_val, bool) and not isinstance(value, bool):
                    typed_val = str(value).lower() in ("true", "1", "yes")
                elif isinstance(old_val, int) and not isinstance(value, int):
                    typed_val = int(value)
                elif isinstance(old_val, float) and not isinstance(value, float):
                    typed_val = float(value)
                elif isinstance(old_val, list) and isinstance(value, str):
                    typed_val = [s.strip().upper() for s in value.split(",") if s.strip()]
                else:
                    typed_val = value
                setattr(config, key, typed_val)
                applied_changes[key] = typed_val
            except Exception as e:
                logger.error("Error cast config key %s to %s: %s", key, type(old_val), e)

    # Sync environment variables if secrets are provided
    if "alpaca_api_key" in applied_changes and applied_changes["alpaca_api_key"]:
        os.environ["ALPACA_API_KEY"] = str(applied_changes["alpaca_api_key"])
    if "alpaca_secret_key" in applied_changes and applied_changes["alpaca_secret_key"]:
        os.environ["ALPACA_SECRET_KEY"] = str(applied_changes["alpaca_secret_key"])

    # Persist public configuration changes (CERO SECRETOS en archivos públicos)
    public_dict = config.to_public_dict() if hasattr(config, "to_public_dict") else {
        k: v for k, v in config.__dict__.items() if k not in SECRET_KEYS and not k.startswith("_")
    }

    save_paths = [
        Path("config/zephyr_config.json"),
        Path("iot_machine_learning/config/zephyr_config.json"),
    ]
    for p in save_paths:
        try:
            if p.exists() or p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(public_dict, indent=2), encoding="utf-8")
                logger.info("Persisted sanitized public live config to %s", p)
        except Exception as e:
            logger.warning("Could not persist config to %s: %s", p, e)

    # Sanitize return dictionary so caller does not receive raw secrets
    sanitized_applied = {
        k: ("***REDACTED***" if k in SECRET_KEYS else v)
        for k, v in applied_changes.items()
    }
    return sanitized_applied


async def handle_emergency_flush(runner: Any, symbol: str | None = None) -> None:
    """Ejecuta flush de emergencia sobre el runner y el execution handler."""
    if not hasattr(runner, "_handler") or not runner._handler:
        raise RuntimeError("Runner has no execution handler")

    symbols = [symbol.upper()] if symbol else list(getattr(runner, "_symbols", []))
    if not symbols and hasattr(runner, "config") and hasattr(runner.config, "symbol"):
        symbols = [runner.config.symbol]
    if not symbols:
        symbols = ["SPY"]

    logger.warning("Executing emergency flush for symbols: %s", symbols)
    errors = []
    for sym in symbols:
        try:
            await runner._handler.trigger_emergency_flush("TUI Emergency Flush", symbol=sym)
        except Exception as e:
            logger.critical("Failed to flush %s: %s", sym, e, exc_info=True)
            errors.append(f"{sym}: {e}")

    if errors:
        raise RuntimeError(f"Emergency flush failed for: {', '.join(errors)}")
