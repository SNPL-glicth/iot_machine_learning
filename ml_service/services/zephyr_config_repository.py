"""Repository for Zephyr JSON configurations, profiles, and credentials."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


class ZephyrConfigRepository:
    """Handles read/write operations on Zephyr JSON files."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = (
                Path(__file__).resolve().parents[2]
                / "infrastructure" / "adapters" / "market" / "zephyr" / "config"
            )
        self.main_config_path = self.config_dir / "zephyr_config.json"
        self.secrets_path = self.config_dir / "zephyr_secrets.json"

    def read_active_config(self) -> Dict[str, Any]:
        """Reads active configuration from zephyr_config.json."""
        if not self.main_config_path.exists():
            return {}
        try:
            with open(self.main_config_path, "r", encoding="utf-8") as f:
                return cast(Dict[str, Any], json.load(f))
        except Exception as e:
            logger.error("Failed to read %s: %s", self.main_config_path, e)
            return {}

    def save_active_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Applies updates to zephyr_config.json and persists."""
        current = self.read_active_config()
        current.update(updates)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.main_config_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        return current

    def list_accounts(self) -> List[Dict[str, Any]]:
        """Lists account profiles discovered from JSON files."""
        accounts: List[Dict[str, Any]] = []
        cfg = self.read_active_config()
        act_broker, act_sym, act_test = cfg.get("broker", "alpaca"), cfg.get("symbol", "SPY"), cfg.get("testnet", True)

        for p in cfg.get("profiles", []):
            pid = p.get("profile_name") or p.get("id")
            if not pid:
                continue
            is_active = (p.get("broker") == act_broker and p.get("symbol") == act_sym and p.get("testnet") == act_test)
            mode = "paper" if p.get("testnet", True) else "live"
            accounts.append({
                "id": pid, "name": p.get("name", f"{p.get('broker', '').capitalize()} ({mode.capitalize()})"),
                "mode": mode, "broker": p.get("broker", "alpaca"), "symbol": p.get("symbol", "SPY"),
                "symbols": p.get("symbols", [p.get("symbol", "SPY")]),
                "accountNumber": p.get("accountNumber", f"ACC-{p.get('broker', 'ALP')[:3].upper()}-{pid[:4].upper()}"),
                "isActive": is_active, "source": "zephyr_config",
            })

        if self.config_dir.exists():
            for path in sorted(self.config_dir.glob("*.json")):
                if path.name.endswith("secrets.json") or path.name.endswith(".template") or path.name == "zephyr_config.json":
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    pid = path.stem
                    sym = data.get("symbol", "SPY")
                    brk = data.get("broker") or ("binance" if ("USDT" in sym or "BTC" in sym) else "alpaca")
                    test = data.get("testnet", True)
                    mode = "paper" if test else "live"
                    is_active = (brk == act_broker and sym == act_sym and test == act_test)
                    accounts.append({
                        "id": pid, "name": f"{brk.capitalize()} {mode.capitalize()} ({sym})",
                        "mode": mode, "broker": brk, "symbol": sym, "symbols": data.get("symbols", [sym]),
                        "accountNumber": f"ACC-{brk[:3].upper()}-{pid[:4].upper()}",
                        "isActive": is_active, "source": path.name,
                    })
                except Exception as ex:
                    logger.debug("Could not read profile %s: %s", path.name, ex)

        if not any(a.get("isActive") for a in accounts) and accounts:
            accounts[0]["isActive"] = True
        return accounts

    def create_account_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates and stores a new account profile into zephyr_config.json."""
        active_cfg = self.read_active_config()
        profiles = list(active_cfg.get("profiles", []))
        brk = data.get("broker", "alpaca").lower()
        mode = data.get("mode", "paper").lower()
        testnet = (mode == "paper")
        sym = data.get("symbol", "SPY").upper()
        pid = data.get("id") or f"{brk}_{mode}_{sym.lower()}"
        name = data.get("name") or f"{brk.capitalize()} ({mode.capitalize()})"
        acc_num = data.get("accountNumber") or f"ACC-{brk[:3].upper()}-{pid[:4].upper()}"

        new_profile = {
            "profile_name": pid, "id": pid, "name": name, "broker": brk,
            "mode": mode, "testnet": testnet, "symbol": sym, "symbols": data.get("symbols") or [sym],
            "accountNumber": acc_num,
        }

        api_k, sec_k = data.get("api_key"), data.get("secret_key")
        if api_k or sec_k:
            self._save_credentials(pid, brk, api_k, sec_k)

        profiles = [p for p in profiles if p.get("id") != pid and p.get("profile_name") != pid]
        profiles.append(new_profile)
        active_cfg["profiles"] = profiles

        if data.get("set_active", False):
            active_cfg.update({"broker": brk, "testnet": testnet, "symbol": sym, "symbols": [sym]})
        self.save_active_config(active_cfg)
        return new_profile

    def activate_profile(self, profile_id: str) -> Dict[str, Any]:
        """Activates a profile by id and updates zephyr_config.json."""
        accounts = self.list_accounts()
        target = next((a for a in accounts if a["id"] == profile_id), None)
        if not target:
            raise ValueError(f"Profile {profile_id} not found")
        updates = {
            "broker": target["broker"], "symbol": target["symbol"],
            "symbols": target.get("symbols", [target["symbol"]]), "testnet": (target["mode"] == "paper"),
        }
        self.save_active_config(updates)
        return updates

    def _save_credentials(self, pid: str, broker: str, api_k: Optional[str], sec_k: Optional[str]) -> None:
        secrets: Dict[str, Any] = {}
        if self.secrets_path.exists():
            try:
                with open(self.secrets_path, "r", encoding="utf-8") as f:
                    secrets = json.load(f)
            except Exception:
                pass
        prof_secrets = dict(secrets.get("profiles", {}))
        prof_secrets[pid] = {"broker": broker, "api_key": api_k or "", "secret_key": sec_k or ""}
        secrets["profiles"] = prof_secrets
        with open(self.secrets_path, "w", encoding="utf-8") as f:
            json.dump(secrets, f, indent=2)
