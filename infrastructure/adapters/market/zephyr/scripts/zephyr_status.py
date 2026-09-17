#!/usr/bin/env python3
"""Zephyr Account & Positions Status Inspector."""

from __future__ import annotations

import sys
from pathlib import Path

_CUR = Path(__file__).resolve()
_ST_ROOT = next((p for p in _CUR.parents if (p / "iot_machine_learning").is_dir()), _CUR.parents[5])
_IOT_DIR = _ST_ROOT / "iot_machine_learning"

for p in (str(_ST_ROOT), str(_IOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from iot_machine_learning.infrastructure.adapters.market.zephyr.cli import main

if __name__ == "__main__":
    if "--status" not in sys.argv:
        sys.argv.append("--status")
    main()
