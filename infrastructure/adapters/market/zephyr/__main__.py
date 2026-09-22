"""Executable module entrypoint for Zephyr: python -m zephyr [args]."""

from __future__ import annotations

import sys
from pathlib import Path
# El modulo main para iniciarlo 
# Ensure workspace and iot_machine_learning roots are in sys.path
_IOT_DIR = Path(__file__).resolve().parents[4]
_ST_ROOT = _IOT_DIR.parent
for p in (str(_ST_ROOT), str(_IOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from iot_machine_learning.infrastructure.adapters.market.zephyr.cli import main

if __name__ == "__main__":
    main()
