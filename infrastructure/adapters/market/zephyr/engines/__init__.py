"""Zephyr auxiliary execution engines."""

from __future__ import annotations

import sys
from iot_machine_learning.infrastructure.adapters.market.zephyr.engines import (
    rosa_roja_execution,
)

sys.modules[
    "iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja"
] = rosa_roja_execution

__all__ = ["rosa_roja_execution"]
