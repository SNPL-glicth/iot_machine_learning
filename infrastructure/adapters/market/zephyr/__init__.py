"""Zephyr Institutional Algorithmic Execution System.

Decoupled execution backend obeying ZENIN Master Equation sovereignty.
"""

from __future__ import annotations

from iot_machine_learning.infrastructure.adapters.market.zephyr.config import (
    LiveBotConfig,
    load_zephyr_config,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution import (
    can_execute,
    perform_shutdown,
    process_observation_pipeline,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.master_engine_adapter import (
    MasterEngineAdapter,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.models import (
    ExecutionContext,
    LiveBotState,
)
import sys
from iot_machine_learning.infrastructure.adapters.market.zephyr.runners import (
    LiveBotRunner,
    PaperBotRunner,
    create_live_bot,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.runners import paper_runner as _paper_runner

sys.modules["iot_machine_learning.infrastructure.adapters.market.zephyr.paper_runner"] = _paper_runner

__all__ = [
    "LiveBotRunner",
    "PaperBotRunner",
    "create_live_bot",
    "LiveBotConfig",
    "load_zephyr_config",
    "LiveBotState",
    "ExecutionContext",
    "MasterEngineAdapter",
    "can_execute",
    "process_observation_pipeline",
    "perform_shutdown",
]
