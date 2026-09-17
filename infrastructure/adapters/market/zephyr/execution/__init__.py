"""Unified execution subpackage for Zephyr backend."""

from __future__ import annotations

from iot_machine_learning.infrastructure.adapters.market.zephyr.execution.gating import (
    can_execute,
    get_current_mid,
    log_execution,
    perform_shutdown,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution.lifecycle import (
    create_account,
    create_default_master_orchestrator,
    create_feed,
    create_order_client,
    install_signal_handlers,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution.pipeline import (
    extract_decision_trace_metrics,
    process_observation_pipeline,
    resolve_plan_action_trigger,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution.session import (
    check_market_session,
    sync_and_check_health,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution.storage import (
    load_state,
    save_state,
)

__all__ = [
    "can_execute",
    "get_current_mid",
    "log_execution",
    "perform_shutdown",
    "create_account",
    "create_default_master_orchestrator",
    "create_feed",
    "create_order_client",
    "install_signal_handlers",
    "process_observation_pipeline",
    "resolve_plan_action_trigger",
    "extract_decision_trace_metrics",
    "check_market_session",
    "sync_and_check_health",
    "load_state",
    "save_state",
]
