"""Orchestrator perception helpers — thin shell.

Delegates to extracted modules to keep this file under 180 lines.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ..analysis.types import EnginePerception

from .env_knobs import (
    ML_ENGINE_TIMEOUT_MS,
    ML_PREDICT_ENGINE_TIMEOUT_MS,
    ML_PREDICT_MAX_WORKERS,
)
from .failure_collector import _clear_failures, _record_failure, consume_engine_failures
from .engine_runner import _run_engine_with_timeout
from .fallback import create_fallback_result

logger = logging.getLogger(__name__)


def collect_perceptions(
    engines: List,
    values: List[float],
    timestamps: Optional[List[float]],
) -> List[EnginePerception]:
    """Collect predictions from all capable engines.

    Dispatches between sequential and parallel modes based on
    ``ML_PREDICT_MAX_WORKERS`` and the number of capable engines.
    Falls back to sequential on any ThreadPoolExecutor failure.
    """
    _clear_failures()

    capable = [e for e in engines if e.can_handle(len(values))]
    for e in engines:
        if e not in capable:
            _record_failure(e.name, "cannot_handle")

    if len(capable) <= 1 or ML_PREDICT_MAX_WORKERS <= 1:
        return _collect_perceptions_sequential(capable, values, timestamps)

    try:
        return _collect_perceptions_parallel(
            capable,
            values,
            timestamps,
            max_workers=min(ML_PREDICT_MAX_WORKERS, len(capable)),
            timeout_ms=ML_PREDICT_ENGINE_TIMEOUT_MS,
        )
    except Exception as exc:
        logger.warning(
            "parallel_execution_failed",
            extra={"error": str(exc), "fallback": "sequential"},
        )
        return _collect_perceptions_sequential(capable, values, timestamps)


def _collect_perceptions_sequential(
    engines: List,
    values: List[float],
    timestamps: Optional[List[float]],
) -> List[EnginePerception]:
    """Single-threaded loop with per-engine wall-clock timeout."""
    out: List[EnginePerception] = []
    for eng in engines:
        if not eng.can_handle(len(values)):
            _record_failure(eng.name, "cannot_handle")
            continue
        p = _run_engine_with_timeout(
            eng, values, timestamps, timeout_ms=ML_ENGINE_TIMEOUT_MS,
        )
        if p is not None:
            out.append(p)
    return out


def _collect_perceptions_parallel(
    engines: List,
    values: List[float],
    timestamps: Optional[List[float]],
    *,
    max_workers: int,
    timeout_ms: Optional[float] = None,
) -> List[EnginePerception]:
    """Run capable engines concurrently with per-engine timeout.

    Returns results in input-engine order (not completion order).
    Engines already pre-filtered by ``can_handle`` on the caller side.
    """
    if timeout_ms is None:
        timeout_ms = ML_PREDICT_ENGINE_TIMEOUT_MS

    results: Dict[int, EnginePerception] = {}
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="engine"
    ) as pool:
        future_to_idx = {
            pool.submit(
                _run_engine_with_timeout,
                eng, values, timestamps,
                timeout_ms,
            ): (idx, eng.name)
            for idx, eng in enumerate(engines)
        }
        for future, (idx, name) in future_to_idx.items():
            try:
                perception = future.result()
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "engine_future_failed", extra={"engine": name, "error": str(exc)}
                )
                _record_failure(name, "exception", str(exc))
                continue
            if perception is not None:
                results[idx] = perception

    return [results[i] for i in sorted(results)]


__all__ = [
    "ML_PREDICT_MAX_WORKERS",
    "ML_PREDICT_ENGINE_TIMEOUT_MS",
    "collect_perceptions",
    "consume_engine_failures",
    "create_fallback_result",
    "_collect_perceptions_parallel",
    "_collect_perceptions_sequential",
]
