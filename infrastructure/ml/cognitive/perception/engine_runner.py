"""Single-engine execution with optional individual timeout."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, List, Optional

from ..analysis.types import EnginePerception
from ...interfaces import PredictionResult

from .failure_collector import _record_failure

logger = logging.getLogger(__name__)


def _run_engine(
    eng: Any, values: List[float], timestamps: Optional[List[float]]
) -> Optional[EnginePerception]:
    """Call ``eng.predict`` and convert the result to EnginePerception.

    Returns ``None`` on exception (logged + recorded as failure).
    """
    try:
        r = eng.predict(values, timestamps)
    except Exception as exc:
        logger.warning("engine_failed", extra={"engine": eng.name, "error": str(exc)})
        _record_failure(eng.name, "exception", str(exc))
        return None
    d = r.metadata.get("diagnostic", {}) or {}
    return EnginePerception(
        engine_name=eng.name,
        predicted_value=r.predicted_value,
        confidence=r.confidence,
        trend=r.trend,
        stability=d.get("stability_indicator", 0.0) if isinstance(d, dict) else 0.0,
        local_fit_error=d.get("local_fit_error", 0.0) if isinstance(d, dict) else 0.0,
        metadata=r.metadata,
    )


def _run_engine_with_timeout(
    eng: Any,
    values: List[float],
    timestamps: Optional[List[float]],
    timeout_ms: float,
) -> Optional[EnginePerception]:
    """Ejecuta un engine con timeout individual.

    Usa ThreadPoolExecutor(max_workers=1) para aislar el engine
    y poder abandonar el future si excede el timeout.
    """
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine_single") as ex:
        future = ex.submit(_run_engine, eng, values, timestamps)
        try:
            return future.result(timeout=timeout_ms / 1000.0)
        except FuturesTimeoutError:
            future.cancel()
            logger.warning(
                "engine_timeout",
                extra={"engine": eng.name, "timeout_ms": timeout_ms},
            )
            _record_failure(eng.name, "timeout")
            return None
        except Exception as exc:
            logger.warning(
                "engine_failed",
                extra={"engine": eng.name, "error": str(exc)},
            )
            _record_failure(eng.name, "exception", str(exc))
            return None
