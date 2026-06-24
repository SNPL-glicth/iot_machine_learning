"""Predict Phase — parallel engines with per-engine timeout and Redis metrics.

Runs capable engines concurrently via ThreadPoolExecutor (max_workers=4,
per-engine timeout=150ms).  Failed or timed-out engines are recorded
as failures and do NOT block the pipeline (fail-open).

Redis metrics:
  zenin:metrics:engine_time:{engine}  — per-engine execution time (ms)
  zenin:metrics:engine_failures:{engine} — cumulative failure counter
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from . import PipelineContext

from ...explanation import ExplanationBuilder
from ...perception.helpers import consume_engine_failures
from ...perception.failure_collector import _record_failure
from ...perception.engine_runner import _run_engine_with_timeout
from ..fallback_handler import handle_fallback

logger = logging.getLogger(__name__)

_ENGINE_TIMEOUT_MS = 150
_MAX_WORKERS = 4


def _redis_metrics(ctx: PipelineContext) -> Any:
    store = getattr(ctx.orchestrator, "_series_values_store", None)
    if store is not None:
        return getattr(store, "_redis", None)
    return None


def _record_engine_time(redis: Any, engine: str, elapsed_ms: float) -> None:
    if redis is None:
        return
    try:
        key = f"zenin:metrics:engine_time:{engine}"
        redis.setex(key, 3600, str(round(elapsed_ms, 1)))
    except Exception:
        pass


def _record_engine_failure(redis: Any, engine: str) -> None:
    if redis is None:
        return
    try:
        key = f"zenin:metrics:engine_failures:{engine}"
        redis.incr(key)
    except Exception:
        pass


def _collect_perceptions_parallel(
    engines: List[Any],
    values: List[float],
    timestamps: Optional[List[float]],
    redis: Any,
) -> List[Any]:
    """Run engines in parallel with per-engine timeout, fail-open."""
    from ...analysis.types import EnginePerception

    out: Dict[int, EnginePerception] = {}
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(engines))) as pool:
        future_map = {}
        for idx, eng in enumerate(engines):
            t0 = time.monotonic()
            future = pool.submit(eng.predict, values, timestamps)
            future_map[future] = (idx, eng.name, t0)

        for future, (idx, name, t0) in future_map.items():
            elapsed_ms = (time.monotonic() - t0) * 1000
            _record_engine_time(redis, name, elapsed_ms)
            try:
                result = future.result(timeout=_ENGINE_TIMEOUT_MS / 1000)
                out[idx] = EnginePerception(
                    engine_name=name,
                    predicted_value=result.predicted_value,
                    confidence=result.confidence,
                    trend=result.trend,
                )
            except FutureTimeoutError:
                logger.warning(
                    "engine_timeout",
                    extra={"engine": name, "timeout_ms": _ENGINE_TIMEOUT_MS},
                )
                _record_failure(name, "timeout")
                _record_engine_failure(redis, name)
            except Exception as e:
                logger.warning(
                    "engine_failed",
                    extra={"engine": name, "error": str(e)},
                )
                _record_failure(name, "exception", str(e))
                _record_engine_failure(redis, name)

    return [out[i] for i in sorted(out)]


class PredictPhase:
    """Phase 2: parallel engine perception collection."""

    @property
    def name(self) -> str:
        return "predict"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        orchestrator = ctx.orchestrator
        redis = _redis_metrics(ctx)

        builder = ExplanationBuilder(ctx.series_id)
        if ctx.profile:
            builder.set_signal(ctx.profile)

        # Sensor profile + structural engine filter
        sensor_profile = None
        repo = getattr(orchestrator, "_sensor_profile_repository", None)
        if repo and ctx.series_id != "unknown":
            try:
                sensor_profile = repo.get_by_series_id(ctx.series_id)
            except Exception as e:
                logger.debug(f"predict_phase_profile_load_failed: {e}")

        ineligible: set[str] = set()
        if sensor_profile is not None:
            ef = getattr(orchestrator, "_engine_filter", None)
            if ef:
                ineligible = ef.get_ineligible_engines(sensor_profile.equipment_class)
                if ineligible:
                    logger.debug(
                        "structural_engine_filter",
                        extra={
                            "series": ctx.series_id,
                            "eq": sensor_profile.equipment_class.value,
                            "in": list(ineligible),
                        },
                    )

        engines = [e for e in orchestrator._engines if e.name not in ineligible]

        # Budget check before execution
        if ctx.timer.total_ms > ctx.timer.budget_ms:
            logger.warning(
                "pipeline_budget_exceeded_before_predict",
                extra={
                    "elapsed_ms": round(ctx.timer.total_ms, 2),
                    "budget_ms": ctx.timer.budget_ms,
                },
            )
            result, diag, expl, reg, perc = handle_fallback(
                ctx.values, ctx.profile, builder, ctx.timer, "budget_exceeded",
            )
            return ctx.with_field(
                is_fallback=True,
                fallback_reason="budget_exceeded_before_predict",
                diagnostic=diag,
                explanation=expl,
                engine_failures={},
                metadata={
                    "cognitive_diagnostic": diag.to_dict() if diag else None,
                    "explanation": expl.to_dict() if expl else None,
                    "engine_failures": {},
                },
            )

        # Parallel perception collection
        capable = [e for e in engines if e.can_handle(len(ctx.values or []))]
        for e in engines:
            if e not in capable:
                _record_failure(e.name, "cannot_handle")

        if not capable:
            result, diag, expl, reg, perc = handle_fallback(
                ctx.values, ctx.profile, builder, ctx.timer, "no_valid_perceptions",
            )
            return ctx.with_field(
                is_fallback=True,
                fallback_reason="no_valid_perceptions",
                diagnostic=diag,
                explanation=expl,
                engine_failures=consume_engine_failures(),
                metadata={
                    "cognitive_diagnostic": diag.to_dict() if diag else None,
                    "explanation": expl.to_dict() if expl else None,
                    "engine_failures": consume_engine_failures(),
                },
            )

        perceptions = _collect_perceptions_parallel(
            capable, ctx.values, ctx.timestamps, redis,
        )
        engine_failures = consume_engine_failures()

        if not perceptions:
            result, diag, expl, reg, perc = handle_fallback(
                ctx.values, ctx.profile, builder, ctx.timer, "no_valid_perceptions",
            )
            return ctx.with_field(
                is_fallback=True,
                fallback_reason="no_valid_perceptions",
                diagnostic=diag,
                explanation=expl,
                engine_failures=engine_failures,
                metadata={
                    "cognitive_diagnostic": diag.to_dict() if diag else None,
                    "explanation": expl.to_dict() if expl else None,
                    "engine_failures": engine_failures,
                },
            )

        # Cold start weights
        COLD_START_THRESHOLD = 50
        n_points = len(ctx.values or [])
        plasticity_weights = None
        if sensor_profile is not None and n_points < COLD_START_THRESHOLD:
            wi = getattr(orchestrator, "_weight_initializer", None)
            if wi:
                names = [p.engine_name for p in perceptions]
                cold = wi.get_initial_weights(
                    sensor_profile.equipment_class,
                    [n for n in names if n not in ineligible],
                )
                base = {}
                wr = getattr(orchestrator, "_weight_resolver", None)
                if wr and hasattr(wr, "_base_weights"):
                    base = {k: v for k, v in wr._base_weights.items() if k in names}
                for n in names:
                    base.setdefault(n, 1.0 / len(names))
                b = n_points / COLD_START_THRESHOLD
                for n in base:
                    if n in cold:
                        base[n] = (1 - b) * cold[n] + b * base[n]
                plasticity_weights = base
                logger.debug(
                    "cold_start_weights",
                    extra={"series": ctx.series_id, "n": n_points, "blend": round(b, 2)},
                )

        builder.set_perceptions(perceptions, n_engines_total=len(engines))

        # Metrics
        if ctx.metrics_collector is not None:
            try:
                for perception in perceptions:
                    ctx.metrics_collector.record_retrieval(
                        hit=True,
                        similarity=perception.confidence,
                    )
            except Exception as e:
                logger.debug(f"metrics_collection_failed: {e}")

        return ctx.with_field(
            perceptions=perceptions,
            explanation=builder,
            engine_failures=engine_failures,
            plasticity_weights=plasticity_weights,
        )
