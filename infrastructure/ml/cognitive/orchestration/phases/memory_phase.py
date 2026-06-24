"""Memory Phase — async Weaviate integration for persistent cognitive memory.

Key design decisions:
  1. Storage (save prediction, anomaly) runs in a background
     ``ThreadPoolExecutor`` so the pipeline never waits on Weaviate.
  2. Similar-case retrieval runs synchronously (fast filtered query)
     and returns ``similar_cases`` to the context for ExplainPhase.
  3. TTL is 90 days — enforced by the ``timestamp`` filter at query time;
     Weaviate's built-in TTL or periodic cleanup handles physical deletion.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from . import PipelineContext

from domain.entities.memory import MemoryEvent

try:
    from iot_machine_learning.infrastructure.ml.cognitive.memory import (
        AnomalyMemoryStore,
    )
except (ImportError, ModuleNotFoundError):
    AnomalyMemoryStore = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

SIMILAR_CASES_LOOKBACK_DAYS = 30
SIMILAR_CASES_TOP_K = 5
STORAGE_TTL_S = 90 * 24 * 3600  # 90 days


@dataclass
class SimilarCase:
    date: str
    param: str
    z_score: float
    regime: str
    narrative: str = ""


def _redis(store: Any) -> Optional[Any]:
    return store._redis if store and hasattr(store, "_redis") else None


def _get_equipment_id(ctx: PipelineContext) -> Optional[str]:
    fc = getattr(ctx, "feature_context", None)
    if fc is not None:
        sp = getattr(fc, "sensor_profile", None)
        if sp is not None:
            eq = getattr(sp, "equipment_class", None)
            if eq is not None:
                return eq.value if hasattr(eq, "value") else str(eq).lower()
    parts = ctx.series_id.split("_")
    return parts[0] if len(parts) > 1 else None


def _anomaly_severity(z: float) -> str:
    if abs(z) > 4.0:
        return "critical"
    if abs(z) > 3.0:
        return "high"
    if abs(z) > 2.5:
        return "medium"
    return "low"


# Background pool shared across all MemoryPhase instances
_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory")
    return _EXECUTOR


def _store_prediction_async(
    store: AnomalyMemoryStore, series_id: str, value: float,
    confidence: float, regime: Optional[str], fused_trend: Optional[str],
) -> None:
    """Fire-and-forget: store prediction in Weaviate."""
    try:
        event = MemoryEvent(
            sensor_id=series_id,
            sensor_type="prediction",
            timestamp=time.time(),
            event_type="prediction",
            semantic_text=(
                f"Predicción {series_id}: valor={value:.2f}, "
                f"confianza={confidence:.2f}, régimen={regime}, tendencia={fused_trend}"
            ),
            regime=regime or "unknown",
            anomaly_score=0.0,
            dynamic_features={},
            metadata={
                "value": round(value, 4),
                "confidence": round(confidence, 4),
                "trend": fused_trend,
            },
        )
        store.store(event, STORAGE_TTL_S)
    except Exception as exc:
        logger.debug("async_prediction_store_failed", extra={"series_id": series_id, "error": str(exc)})


def _store_anomaly_async(
    store: AnomalyMemoryStore, series_id: str, z_score: float,
    regime: Optional[str], equipment_id: Optional[str],
) -> None:
    """Fire-and-forget: store anomaly in Weaviate."""
    try:
        sev = _anomaly_severity(z_score)
        event = MemoryEvent(
            sensor_id=series_id,
            sensor_type="anomaly",
            timestamp=time.time(),
            event_type="anomaly",
            semantic_text=(
                f"Anomalía {series_id}: z={z_score:.2f} ({sev}), "
                f"equipo={equipment_id}, régimen={regime}"
            ),
            regime=regime or "unknown",
            anomaly_score=abs(z_score),
            dynamic_features={},
            metadata={
                "z_score": round(z_score, 4),
                "severity": sev,
                "equipment_id": equipment_id,
            },
        )
        store.store(event, STORAGE_TTL_S)
    except Exception as exc:
        logger.debug("async_anomaly_store_failed", extra={"series_id": series_id, "error": str(exc)})


def _search_similar_cases(
    store: AnomalyMemoryStore, series_id: str, equipment_id: Optional[str],
) -> List[SimilarCase]:
    """Search Weaviate for similar anomalies in the same equipment."""
    if not store._client:
        return []
    now = time.time()
    cutoff = now - SIMILAR_CASES_LOOKBACK_DAYS * 86400
    dummy_embedding = [0.0] * 1536  # Weaviate can search without a real embedding via `near_vector`
    try:
        events = store.retrieve_similar(
            query_embedding=dummy_embedding,
            sensor_type="anomaly",
            time_window=(cutoff, now),
            top_k=SIMILAR_CASES_TOP_K,
        )
        results: List[SimilarCase] = []
        for ev in events:
            ts = getattr(ev, "timestamp", 0)
            date_str = time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts)) if ts else "fecha desconocida"
            sid = getattr(ev, "sensor_id", "?") or "?"
            z = getattr(ev, "anomaly_score", 0.0) or 0.0
            reg = getattr(ev, "regime", "?") or "?"
            meta = getattr(ev, "metadata", {}) or {}
            eq = meta.get("equipment_id", "")
            if equipment_id and eq and eq != equipment_id:
                continue
            sev = _anomaly_severity(z)
            results.append(SimilarCase(
                date=date_str,
                param=sid,
                z_score=z,
                regime=reg,
                narrative=f"Anomalía similar en {sid} el {date_str} (z={z:.2f}, {sev})",
            ))
        return results
    except Exception as exc:
        logger.debug("similar_cases_search_failed", extra={"error": str(exc)})
        return []


class MemoryPhase:
    """Phase: async Weaviate memory — never blocks the pipeline."""

    def __init__(self, max_workers: int = 2) -> None:
        self._max_workers = max_workers

    @property
    def name(self) -> str:
        return "memory"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        orchestrator = ctx.orchestrator
        memory_registry = getattr(orchestrator, "_memory_registry", None)
        if memory_registry is None:
            return ctx

        store = getattr(memory_registry, "_store", None) or getattr(
            memory_registry, "anomaly_memory_store", None
        )
        if store is None:
            store = getattr(orchestrator, "_anomaly_memory_store", None)

        if store is None:
            logger.debug("memory_phase_no_store")
            return ctx

        equipment_id = _get_equipment_id(ctx)
        executor = _get_executor()

        # ── Search similar cases (sync, fast) ───────────────────
        similar = _search_similar_cases(store, ctx.series_id, equipment_id)
        if similar:
            ctx.metadata["similar_cases"] = [
                {"date": s.date, "param": s.param, "narrative": s.narrative}
                for s in similar
            ]

        # ── Store prediction (async) ────────────────────────────
        confidence = getattr(ctx, "fused_confidence", None) or 0.0
        value = getattr(ctx, "fused_value", None) or 0.0
        trend = getattr(ctx, "fused_trend", None)
        executor.submit(
            _store_prediction_async,
            store, ctx.series_id, value, confidence, ctx.regime, trend,
        )

        # ── Store anomaly (async) if present ────────────────────
        if ctx.profile:
            z = abs(getattr(ctx.profile, "z_score", 0.0))
            if z > 2.5:
                executor.submit(
                    _store_anomaly_async,
                    store, ctx.series_id, z, ctx.regime, equipment_id,
                )

        logger.debug(
            "memory_phase_completed",
            extra={
                "series_id": ctx.series_id,
                "similar_cases": len(similar),
                "async_tasks_submitted": 2 if (ctx.profile and abs(getattr(ctx.profile, "z_score", 0.0)) > 2.5) else 1,
            },
        )
        return ctx
