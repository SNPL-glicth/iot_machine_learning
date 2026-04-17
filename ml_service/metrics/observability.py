"""Observability metrics for ML pipeline runtime validation.

Instruments: fallbacks, engine usage, pipeline phases, silent failures.
Restriction: < 180 lines.
"""
from __future__ import annotations
import logging
import time
from typing import Dict, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FallbackMetrics:
    """Enterprise fallback tracking."""
    total: int = 0
    per_sensor: Dict[int, int] = field(default_factory=dict)
    last_reasons: Dict[int, str] = field(default_factory=dict)
    
    def record(self, sensor_id: int, reason: str) -> None:
        self.total += 1
        self.per_sensor[sensor_id] = self.per_sensor.get(sensor_id, 0) + 1
        self.last_reasons[sensor_id] = reason
        # CRITICAL alert if rate exceeds threshold (check externally)
        logger.warning("ml.enterprise.fallback", extra={
            "sensor_id": sensor_id, "reason": reason[:100],
            "count": self.per_sensor[sensor_id]
        })


@dataclass  
class EngineUsageMetrics:
    """Engine usage distribution."""
    baseline: int = 0
    taylor: int = 0
    moe: int = 0
    fallback: int = 0
    per_sensor: Dict[int, str] = field(default_factory=dict)
    
    def record(self, engine_name: str, sensor_id: int) -> None:
        self.per_sensor[sensor_id] = engine_name
        if "baseline" in engine_name.lower():
            self.baseline += 1
            metric_name = "ml.engine.usage.baseline"
        elif "taylor" in engine_name.lower():
            self.taylor += 1
            metric_name = "ml.engine.usage.taylor"
        elif "moe" in engine_name.lower():
            self.moe += 1
            metric_name = "ml.engine.usage.moe"
        else:
            self.fallback += 1
            metric_name = "ml.engine.usage.other"
        logger.info(metric_name, extra={"sensor_id": sensor_id, "engine": engine_name})


@dataclass
class SemanticMetrics:
    """Semantic enrichment tracking."""
    executed: int = 0
    skipped: int = 0
    entities_total: int = 0
    critical_detected: int = 0
    errors: int = 0
    
    def record_execution(self, entity_count: int, has_critical: bool) -> None:
        self.executed += 1
        self.entities_total += entity_count
        if has_critical:
            self.critical_detected += 1
        logger.info("ml.semantic.enrichment.executed", extra={
            "entity_count": entity_count, "critical": has_critical
        })
    
    def record_skip(self, reason: str) -> None:
        self.skipped += 1
        logger.debug("ml.semantic.enrichment.skipped", extra={"reason": reason})
    
    def record_error(self, error: str) -> None:
        self.errors += 1
        logger.error("ml.semantic.enrichment.error", extra={"error": error[:100]})


@dataclass
class SilentFailureMetrics:
    """Tracks silently caught exceptions."""
    errors_by_location: Dict[str, int] = field(default_factory=dict)
    
    def record(self, location: str, error: str, context: Optional[Dict] = None) -> None:
        self.errors_by_location[location] = self.errors_by_location.get(location, 0) + 1
        extra = {"location": location, "error": error[:200], "count": self.errors_by_location[location]}
        if context:
            extra.update(context)
        logger.error("ml.silent.failure.detected", extra=extra)


class ObservabilityCollector:
    """Singleton collector for all observability metrics."""
    _instance: Optional[ObservabilityCollector] = None
    
    def __new__(cls) -> ObservabilityCollector:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.fallback = FallbackMetrics()
            cls._instance.engine_usage = EngineUsageMetrics()
            cls._instance.semantic = SemanticMetrics()
            cls._instance.silent_failures = SilentFailureMetrics()
            cls._instance._start_time = time.monotonic()
        return cls._instance
    
    def get_fallback_rate(self, total_predictions: int) -> float:
        if total_predictions == 0:
            return 0.0
        return self.fallback.total / total_predictions
    
    def to_dict(self) -> Dict:
        uptime = time.monotonic() - self._start_time
        return {
            "fallback": {"total": self.fallback.total, "per_sensor_count": len(self.fallback.per_sensor)},
            "engine_usage": {
                "baseline": self.engine_usage.baseline,
                "taylor": self.engine_usage.taylor,
                "moe": self.engine_usage.moe,
                "fallback": self.engine_usage.fallback,
            },
            "semantic": {
                "executed": self.semantic.executed,
                "skipped": self.semantic.skipped,
                "entities_total": self.semantic.entities_total,
                "critical_detected": self.semantic.critical_detected,
                "errors": self.semantic.errors,
            },
            "silent_failures": self.silent_failures.errors_by_location,
            "uptime_seconds": int(uptime),
        }


def get_observability() -> ObservabilityCollector:
    return ObservabilityCollector()
