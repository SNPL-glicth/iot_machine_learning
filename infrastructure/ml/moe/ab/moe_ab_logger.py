"""A/B Test Logger para MoE staging.

Loggea cada predicción con métricas comparativas:
- engine_used, prediction_value, confidence, latency_ms
- regime, expert_weights, selected_experts

Genera reporte agregado: MAE vs actual, latencia p50/p95,
distribución de expert selection.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger("moe.ab")


@dataclass
class ABLogEntry:
    """Entrada de log para una predicción individual."""
    timestamp: str
    engine_used: str
    prediction_value: float
    confidence: float
    latency_ms: float
    regime: str
    expert_weights: Dict[str, float]
    selected_experts: List[str]
    dominant_expert: str
    actual_value: Optional[float] = None
    cell: str = "unknown"  # "A" (control) o "B" (treatment)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cell": self.cell,
            "engine_used": self.engine_used,
            "prediction_value": self.prediction_value,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 3),
            "regime": self.regime,
            "expert_weights": self.expert_weights,
            "selected_experts": self.selected_experts,
            "dominant_expert": self.dominant_expert,
            "actual_value": self.actual_value,
            **self.metadata,
        }


class MoEABLogger:
    """Logger estructurado para A/B testing de MoE.

    Uso:
        ab_logger = MoEABLogger()
        ab_logger.log_prediction(entry)
        report = ab_logger.generate_report()
    """

    def __init__(self) -> None:
        self._entries: List[ABLogEntry] = []

    def log_prediction(self, entry: ABLogEntry) -> None:
        """Loggea una predicción individual."""
        self._entries.append(entry)
        logger.info("moe_ab_prediction", extra=entry.to_dict())

    def generate_report(self) -> Dict[str, Any]:
        """Genera reporte agregado de métricas A/B.

        Returns:
            Dict con MAE, latencia p50/p95, distribución de expertos.
        """
        if not self._entries:
            return {"status": "no_data"}

        # Agrupar por celda A/B
        cell_a = [e for e in self._entries if e.cell == "A"]
        cell_b = [e for e in self._entries if e.cell == "B"]

        def _cell_report(entries: List[ABLogEntry]) -> Dict[str, Any]:
            if not entries:
                return {"count": 0}

            latencies = [e.latency_ms for e in entries]
            sorted_lat = sorted(latencies)
            p50 = statistics.median(sorted_lat)
            p95_idx = int(len(sorted_lat) * 0.95)
            p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]

            # MAE vs actual (si hay actual_value)
            mae_entries = [e for e in entries if e.actual_value is not None]
            mae = None
            if mae_entries:
                mae = sum(
                    abs(e.prediction_value - e.actual_value)
                    for e in mae_entries
                ) / len(mae_entries)

            # Distribución de expertos dominantes
            expert_counts: Dict[str, int] = {}
            for e in entries:
                expert_counts[e.dominant_expert] = expert_counts.get(e.dominant_expert, 0) + 1
            total = len(entries)
            expert_dist = {
                k: round(v / total, 4) for k, v in expert_counts.items()
            }

            return {
                "count": len(entries),
                "latency_ms": {
                    "p50": round(p50, 3),
                    "p95": round(p95, 3),
                    "min": round(min(latencies), 3),
                    "max": round(max(latencies), 3),
                },
                "mae": round(mae, 4) if mae is not None else None,
                "expert_distribution": expert_dist,
                "avg_confidence": round(
                    sum(e.confidence for e in entries) / len(entries), 4
                ),
            }

        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_predictions": len(self._entries),
            "cell_a": _cell_report(cell_a),
            "cell_b": _cell_report(cell_b),
        }

    def clear(self) -> None:
        """Limpia entradas acumuladas."""
        self._entries.clear()
