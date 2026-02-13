"""Null implementation of CognitiveMemoryPort.

No-op adapter that silently discards all write operations and returns
empty results for all queries.  Used when:

    - ``ML_ENABLE_COGNITIVE_MEMORY`` is ``False`` (default).
    - Weaviate is not available or not configured.
    - Testing domain logic without a vector database.

This adapter guarantees that the ML system operates identically to
its pre-Weaviate behavior when the cognitive memory layer is disabled.

All methods are O(1) with zero side effects.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ...domain.entities.anomaly import AnomalyResult
from ...domain.entities.memory_search_result import MemorySearchResult
from ...domain.entities.pattern import PatternResult
from ...domain.entities.prediction import Prediction
from ...domain.ports.cognitive_memory_port import CognitiveMemoryPort

logger = logging.getLogger(__name__)


class NullCognitiveAdapter(CognitiveMemoryPort):
    """No-op implementation of CognitiveMemoryPort.

    All ``remember_*`` methods return ``None``.
    All ``recall_*`` methods return ``[]``.

    Logs a single debug message on first use to confirm the adapter
    is active, then operates silently.
    """

    def __init__(self) -> None:
        self._logged_init = False

    def _log_once(self) -> None:
        if not self._logged_init:
            logger.debug(
                "cognitive_memory_null_adapter_active",
                extra={"info": "All cognitive memory operations are no-ops."},
            )
            self._logged_init = True

    # ------------------------------------------------------------------
    # Write operations (all return None)
    # ------------------------------------------------------------------

    def remember_explanation(
        self,
        prediction: Prediction,
        source_record_id: int,
        *,
        explanation_text: str = "",
        domain_name: str = "iot",
    ) -> Optional[str]:
        self._log_once()
        return None

    def remember_anomaly(
        self,
        anomaly: AnomalyResult,
        source_record_id: int,
        *,
        event_code: str = "ANOMALY_DETECTED",
        behavior_pattern: str = "",
        operational_context: str = "",
        domain_name: str = "iot",
    ) -> Optional[str]:
        self._log_once()
        return None

    def remember_pattern(
        self,
        pattern: PatternResult,
        *,
        source_record_id: Optional[int] = None,
        domain_name: str = "iot",
    ) -> Optional[str]:
        self._log_once()
        return None

    def remember_decision(
        self,
        decision_data: Dict[str, object],
        source_record_id: int,
        *,
        domain_name: str = "iot",
    ) -> Optional[str]:
        self._log_once()
        return None

    # ------------------------------------------------------------------
    # Read operations (all return empty list)
    # ------------------------------------------------------------------

    def recall_similar_explanations(
        self,
        query: str,
        *,
        series_id: Optional[str] = None,
        engine_name: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        self._log_once()
        return []

    def recall_similar_anomalies(
        self,
        query: str,
        *,
        series_id: Optional[str] = None,
        severity: Optional[str] = None,
        event_code: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        self._log_once()
        return []

    def recall_similar_patterns(
        self,
        query: str,
        *,
        series_id: Optional[str] = None,
        pattern_type: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        self._log_once()
        return []

    def recall_similar_decisions(
        self,
        query: str,
        *,
        device_id: Optional[int] = None,
        severity: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        self._log_once()
        return []
