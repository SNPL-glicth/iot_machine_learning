"""Weaviate implementation of CognitiveMemoryPort.

Uses raw HTTP (``urllib.request``) to communicate with Weaviate REST API.
No SDK dependency — only stdlib.  This keeps the adapter lightweight and
avoids version lock-in with the ``weaviate-client`` package.

Fail-safe behaviour:
    - Every public method catches all exceptions, logs them, and returns
      the safe default (``None`` for writes, ``[]`` for reads).
    - The ML pipeline is NEVER interrupted by a Weaviate failure.

Feature-flag aware:
    - ``ML_ENABLE_COGNITIVE_MEMORY``: master switch.
    - ``ML_COGNITIVE_MEMORY_DRY_RUN``: logs payloads without sending.
    - ``ML_COGNITIVE_MEMORY_URL``: Weaviate REST endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ....domain.entities.anomaly import AnomalyResult
from ....domain.entities.memory_search_result import MemorySearchResult
from ....domain.entities.pattern import PatternResult
from ....domain.entities.prediction import Prediction
from ....domain.ports.cognitive_memory_port import CognitiveMemoryPort
from .memory_readers import (
    recall_similar_anomalies,
    recall_similar_decisions,
    recall_similar_explanations,
    recall_similar_patterns,
)
from .memory_writers import (
    remember_anomaly,
    remember_decision,
    remember_explanation,
    remember_pattern,
)

_TIMEOUT_SECONDS = 10


class WeaviateCognitiveAdapter(CognitiveMemoryPort):
    """Weaviate-backed cognitive memory via raw HTTP.

    Args:
        base_url: Weaviate REST API base URL (e.g. ``http://localhost:8080``).
        enabled: Master switch.  If ``False``, behaves like NullCognitiveAdapter.
        dry_run: If ``True``, builds payloads and logs them but does not send.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        enabled: bool = True,
        dry_run: bool = False,
        timeout: int = _TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._enabled = enabled
        self._dry_run = dry_run
        self._timeout = timeout
        self._objects_url = f"{self._base_url}/v1/objects"
        self._graphql_url = f"{self._base_url}/v1/graphql"

    def remember_explanation(
        self,
        prediction: Prediction,
        source_record_id: int,
        *,
        explanation_text: str = "",
        domain_name: str = "iot",
    ) -> Optional[str]:
        return remember_explanation(
            self._objects_url,
            prediction,
            source_record_id,
            explanation_text=explanation_text,
            domain_name=domain_name,
            enabled=self._enabled,
            dry_run=self._dry_run,
            timeout=self._timeout,
        )

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
        return remember_anomaly(
            self._objects_url,
            anomaly,
            source_record_id,
            event_code=event_code,
            behavior_pattern=behavior_pattern,
            operational_context=operational_context,
            domain_name=domain_name,
            enabled=self._enabled,
            dry_run=self._dry_run,
            timeout=self._timeout,
        )

    def remember_pattern(
        self,
        pattern: PatternResult,
        *,
        source_record_id: Optional[int] = None,
        domain_name: str = "iot",
    ) -> Optional[str]:
        return remember_pattern(
            self._objects_url,
            pattern,
            source_record_id=source_record_id,
            domain_name=domain_name,
            enabled=self._enabled,
            dry_run=self._dry_run,
            timeout=self._timeout,
        )

    def remember_decision(
        self,
        decision_data: Dict[str, object],
        source_record_id: int,
        *,
        domain_name: str = "iot",
    ) -> Optional[str]:
        return remember_decision(
            self._objects_url,
            decision_data,
            source_record_id,
            domain_name=domain_name,
            enabled=self._enabled,
            dry_run=self._dry_run,
            timeout=self._timeout,
        )

    def recall_similar_explanations(
        self,
        query: str,
        *,
        series_id: Optional[str] = None,
        engine_name: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        return recall_similar_explanations(
            self._graphql_url,
            query,
            series_id=series_id,
            engine_name=engine_name,
            limit=limit,
            min_certainty=min_certainty,
            enabled=self._enabled,
            dry_run=self._dry_run,
            timeout=self._timeout,
        )

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
        return recall_similar_anomalies(
            self._graphql_url,
            query,
            series_id=series_id,
            severity=severity,
            event_code=event_code,
            limit=limit,
            min_certainty=min_certainty,
            enabled=self._enabled,
            dry_run=self._dry_run,
            timeout=self._timeout,
        )

    def recall_similar_patterns(
        self,
        query: str,
        *,
        series_id: Optional[str] = None,
        pattern_type: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        return recall_similar_patterns(
            self._graphql_url,
            query,
            series_id=series_id,
            pattern_type=pattern_type,
            limit=limit,
            min_certainty=min_certainty,
            enabled=self._enabled,
            dry_run=self._dry_run,
            timeout=self._timeout,
        )

    def recall_similar_decisions(
        self,
        query: str,
        *,
        device_id: Optional[int] = None,
        severity: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        return recall_similar_decisions(
            self._graphql_url,
            query,
            device_id=device_id,
            severity=severity,
            limit=limit,
            min_certainty=min_certainty,
            enabled=self._enabled,
            dry_run=self._dry_run,
            timeout=self._timeout,
        )
