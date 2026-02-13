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

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...domain.entities.anomaly import AnomalyResult
from ...domain.entities.memory_search_result import MemorySearchResult
from ...domain.entities.pattern import PatternResult
from ...domain.entities.prediction import Prediction
from ...domain.ports.cognitive_memory_port import CognitiveMemoryPort

logger = logging.getLogger(__name__)

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

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """POST JSON to Weaviate.  Returns parsed response or None on error."""
        body = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            logger.warning(
                "weaviate_http_error",
                extra={"status": exc.code, "url": url, "body": error_body[:500]},
            )
            return None
        except Exception as exc:
            logger.warning(
                "weaviate_request_error",
                extra={"url": url, "error": str(exc)},
            )
            return None

    def _create_object(
        self, class_name: str, properties: Dict[str, Any]
    ) -> Optional[str]:
        """Create a Weaviate object.  Returns UUID or None."""
        if not self._enabled:
            return None

        payload = {"class": class_name, "properties": properties}

        if self._dry_run:
            logger.info(
                "weaviate_dry_run_create",
                extra={"class": class_name, "properties": properties},
            )
            return "dry-run-uuid"

        resp = self._post_json(self._objects_url, payload)
        if resp and "id" in resp:
            uuid = resp["id"]
            logger.debug(
                "weaviate_object_created",
                extra={"class": class_name, "uuid": uuid},
            )
            return uuid

        logger.warning(
            "weaviate_create_failed",
            extra={"class": class_name, "response": resp},
        )
        return None

    def _graphql_near_text(
        self,
        class_name: str,
        concepts: List[str],
        return_fields: List[str],
        *,
        where_filter: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        certainty: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Execute a nearText GraphQL query.  Returns list of result dicts."""
        if not self._enabled:
            return []

        fields_str = " ".join(return_fields)
        additional = "_additional { id certainty }"

        concepts_json = json.dumps(concepts)
        near_text = f'nearText: {{ concepts: {concepts_json}, certainty: {certainty} }}'

        where_clause = ""
        if where_filter:
            where_json = json.dumps(where_filter, default=str)
            where_clause = f", where: {where_json}"

        query = (
            "{ Get { "
            f"{class_name}({near_text}{where_clause}, limit: {limit}) "
            f"{{ {fields_str} {additional} }} "
            "} }"
        )

        if self._dry_run:
            logger.info(
                "weaviate_dry_run_query",
                extra={"class": class_name, "query": query},
            )
            return []

        resp = self._post_json(self._graphql_url, {"query": query})
        if not resp:
            return []

        try:
            results = resp["data"]["Get"][class_name]
            return results if results else []
        except (KeyError, TypeError):
            errors = resp.get("errors", [])
            if errors:
                logger.warning(
                    "weaviate_graphql_errors",
                    extra={"class": class_name, "errors": errors[:3]},
                )
            return []

    @staticmethod
    def _now_iso() -> str:
        """Current UTC timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _safe_json(obj: Any) -> str:
        """Serialize to JSON string, handling non-serializable types."""
        try:
            return json.dumps(obj, default=str)
        except Exception:
            return "{}"

    @staticmethod
    def _build_where_filter(
        operands: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Build a Weaviate where filter from a list of operand dicts.

        Skips None-valued operands.  Returns None if no operands remain.
        """
        valid = [op for op in operands if op is not None]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        return {"operator": "And", "operands": valid}

    @staticmethod
    def _where_eq_text(path: str, value: Optional[str]) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        return {
            "path": [path],
            "operator": "Equal",
            "valueText": value,
        }

    @staticmethod
    def _where_eq_int(path: str, value: Optional[int]) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        return {
            "path": [path],
            "operator": "Equal",
            "valueInt": value,
        }

    # ------------------------------------------------------------------
    # Write: remember_explanation
    # ------------------------------------------------------------------

    def remember_explanation(
        self,
        prediction: Prediction,
        source_record_id: int,
        *,
        explanation_text: str = "",
        domain_name: str = "iot",
    ) -> Optional[str]:
        try:
            text = explanation_text or str(
                prediction.metadata.get("explanation", "")
            )
            props = {
                "seriesId": prediction.series_id,
                "domainName": domain_name,
                "engineName": prediction.engine_name,
                "explanationText": text,
                "trend": prediction.trend,
                "confidenceScore": prediction.confidence_score,
                "confidenceLevel": prediction.confidence_level.value,
                "predictedValue": prediction.predicted_value,
                "horizonSteps": prediction.horizon_steps,
                "featureContributions": self._safe_json(
                    prediction.feature_contributions
                ),
                "sourceRecordId": source_record_id,
                "auditTraceId": prediction.audit_trace_id or "",
                "createdAt": self._now_iso(),
                "metadata": self._safe_json(prediction.metadata),
            }
            return self._create_object("MLExplanation", props)
        except Exception as exc:
            logger.warning(
                "remember_explanation_error",
                extra={"error": str(exc), "series_id": prediction.series_id},
            )
            return None

    # ------------------------------------------------------------------
    # Write: remember_anomaly
    # ------------------------------------------------------------------

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
        try:
            context_text = operational_context or self._safe_json(
                anomaly.context
            )
            props = {
                "seriesId": anomaly.series_id,
                "domainName": domain_name,
                "isAnomaly": anomaly.is_anomaly,
                "anomalyScore": anomaly.score,
                "confidence": anomaly.confidence,
                "severity": anomaly.severity.value,
                "explanationText": anomaly.explanation,
                "methodVotes": self._safe_json(anomaly.method_votes),
                "eventCode": event_code,
                "behaviorPattern": behavior_pattern,
                "operationalContext": context_text,
                "sourceRecordId": source_record_id,
                "relatedPredictionId": 0,
                "auditTraceId": anomaly.audit_trace_id or "",
                "createdAt": self._now_iso(),
                "metadata": "{}",
            }
            return self._create_object("AnomalyMemory", props)
        except Exception as exc:
            logger.warning(
                "remember_anomaly_error",
                extra={"error": str(exc), "series_id": anomaly.series_id},
            )
            return None

    # ------------------------------------------------------------------
    # Write: remember_pattern
    # ------------------------------------------------------------------

    def remember_pattern(
        self,
        pattern: PatternResult,
        *,
        source_record_id: Optional[int] = None,
        domain_name: str = "iot",
    ) -> Optional[str]:
        try:
            props = {
                "seriesId": pattern.series_id,
                "domainName": domain_name,
                "patternType": pattern.pattern_type.value,
                "confidence": pattern.confidence,
                "descriptionText": pattern.description,
                "changePointIndex": 0,
                "changeMagnitude": 0.0,
                "spikeClassification": "normal",
                "regimeName": "",
                "regimeMeanValue": 0.0,
                "persistenceScore": 0.0,
                "sourceRecordId": source_record_id or 0,
                "auditTraceId": "",
                "createdAt": self._now_iso(),
                "metadata": self._safe_json(pattern.metadata),
            }
            return self._create_object("PatternMemory", props)
        except Exception as exc:
            logger.warning(
                "remember_pattern_error",
                extra={"error": str(exc), "series_id": pattern.series_id},
            )
            return None

    # ------------------------------------------------------------------
    # Write: remember_decision
    # ------------------------------------------------------------------

    def remember_decision(
        self,
        decision_data: Dict[str, object],
        source_record_id: int,
        *,
        domain_name: str = "iot",
    ) -> Optional[str]:
        try:
            affected = decision_data.get("affected_series_ids", [])
            if not isinstance(affected, list):
                affected = [str(affected)]
            else:
                affected = [str(s) for s in affected]

            props = {
                "deviceId": int(decision_data.get("device_id", 0)),
                "domainName": domain_name,
                "patternSignature": str(
                    decision_data.get("pattern_signature", "")
                ),
                "decisionType": str(decision_data.get("decision_type", "")),
                "priority": str(decision_data.get("priority", "medium")),
                "severity": str(decision_data.get("severity", "info")),
                "titleText": str(decision_data.get("title", "")),
                "summaryText": str(decision_data.get("summary", "")),
                "explanationText": str(decision_data.get("explanation", "")),
                "recommendedActions": self._safe_json(
                    decision_data.get("recommended_actions", [])
                ),
                "affectedSeriesIds": affected,
                "eventCount": int(decision_data.get("event_count", 0)),
                "confidenceScore": float(
                    decision_data.get("confidence_score", 0.0)
                ),
                "isRecurring": bool(decision_data.get("is_recurring", False)),
                "historicalResolutionRate": float(
                    decision_data.get("historical_resolution_rate", 0.0)
                ),
                "reasonTrace": self._safe_json(
                    decision_data.get("reason_trace", {})
                ),
                "sourceRecordId": source_record_id,
                "auditTraceId": str(
                    decision_data.get("audit_trace_id", "")
                ),
                "createdAt": self._now_iso(),
                "metadata": self._safe_json(
                    decision_data.get("metadata", {})
                ),
            }
            return self._create_object("DecisionReasoning", props)
        except Exception as exc:
            logger.warning(
                "remember_decision_error",
                extra={"error": str(exc), "source_record_id": source_record_id},
            )
            return None

    # ------------------------------------------------------------------
    # Read: recall_similar_explanations
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
        try:
            where = self._build_where_filter([
                self._where_eq_text("seriesId", series_id),
                self._where_eq_text("engineName", engine_name),
            ])
            results = self._graphql_near_text(
                "MLExplanation",
                [query],
                ["seriesId", "explanationText", "confidenceScore",
                 "sourceRecordId", "createdAt", "engineName", "trend"],
                where_filter=where,
                limit=limit,
                certainty=min_certainty,
            )
            return [self._to_memory_result(r, "explanationText") for r in results]
        except Exception as exc:
            logger.warning(
                "recall_explanations_error",
                extra={"error": str(exc)},
            )
            return []

    # ------------------------------------------------------------------
    # Read: recall_similar_anomalies
    # ------------------------------------------------------------------

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
        try:
            where = self._build_where_filter([
                self._where_eq_text("seriesId", series_id),
                self._where_eq_text("severity", severity),
                self._where_eq_text("eventCode", event_code),
            ])
            results = self._graphql_near_text(
                "AnomalyMemory",
                [query],
                ["seriesId", "explanationText", "anomalyScore", "severity",
                 "sourceRecordId", "createdAt", "eventCode", "behaviorPattern"],
                where_filter=where,
                limit=limit,
                certainty=min_certainty,
            )
            return [self._to_memory_result(r, "explanationText") for r in results]
        except Exception as exc:
            logger.warning(
                "recall_anomalies_error",
                extra={"error": str(exc)},
            )
            return []

    # ------------------------------------------------------------------
    # Read: recall_similar_patterns
    # ------------------------------------------------------------------

    def recall_similar_patterns(
        self,
        query: str,
        *,
        series_id: Optional[str] = None,
        pattern_type: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        try:
            where = self._build_where_filter([
                self._where_eq_text("seriesId", series_id),
                self._where_eq_text("patternType", pattern_type),
            ])
            results = self._graphql_near_text(
                "PatternMemory",
                [query],
                ["seriesId", "descriptionText", "patternType", "confidence",
                 "sourceRecordId", "createdAt"],
                where_filter=where,
                limit=limit,
                certainty=min_certainty,
            )
            return [self._to_memory_result(r, "descriptionText") for r in results]
        except Exception as exc:
            logger.warning(
                "recall_patterns_error",
                extra={"error": str(exc)},
            )
            return []

    # ------------------------------------------------------------------
    # Read: recall_similar_decisions
    # ------------------------------------------------------------------

    def recall_similar_decisions(
        self,
        query: str,
        *,
        device_id: Optional[int] = None,
        severity: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        try:
            where = self._build_where_filter([
                self._where_eq_int("deviceId", device_id),
                self._where_eq_text("severity", severity),
            ])
            results = self._graphql_near_text(
                "DecisionReasoning",
                [query],
                ["summaryText", "explanationText", "severity", "decisionType",
                 "sourceRecordId", "createdAt", "affectedSeriesIds"],
                where_filter=where,
                limit=limit,
                certainty=min_certainty,
            )
            return [
                self._to_memory_result(r, "summaryText", series_key=None)
                for r in results
            ]
        except Exception as exc:
            logger.warning(
                "recall_decisions_error",
                extra={"error": str(exc)},
            )
            return []

    # ------------------------------------------------------------------
    # Result mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _to_memory_result(
        raw: Dict[str, Any],
        text_field: str,
        *,
        series_key: Optional[str] = "seriesId",
    ) -> MemorySearchResult:
        """Convert a Weaviate GraphQL result dict to MemorySearchResult."""
        additional = raw.get("_additional", {})
        source_id = raw.get("sourceRecordId")

        metadata = {
            k: v for k, v in raw.items()
            if k not in ("_additional", text_field, "seriesId",
                         "sourceRecordId", "createdAt")
            and v is not None
        }

        series_id = ""
        if series_key and series_key in raw:
            series_id = str(raw[series_key])
        elif "affectedSeriesIds" in raw:
            ids = raw["affectedSeriesIds"]
            series_id = ids[0] if ids else ""

        return MemorySearchResult(
            memory_id=additional.get("id", ""),
            series_id=series_id,
            text=raw.get(text_field, ""),
            certainty=float(additional.get("certainty", 0.0)),
            source_record_id=int(source_id) if source_id else None,
            created_at=raw.get("createdAt"),
            metadata=metadata,
        )
