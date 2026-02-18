"""Memory write operations for Weaviate cognitive memory."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ....domain.entities.anomaly import AnomalyResult
from ....domain.entities.pattern import PatternResult
from ....domain.entities.prediction import Prediction
from .filter_builders import now_iso, safe_json
from .object_operations import create_object

logger = logging.getLogger(__name__)


def remember_explanation(
    objects_url: str,
    prediction: Prediction,
    source_record_id: int,
    *,
    explanation_text: str = "",
    domain_name: str = "iot",
    enabled: bool = True,
    dry_run: bool = False,
    timeout: int = 10,
) -> Optional[str]:
    """Store a prediction explanation in cognitive memory.
    
    Args:
        objects_url: Full URL to /v1/objects endpoint
        prediction: Prediction entity with metadata
        source_record_id: Database record ID
        explanation_text: Override explanation text (uses prediction.metadata if empty)
        domain_name: Domain context
        enabled: Master switch
        dry_run: If True, logs but doesn't send
        timeout: Request timeout
        
    Returns:
        Memory UUID, or None on error
    """
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
            "featureContributions": safe_json(
                prediction.feature_contributions
            ),
            "sourceRecordId": source_record_id,
            "auditTraceId": prediction.audit_trace_id or "",
            "createdAt": now_iso(),
            "metadata": safe_json(prediction.metadata),
        }
        return create_object(
            objects_url,
            "MLExplanation",
            props,
            enabled=enabled,
            dry_run=dry_run,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning(
            "remember_explanation_error",
            extra={"error": str(exc), "series_id": prediction.series_id},
        )
        return None


def remember_anomaly(
    objects_url: str,
    anomaly: AnomalyResult,
    source_record_id: int,
    *,
    event_code: str = "ANOMALY_DETECTED",
    behavior_pattern: str = "",
    operational_context: str = "",
    domain_name: str = "iot",
    enabled: bool = True,
    dry_run: bool = False,
    timeout: int = 10,
) -> Optional[str]:
    """Store an anomaly detection result in cognitive memory.
    
    Args:
        objects_url: Full URL to /v1/objects endpoint
        anomaly: AnomalyResult entity
        source_record_id: Database record ID
        event_code: Event classification code
        behavior_pattern: Pattern description
        operational_context: Context description (uses anomaly.context if empty)
        domain_name: Domain context
        enabled: Master switch
        dry_run: If True, logs but doesn't send
        timeout: Request timeout
        
    Returns:
        Memory UUID, or None on error
    """
    try:
        context_text = operational_context or safe_json(
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
            "methodVotes": safe_json(anomaly.method_votes),
            "eventCode": event_code,
            "behaviorPattern": behavior_pattern,
            "operationalContext": context_text,
            "sourceRecordId": source_record_id,
            "relatedPredictionId": 0,
            "auditTraceId": anomaly.audit_trace_id or "",
            "createdAt": now_iso(),
            "metadata": "{}",
        }
        return create_object(
            objects_url,
            "AnomalyMemory",
            props,
            enabled=enabled,
            dry_run=dry_run,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning(
            "remember_anomaly_error",
            extra={"error": str(exc), "series_id": anomaly.series_id},
        )
        return None


def remember_pattern(
    objects_url: str,
    pattern: PatternResult,
    *,
    source_record_id: Optional[int] = None,
    domain_name: str = "iot",
    enabled: bool = True,
    dry_run: bool = False,
    timeout: int = 10,
) -> Optional[str]:
    """Store a pattern detection result in cognitive memory.
    
    Args:
        objects_url: Full URL to /v1/objects endpoint
        pattern: PatternResult entity
        source_record_id: Database record ID (optional)
        domain_name: Domain context
        enabled: Master switch
        dry_run: If True, logs but doesn't send
        timeout: Request timeout
        
    Returns:
        Memory UUID, or None on error
    """
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
            "createdAt": now_iso(),
            "metadata": safe_json(pattern.metadata),
        }
        return create_object(
            objects_url,
            "PatternMemory",
            props,
            enabled=enabled,
            dry_run=dry_run,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning(
            "remember_pattern_error",
            extra={"error": str(exc), "series_id": pattern.series_id},
        )
        return None


def remember_decision(
    objects_url: str,
    decision_data: Dict[str, object],
    source_record_id: int,
    *,
    domain_name: str = "iot",
    enabled: bool = True,
    dry_run: bool = False,
    timeout: int = 10,
) -> Optional[str]:
    """Store a decision reasoning result in cognitive memory.
    
    Args:
        objects_url: Full URL to /v1/objects endpoint
        decision_data: Decision data dict
        source_record_id: Database record ID
        domain_name: Domain context
        enabled: Master switch
        dry_run: If True, logs but doesn't send
        timeout: Request timeout
        
    Returns:
        Memory UUID, or None on error
    """
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
            "recommendedActions": safe_json(
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
            "reasonTrace": safe_json(
                decision_data.get("reason_trace", {})
            ),
            "sourceRecordId": source_record_id,
            "auditTraceId": str(
                decision_data.get("audit_trace_id", "")
            ),
            "createdAt": now_iso(),
            "metadata": safe_json(
                decision_data.get("metadata", {})
            ),
        }
        return create_object(
            objects_url,
            "DecisionReasoning",
            props,
            enabled=enabled,
            dry_run=dry_run,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning(
            "remember_decision_error",
            extra={"error": str(exc), "source_record_id": source_record_id},
        )
        return None
