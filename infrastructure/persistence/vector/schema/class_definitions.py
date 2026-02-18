"""Weaviate class definitions for cognitive memory."""

from __future__ import annotations

from typing import Any, Dict

from .property_builder import build_property as _prop


def ml_explanation_class() -> Dict[str, Any]:
    """MLExplanation — prediction reasoning cognitive memory."""
    return {
        "class": "MLExplanation",
        "description": (
            "Cognitive memory of ML prediction explanations. "
            "Stores the 'why' behind every prediction. Enables semantic "
            "search across prediction reasoning."
        ),
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-transformers": {
                "vectorizeClassName": False,
            },
        },
        "properties": [
            _prop("seriesId", "text", "UTSAE series identifier (agnostic — not sensor_id).", skip_vectorization=True, tokenization="field"),
            _prop("domainName", "text", "Domain namespace: 'iot', 'finance', 'network', etc.", skip_vectorization=True, tokenization="field"),
            _prop("engineName", "text", "ML engine that produced the prediction: 'taylor', 'ensemble', 'baseline'.", skip_vectorization=True, tokenization="field"),
            _prop("explanationText", "text", "Natural language explanation of the prediction reasoning. THIS IS THE PRIMARY VECTORIZED FIELD."),
            _prop("trend", "text", "Predicted trend direction: 'up', 'down', 'stable'.", skip_vectorization=True, tokenization="field"),
            _prop("confidenceScore", "number", "Numeric confidence score (0.0–1.0).", skip_vectorization=True),
            _prop("confidenceLevel", "text", "Qualitative confidence: 'very_low','low','medium','high','very_high'.", skip_vectorization=True, tokenization="field"),
            _prop("predictedValue", "number", "The predicted value.", skip_vectorization=True),
            _prop("horizonSteps", "int", "Number of steps ahead predicted.", skip_vectorization=True),
            _prop("featureContributions", "text", "JSON string of feature contributions to the prediction.", skip_vectorization=True),
            _prop("sourceRecordId", "int", "Back-reference to dbo.predictions.id in SQL Server.", skip_vectorization=True),
            _prop("auditTraceId", "text", "ISO 27001 audit trace identifier.", skip_vectorization=True, tokenization="field"),
            _prop("createdAt", "date", "Timestamp when the prediction was made.", skip_vectorization=True),
            _prop("metadata", "text", "JSON string with engine-specific metadata.", skip_vectorization=True),
        ],
    }


def anomaly_memory_class() -> Dict[str, Any]:
    """AnomalyMemory — anomaly detection cognitive trace."""
    return {
        "class": "AnomalyMemory",
        "description": (
            "Cognitive memory of anomaly detections. Stores the reasoning "
            "and context of every anomaly event. Enables 'has this happened "
            "before?' queries across series and domains."
        ),
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-transformers": {
                "vectorizeClassName": False,
            },
        },
        "properties": [
            _prop("seriesId", "text", "UTSAE series identifier.", skip_vectorization=True, tokenization="field"),
            _prop("domainName", "text", "Domain namespace.", skip_vectorization=True, tokenization="field"),
            _prop("isAnomaly", "boolean", "Whether an anomaly was detected.", skip_vectorization=True),
            _prop("anomalyScore", "number", "Anomaly score (0.0 = normal, 1.0 = strong anomaly).", skip_vectorization=True),
            _prop("confidence", "number", "Detection confidence (0.0–1.0).", skip_vectorization=True),
            _prop("severity", "text", "Anomaly severity: 'none','low','medium','high','critical'.", skip_vectorization=True, tokenization="field"),
            _prop("explanationText", "text", "Natural language explanation of why this is/isn't anomalous. PRIMARY VECTORIZED FIELD."),
            _prop("methodVotes", "text", "JSON string of detector votes: {'isolation_forest': 0.8, ...}.", skip_vectorization=True),
            _prop("eventCode", "text", "Event classification code: 'ANOMALY_DETECTED', 'PREDICTION_DEVIATION'.", skip_vectorization=True, tokenization="field"),
            _prop("behaviorPattern", "text", "Detected behavior: 'stable','drifting','spike','oscillating'.", skip_vectorization=True, tokenization="field"),
            _prop("operationalContext", "text", "Regime context, correlations, environmental factors. SECONDARY VECTORIZED FIELD."),
            _prop("sourceRecordId", "int", "Back-reference to dbo.ml_events.id in SQL Server.", skip_vectorization=True),
            _prop("relatedPredictionId", "int", "Back-reference to dbo.predictions.id if linked.", skip_vectorization=True),
            _prop("auditTraceId", "text", "ISO 27001 audit trace identifier.", skip_vectorization=True, tokenization="field"),
            _prop("createdAt", "date", "Timestamp of the anomaly detection.", skip_vectorization=True),
            _prop("metadata", "text", "JSON string with detector-specific metadata.", skip_vectorization=True),
        ],
    }


def pattern_memory_class() -> Dict[str, Any]:
    """PatternMemory — behavioral pattern cognitive memory."""
    return {
        "class": "PatternMemory",
        "description": (
            "Cognitive memory of detected behavioral patterns. "
            "Stores pattern descriptions, change points, spike classifications, "
            "and operational regimes. Enables cross-series pattern discovery."
        ),
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-transformers": {
                "vectorizeClassName": False,
            },
        },
        "properties": [
            _prop("seriesId", "text", "UTSAE series identifier.", skip_vectorization=True, tokenization="field"),
            _prop("domainName", "text", "Domain namespace.", skip_vectorization=True, tokenization="field"),
            _prop("patternType", "text", "Pattern type: 'stable','drifting','oscillating','spike','micro_variation','curve_anomaly','regime_transition'.", skip_vectorization=True, tokenization="field"),
            _prop("confidence", "number", "Detection confidence (0.0–1.0).", skip_vectorization=True),
            _prop("descriptionText", "text", "Natural language description of the detected pattern. PRIMARY VECTORIZED FIELD."),
            _prop("changePointIndex", "int", "Index in the time series where the change occurred.", skip_vectorization=True),
            _prop("changeMagnitude", "number", "Magnitude of the detected change.", skip_vectorization=True),
            _prop("spikeClassification", "text", "Spike type: 'delta_spike','noise_spike','normal'.", skip_vectorization=True, tokenization="field"),
            _prop("regimeName", "text", "Operational regime: 'idle','active','peak','cooling'.", skip_vectorization=True, tokenization="field"),
            _prop("regimeMeanValue", "number", "Typical value in this operational regime.", skip_vectorization=True),
            _prop("persistenceScore", "number", "How persistent is this pattern (0–1).", skip_vectorization=True),
            _prop("sourceRecordId", "int", "Back-reference to dbo.ml_events.id if persisted in SQL.", skip_vectorization=True),
            _prop("auditTraceId", "text", "ISO 27001 audit trace identifier.", skip_vectorization=True, tokenization="field"),
            _prop("createdAt", "date", "Timestamp of the pattern detection.", skip_vectorization=True),
            _prop("metadata", "text", "JSON string with detector-specific metadata.", skip_vectorization=True),
        ],
    }


def decision_reasoning_class() -> Dict[str, Any]:
    """DecisionReasoning — decision orchestrator cognitive memory."""
    return {
        "class": "DecisionReasoning",
        "description": (
            "Cognitive memory of decision reasoning chains. Stores the "
            "'why' behind every decision the orchestrator produces. "
            "Enables 'what did we do last time?' queries."
        ),
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-transformers": {
                "vectorizeClassName": False,
            },
        },
        "properties": [
            _prop("deviceId", "int", "Device that triggered the decision.", skip_vectorization=True),
            _prop("domainName", "text", "Domain namespace.", skip_vectorization=True, tokenization="field"),
            _prop("patternSignature", "text", "Deduplication hash of the event pattern.", skip_vectorization=True, tokenization="field"),
            _prop("decisionType", "text", "Decision type: 'investigate','maintenance','alert','escalate'.", skip_vectorization=True, tokenization="field"),
            _prop("priority", "text", "Priority: 'low','medium','high','critical'.", skip_vectorization=True, tokenization="field"),
            _prop("severity", "text", "Severity: 'info','warning','critical'.", skip_vectorization=True, tokenization="field"),
            _prop("titleText", "text", "Short decision title.", skip_vectorization=True),
            _prop("summaryText", "text", "Decision summary. PRIMARY VECTORIZED FIELD."),
            _prop("explanationText", "text", "Full reasoning chain. SECONDARY VECTORIZED FIELD."),
            _prop("recommendedActions", "text", "JSON array of recommended actions.", skip_vectorization=True),
            _prop("affectedSeriesIds", "text[]", "List of UTSAE series_id strings affected by this decision.", skip_vectorization=True),
            _prop("eventCount", "int", "Number of source events that contributed to this decision.", skip_vectorization=True),
            _prop("confidenceScore", "number", "System confidence in this decision (0–1).", skip_vectorization=True),
            _prop("isRecurring", "boolean", "Whether this pattern has been seen 3+ times.", skip_vectorization=True),
            _prop("historicalResolutionRate", "number", "Past success rate for this pattern (0–1).", skip_vectorization=True),
            _prop("reasonTrace", "text", "JSON: source_event_ids, rules_applied, confidence_factors.", skip_vectorization=True),
            _prop("sourceRecordId", "int", "Back-reference to dbo.decision_actions.id in SQL Server.", skip_vectorization=True),
            _prop("auditTraceId", "text", "ISO 27001 audit trace identifier.", skip_vectorization=True, tokenization="field"),
            _prop("createdAt", "date", "Timestamp of the decision.", skip_vectorization=True),
            _prop("metadata", "text", "JSON string with additional metadata.", skip_vectorization=True),
        ],
    }
