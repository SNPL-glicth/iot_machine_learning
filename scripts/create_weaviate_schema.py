"""Weaviate Schema Creation Script — Cognitive Memory Layer.

Creates the 4 Weaviate classes for the UTSAE Cognitive Memory:
  - MLExplanation:      Prediction reasoning (vectorized explanations)
  - AnomalyMemory:      Anomaly detection traces (vectorized explanations + context)
  - PatternMemory:       Behavioral pattern descriptions (vectorized descriptions)
  - DecisionReasoning:  Decision orchestrator reasoning chains (vectorized summaries)

Usage:
    # Ensure Weaviate is running (docker-compose.cognitive.yml)
    python -m iot_machine_learning.scripts.create_weaviate_schema

    # Custom URL:
    python -m iot_machine_learning.scripts.create_weaviate_schema --url http://localhost:8080

    # Dry-run (print schema JSON without applying):
    python -m iot_machine_learning.scripts.create_weaviate_schema --dry-run

    # Delete existing classes first (for development reset):
    python -m iot_machine_learning.scripts.create_weaviate_schema --recreate

Requirements:
    pip install weaviate-client>=4.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

WEAVIATE_DEFAULT_URL = "http://localhost:8080"

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

def _prop(
    name: str,
    data_type: str,
    description: str,
    *,
    skip_vectorization: bool = False,
    tokenization: str | None = None,
) -> Dict[str, Any]:
    """Build a Weaviate property dict.

    Args:
        name: Property name (camelCase).
        data_type: Weaviate data type (``"text"``, ``"number"``, ``"int"``,
            ``"boolean"``, ``"date"``, ``"text[]"``).
        description: Human-readable description.
        skip_vectorization: If ``True``, this property is excluded from
            the vectorization input.  Use for IDs, scores, and structured
            fields that should not influence semantic similarity.
        tokenization: Weaviate tokenization strategy for text fields.
            ``"field"`` = exact match (for IDs, enums).
            ``"word"`` = standard tokenization (default for text).
            ``None`` = use Weaviate default.
    """
    prop: Dict[str, Any] = {
        "name": name,
        "dataType": [data_type],
        "description": description,
    }

    module_config: Dict[str, Any] = {}

    if skip_vectorization:
        module_config["text2vec-transformers"] = {
            "skip": True,
            "vectorizePropertyName": False,
        }

    if module_config:
        prop["moduleConfig"] = module_config

    if tokenization is not None and data_type == "text":
        prop["tokenization"] = tokenization

    return prop


def _ml_explanation_class() -> Dict[str, Any]:
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
            _prop(
                "seriesId", "text",
                "UTSAE series identifier (agnostic — not sensor_id).",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "domainName", "text",
                "Domain namespace: 'iot', 'finance', 'network', etc.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "engineName", "text",
                "ML engine that produced the prediction: 'taylor', 'ensemble', 'baseline'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "explanationText", "text",
                "Natural language explanation of the prediction reasoning. "
                "THIS IS THE PRIMARY VECTORIZED FIELD.",
            ),
            _prop(
                "trend", "text",
                "Predicted trend direction: 'up', 'down', 'stable'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "confidenceScore", "number",
                "Numeric confidence score (0.0–1.0).",
                skip_vectorization=True,
            ),
            _prop(
                "confidenceLevel", "text",
                "Qualitative confidence: 'very_low','low','medium','high','very_high'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "predictedValue", "number",
                "The predicted value.",
                skip_vectorization=True,
            ),
            _prop(
                "horizonSteps", "int",
                "Number of steps ahead predicted.",
                skip_vectorization=True,
            ),
            _prop(
                "featureContributions", "text",
                "JSON string of feature contributions to the prediction.",
                skip_vectorization=True,
            ),
            _prop(
                "sourceRecordId", "int",
                "Back-reference to dbo.predictions.id in SQL Server.",
                skip_vectorization=True,
            ),
            _prop(
                "auditTraceId", "text",
                "ISO 27001 audit trace identifier.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "createdAt", "date",
                "Timestamp when the prediction was made.",
                skip_vectorization=True,
            ),
            _prop(
                "metadata", "text",
                "JSON string with engine-specific metadata.",
                skip_vectorization=True,
            ),
        ],
    }


def _anomaly_memory_class() -> Dict[str, Any]:
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
            _prop(
                "seriesId", "text",
                "UTSAE series identifier.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "domainName", "text",
                "Domain namespace.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "isAnomaly", "boolean",
                "Whether an anomaly was detected.",
                skip_vectorization=True,
            ),
            _prop(
                "anomalyScore", "number",
                "Anomaly score (0.0 = normal, 1.0 = strong anomaly).",
                skip_vectorization=True,
            ),
            _prop(
                "confidence", "number",
                "Detection confidence (0.0–1.0).",
                skip_vectorization=True,
            ),
            _prop(
                "severity", "text",
                "Anomaly severity: 'none','low','medium','high','critical'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "explanationText", "text",
                "Natural language explanation of why this is/isn't anomalous. "
                "PRIMARY VECTORIZED FIELD.",
            ),
            _prop(
                "methodVotes", "text",
                "JSON string of detector votes: {'isolation_forest': 0.8, ...}.",
                skip_vectorization=True,
            ),
            _prop(
                "eventCode", "text",
                "Event classification code: 'ANOMALY_DETECTED', 'PREDICTION_DEVIATION'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "behaviorPattern", "text",
                "Detected behavior: 'stable','drifting','spike','oscillating'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "operationalContext", "text",
                "Regime context, correlations, environmental factors. "
                "SECONDARY VECTORIZED FIELD.",
            ),
            _prop(
                "sourceRecordId", "int",
                "Back-reference to dbo.ml_events.id in SQL Server.",
                skip_vectorization=True,
            ),
            _prop(
                "relatedPredictionId", "int",
                "Back-reference to dbo.predictions.id if linked.",
                skip_vectorization=True,
            ),
            _prop(
                "auditTraceId", "text",
                "ISO 27001 audit trace identifier.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "createdAt", "date",
                "Timestamp of the anomaly detection.",
                skip_vectorization=True,
            ),
            _prop(
                "metadata", "text",
                "JSON string with detector-specific metadata.",
                skip_vectorization=True,
            ),
        ],
    }


def _pattern_memory_class() -> Dict[str, Any]:
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
            _prop(
                "seriesId", "text",
                "UTSAE series identifier.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "domainName", "text",
                "Domain namespace.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "patternType", "text",
                "Pattern type: 'stable','drifting','oscillating','spike',"
                "'micro_variation','curve_anomaly','regime_transition'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "confidence", "number",
                "Detection confidence (0.0–1.0).",
                skip_vectorization=True,
            ),
            _prop(
                "descriptionText", "text",
                "Natural language description of the detected pattern. "
                "PRIMARY VECTORIZED FIELD.",
            ),
            _prop(
                "changePointIndex", "int",
                "Index in the time series where the change occurred.",
                skip_vectorization=True,
            ),
            _prop(
                "changeMagnitude", "number",
                "Magnitude of the detected change.",
                skip_vectorization=True,
            ),
            _prop(
                "spikeClassification", "text",
                "Spike type: 'delta_spike','noise_spike','normal'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "regimeName", "text",
                "Operational regime: 'idle','active','peak','cooling'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "regimeMeanValue", "number",
                "Typical value in this operational regime.",
                skip_vectorization=True,
            ),
            _prop(
                "persistenceScore", "number",
                "How persistent is this pattern (0–1).",
                skip_vectorization=True,
            ),
            _prop(
                "sourceRecordId", "int",
                "Back-reference to dbo.ml_events.id if persisted in SQL.",
                skip_vectorization=True,
            ),
            _prop(
                "auditTraceId", "text",
                "ISO 27001 audit trace identifier.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "createdAt", "date",
                "Timestamp of the pattern detection.",
                skip_vectorization=True,
            ),
            _prop(
                "metadata", "text",
                "JSON string with detector-specific metadata.",
                skip_vectorization=True,
            ),
        ],
    }


def _decision_reasoning_class() -> Dict[str, Any]:
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
            _prop(
                "deviceId", "int",
                "Device that triggered the decision.",
                skip_vectorization=True,
            ),
            _prop(
                "domainName", "text",
                "Domain namespace.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "patternSignature", "text",
                "Deduplication hash of the event pattern.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "decisionType", "text",
                "Decision type: 'investigate','maintenance','alert','escalate'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "priority", "text",
                "Priority: 'low','medium','high','critical'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "severity", "text",
                "Severity: 'info','warning','critical'.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "titleText", "text",
                "Short decision title.",
                skip_vectorization=True,
            ),
            _prop(
                "summaryText", "text",
                "Decision summary. PRIMARY VECTORIZED FIELD.",
            ),
            _prop(
                "explanationText", "text",
                "Full reasoning chain. SECONDARY VECTORIZED FIELD.",
            ),
            _prop(
                "recommendedActions", "text",
                "JSON array of recommended actions.",
                skip_vectorization=True,
            ),
            _prop(
                "affectedSeriesIds", "text[]",
                "List of UTSAE series_id strings affected by this decision.",
                skip_vectorization=True,
            ),
            _prop(
                "eventCount", "int",
                "Number of source events that contributed to this decision.",
                skip_vectorization=True,
            ),
            _prop(
                "confidenceScore", "number",
                "System confidence in this decision (0–1).",
                skip_vectorization=True,
            ),
            _prop(
                "isRecurring", "boolean",
                "Whether this pattern has been seen 3+ times.",
                skip_vectorization=True,
            ),
            _prop(
                "historicalResolutionRate", "number",
                "Past success rate for this pattern (0–1).",
                skip_vectorization=True,
            ),
            _prop(
                "reasonTrace", "text",
                "JSON: source_event_ids, rules_applied, confidence_factors.",
                skip_vectorization=True,
            ),
            _prop(
                "sourceRecordId", "int",
                "Back-reference to dbo.decision_actions.id in SQL Server.",
                skip_vectorization=True,
            ),
            _prop(
                "auditTraceId", "text",
                "ISO 27001 audit trace identifier.",
                skip_vectorization=True,
                tokenization="field",
            ),
            _prop(
                "createdAt", "date",
                "Timestamp of the decision.",
                skip_vectorization=True,
            ),
            _prop(
                "metadata", "text",
                "JSON string with additional metadata.",
                skip_vectorization=True,
            ),
        ],
    }


def get_all_classes() -> List[Dict[str, Any]]:
    """Return all 4 cognitive memory class definitions."""
    return [
        _ml_explanation_class(),
        _anomaly_memory_class(),
        _pattern_memory_class(),
        _decision_reasoning_class(),
    ]


# ---------------------------------------------------------------------------
# Schema operations
# ---------------------------------------------------------------------------

def create_schema(
    weaviate_url: str,
    *,
    dry_run: bool = False,
    recreate: bool = False,
) -> bool:
    """Create the cognitive memory schema in Weaviate.

    Args:
        weaviate_url: Weaviate REST API URL.
        dry_run: If ``True``, print the schema JSON and exit.
        recreate: If ``True``, delete existing classes before creating.

    Returns:
        ``True`` if all classes were created successfully.
    """
    classes = get_all_classes()

    if dry_run:
        schema = {"classes": classes}
        print(json.dumps(schema, indent=2, ensure_ascii=False))
        logger.info("Dry-run: printed schema JSON for %d classes.", len(classes))
        return True

    try:
        import weaviate
    except ImportError:
        logger.error(
            "weaviate-client not installed. "
            "Run: pip install weaviate-client>=4.0.0"
        )
        return False

    logger.info("Connecting to Weaviate at %s ...", weaviate_url)

    try:
        client = weaviate.connect_to_local(
            host=weaviate_url.replace("http://", "").split(":")[0],
            port=int(weaviate_url.split(":")[-1]),
        )
    except Exception:
        logger.exception("Failed to connect to Weaviate at %s", weaviate_url)
        return False

    try:
        if not client.is_ready():
            logger.error("Weaviate is not ready at %s", weaviate_url)
            return False

        logger.info("Weaviate is ready. Creating schema...")

        existing = {c.name for c in client.collections.list_all().values()}

        for class_def in classes:
            class_name = class_def["class"]

            if class_name in existing:
                if recreate:
                    logger.warning(
                        "Deleting existing class '%s' (--recreate).", class_name
                    )
                    client.collections.delete(class_name)
                else:
                    logger.info(
                        "Class '%s' already exists. Skipping.", class_name
                    )
                    continue

            _create_class_v4(client, class_def)
            logger.info("Created class '%s' (%d properties).",
                        class_name, len(class_def["properties"]))

        logger.info("Schema creation complete.")
        return True

    finally:
        client.close()


def _create_class_v4(client: Any, class_def: Dict[str, Any]) -> None:
    """Create a single class using the weaviate v4 client API.

    Falls back to the raw REST endpoint if the high-level API
    does not support the full property config.
    """
    import weaviate.classes.config as wvc

    # Map data type strings to weaviate v4 DataType enums
    _type_map = {
        "text": wvc.DataType.TEXT,
        "text[]": wvc.DataType.TEXT_ARRAY,
        "int": wvc.DataType.INT,
        "number": wvc.DataType.NUMBER,
        "boolean": wvc.DataType.BOOL,
        "date": wvc.DataType.DATE,
    }

    properties = []
    for p in class_def["properties"]:
        dt_str = p["dataType"][0]
        dt = _type_map.get(dt_str)
        if dt is None:
            raise ValueError(f"Unknown data type: {dt_str}")

        skip = False
        vectorize_name = True
        mc = p.get("moduleConfig", {}).get("text2vec-transformers", {})
        if mc:
            skip = mc.get("skip", False)
            vectorize_name = mc.get("vectorizePropertyName", True)

        tokenization_val = None
        tok_str = p.get("tokenization")
        if tok_str == "field":
            tokenization_val = wvc.Tokenization.FIELD
        elif tok_str == "word":
            tokenization_val = wvc.Tokenization.WORD

        prop_kwargs: Dict[str, Any] = {
            "name": p["name"],
            "data_type": dt,
            "description": p.get("description", ""),
            "skip_vectorization": skip,
            "vectorize_property_name": vectorize_name,
        }
        if tokenization_val is not None:
            prop_kwargs["tokenization"] = tokenization_val

        properties.append(wvc.Property(**prop_kwargs))

    client.collections.create(
        name=class_def["class"],
        description=class_def.get("description", ""),
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(
            vectorize_collection_name=False,
        ),
        properties=properties,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Weaviate schema for UTSAE Cognitive Memory Layer."
    )
    parser.add_argument(
        "--url",
        default=WEAVIATE_DEFAULT_URL,
        help=f"Weaviate REST API URL (default: {WEAVIATE_DEFAULT_URL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schema JSON without applying to Weaviate.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing classes before creating (development only).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    success = create_schema(
        args.url,
        dry_run=args.dry_run,
        recreate=args.recreate,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
