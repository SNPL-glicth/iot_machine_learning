"""Result persistence: analysis_result upsert + Weaviate save.

Single responsibility: take a completed analysis and persist it to
both SQL Server (analysis_results) and optionally Weaviate.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from . import queue_repository as repo

logger = logging.getLogger(__name__)

# Graceful import - semantic namer optional
_SEMANTIC_NAMER_AVAILABLE = False
try:
    from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.semantic_namer import generate_semantic_name
    _SEMANTIC_NAMER_AVAILABLE = True
    logger.info("semantic_namer_available")
except Exception as e:
    logger.warning(f"semantic_namer_unavailable: {e}")


def build_ml_result_json(
    analysis: Dict[str, Any],
    ml_doc_id: Optional[str],
) -> str:
    """Serialize analysis into the ML result JSON stored in DB."""
    return json.dumps({
        "analysis": analysis.get("analysis", {}),
        "adaptive_thresholds": analysis.get("adaptive_thresholds", {}),
        "confidence": analysis.get("confidence", 0.0),
        "processing_time_ms": analysis.get("processing_time_ms", 0),
        "weaviate_doc_id": ml_doc_id,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "decision_recommendation": analysis.get("decision_recommendation"),
    })


def write_result(
    conn,
    *,
    queue_id,
    tenant_id,
    user_id,
    filename: str,
    file_extension: str,
    content_type: str,
    metadata: Dict[str, Any],
    analysis: Dict[str, Any],
    ml_doc_id: Optional[str],
) -> Optional[str]:
    """Write analysis result to DB and mark queue item completed.

    Handles both UPDATE (row created by .NET) and INSERT (edge case).
    Returns the analysis_result_id.
    """
    ml_result_json = build_ml_result_json(analysis, ml_doc_id)
    conclusion = analysis.get("conclusion", "Análisis completado.")

    # Check if analysis_results row already exists
    analysis_result_id = metadata.get("analysis_result_id")
    if analysis_result_id:
        logger.info(
            "[RESULT_WRITER] Found analysis_result_id=%s from metadata",
            analysis_result_id,
        )
    else:
        analysis_result_id = repo.find_analysis_result(
            conn, tenant_id, user_id, filename
        )
        logger.info(
            "[RESULT_WRITER] Looked up analysis_result_id=%s from DB",
            analysis_result_id,
        )

    if analysis_result_id:
        # Generate semantic name if available
        semantic_name = None
        if _SEMANTIC_NAMER_AVAILABLE:
            try:
                domain = analysis.get("domain", "general")
                analyzed_at = datetime.now(timezone.utc)
                
                # Extract Monte Carlo confidence if available
                monte_carlo_conf = None
                monte_carlo = analysis.get("monte_carlo")
                if monte_carlo and isinstance(monte_carlo, dict):
                    monte_carlo_conf = monte_carlo.get("confidence_score")
                
                semantic_name = generate_semantic_name(
                    conclusion, domain, analyzed_at, monte_carlo_conf
                )
                logger.info(
                    f"[RESULT_WRITER] Generated semantic_name: {semantic_name}",
                    extra={"queue_id": queue_id},
                )
            except Exception as e:
                logger.warning(
                    f"semantic_name_generation_failed: {e}",
                    extra={"queue_id": queue_id},
                )
        repo.update_analysis_result(
            conn,
            analysis_id=analysis_result_id,
            classification=content_type,
            ml_result_json=ml_result_json,
            conclusion=conclusion,
            ml_doc_id=ml_doc_id,
        )
    else:
        # Generate semantic name if available
        semantic_name = None
        if _SEMANTIC_NAMER_AVAILABLE:
            try:
                domain = analysis.get("domain", "general")
                analyzed_at = datetime.now(timezone.utc)
                
                # Extract Monte Carlo confidence if available
                monte_carlo_conf = None
                monte_carlo = analysis.get("monte_carlo")
                if monte_carlo and isinstance(monte_carlo, dict):
                    monte_carlo_conf = monte_carlo.get("confidence_score")
                
                semantic_name = generate_semantic_name(
                    conclusion, domain, analyzed_at, monte_carlo_conf
                )
                logger.info(
                    f"[RESULT_WRITER] Generated semantic_name: {semantic_name}",
                    extra={"queue_id": queue_id},
                )
            except Exception as e:
                logger.warning(
                    f"semantic_name_generation_failed: {e}",
                    extra={"queue_id": queue_id},
                )
        analysis_result_id = repo.insert_analysis_result(
            conn,
            tenant_id=tenant_id,
            user_id=user_id,
            filename=filename,
            extension=file_extension,
            file_size=metadata.get("file_size_bytes", 0),
            classification=content_type,
            ml_result_json=ml_result_json,
            conclusion=conclusion,
            semantic_name=semantic_name,
            ml_doc_id=ml_doc_id,
        )

    # Mark queue item completed
    repo.mark_completed(conn, queue_id, analysis_result_id)

    logger.info(
        "[RESULT_WRITER] Completed %s -> analysis_id=%s",
        queue_id, analysis_result_id,
    )
    return analysis_result_id


def save_to_weaviate(
    weaviate_url: Optional[str],
    *,
    tenant_id: str,
    filename: str,
    content: str,
    analysis: Dict[str, Any],
) -> Optional[str]:
    """Save document embedding to Weaviate via raw HTTP.

    Fire-and-forget: failures are logged but never propagated.
    Returns the Weaviate doc_id or None.
    """
    if not weaviate_url:
        return None

    try:
        import urllib.request
        import uuid as _uuid

        inner = analysis.get("analysis", {})
        doc_id = str(_uuid.uuid4())

        payload = json.dumps({
            "class": "MLExplanation",
            "id": doc_id,
            "properties": {
                "domainName": "zenin_docs",
                "seriesId": filename,
                "engineName": "document_analyzer",
                "explanationText": content[:2000],
                "trend": inner.get("sentiment", "neutral"),
                "confidenceScore": analysis.get("confidence", 0.0),
                "confidenceLevel": "medium",
                "predictedValue": inner.get("urgency_score", 0.0),
                "horizonSteps": 0,
                "featureContributions": json.dumps(
                    inner.get("triggers_activated", [])
                ),
                "auditTraceId": analysis.get("document_id", ""),
                "metadata": json.dumps({
                    "tenant_id": tenant_id,
                    "filename": filename,
                    "sentiment": inner.get("sentiment", "neutral"),
                    "urgency_score": inner.get("urgency_score", 0.0),
                }),
            },
        }).encode("utf-8")

        url = f"{weaviate_url}/v1/objects"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 201):
                logger.info(
                    "[RESULT_WRITER] Saved to Weaviate: %s (%s)",
                    filename, doc_id,
                )
                return doc_id

        return None
    except Exception as e:
        logger.warning("[RESULT_WRITER] Weaviate save failed: %s", e)
        return None


def resolve_weaviate_url() -> Optional[str]:
    """Return Weaviate base URL if enabled, else None."""
    enabled = os.environ.get("WEAVIATE_ENABLED", "false").lower() == "true"
    if not enabled:
        logger.info(
            "[RESULT_WRITER] Weaviate disabled — documents won't be indexed"
        )
        return None
    url = os.environ.get("WEAVIATE_URL", "http://localhost:8080").rstrip("/")
    logger.info("[RESULT_WRITER] Weaviate enabled: %s", url)
    return url
