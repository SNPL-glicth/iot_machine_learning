"""Training data extractor — DB query and row parsing.

Reads from zenin_docs using ZeninDbConnection (production singleton).
Does NOT modify any production table.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
    ZeninDbConnection,
)

logger = logging.getLogger(__name__)

_QUERY = text("""
    SELECT
        iq.Content          AS full_text,
        ar.MlResult         AS ml_result_json,
        ar.Conclusion       AS conclusion,
        ar.Id               AS analysis_id
    FROM zenin_docs.ingestion_queue iq
    JOIN zenin_docs.analysis_results ar
        ON iq.AnalysisResultId = ar.Id
    WHERE ar.Status    = 'analyzed'
      AND ar.IsDeleted = 0
""")


def _parse_ml_result(ml_result_json: Optional[str]) -> Dict[str, Any]:
    """Parse MlResult JSON blob. Returns empty dict on failure."""
    if not ml_result_json:
        return {}
    try:
        return json.loads(ml_result_json)
    except (json.JSONDecodeError, TypeError):
        return {}


def _extract_fields(ml_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training fields from the parsed MlResult blob."""
    analysis = ml_result.get("analysis", {})
    return {
        "situation_vector": analysis.get("situation_vector", []),
        "domain": analysis.get("domain", "general"),
        "severity": (
            analysis
            .get("adaptive_thresholds", {})
            .get("current_severity", "info")
        ),
        "urgency_score": float(analysis.get("urgency_score", 0.0)),
        "confidence": float(ml_result.get("confidence", 0.0)),
    }


def _is_valid_vector(situation_vector: Any) -> bool:
    """Return True only if vector is a list of exactly 18 floats."""
    return (
        isinstance(situation_vector, list)
        and len(situation_vector) == 18
    )


def fetch_rows() -> List[Dict[str, Any]]:
    """Execute the extraction query and return raw rows as dicts."""
    with ZeninDbConnection.get_connection() as conn:
        result = conn.execute(_QUERY)
        rows = result.fetchall()
    logger.info("[EXTRACTOR] Fetched %d rows from DB", len(rows))
    return [dict(row._mapping) for row in rows]


def build_training_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse rows and return valid training records.

    Filters out records where situation_vector is missing or < 18 dims.
    """
    records: List[Dict[str, Any]] = []
    skipped = 0

    for row in rows:
        ml_result = _parse_ml_result(row.get("ml_result_json"))
        fields = _extract_fields(ml_result)

        if not _is_valid_vector(fields["situation_vector"]):
            skipped += 1
            continue

        records.append({
            "analysis_id": str(row["analysis_id"]),
            "situation_vector": fields["situation_vector"],
            "domain": fields["domain"],
            "severity": fields["severity"],
            "urgency_score": fields["urgency_score"],
            "confidence": fields["confidence"],
            "conclusion": row.get("conclusion") or "",
        })

    logger.info(
        "[EXTRACTOR] Valid records: %d | Skipped (no vector): %d",
        len(records), skipped,
    )
    return records


def compute_summary(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Return domain and severity distribution counts."""
    domain_dist: Dict[str, int] = {}
    severity_dist: Dict[str, int] = {}

    for r in records:
        domain_dist[r["domain"]] = domain_dist.get(r["domain"], 0) + 1
        severity_dist[r["severity"]] = severity_dist.get(r["severity"], 0) + 1

    return {"by_domain": domain_dist, "by_severity": severity_dist}
