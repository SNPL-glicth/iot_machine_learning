#!/usr/bin/env python3
"""
Sube los resúmenes semánticos de weaviate_ready_output.json a Weaviate.

Asigna cada tipo de resumen a la clase cognitiva correspondiente:
  - equipment_profile  → MLExplanation
  - parameter_profile  → MLExplanation
  - anomaly            → AnomalyMemory
  - temporal_pattern   → PatternMemory

Uso:
  python upload_to_weaviate.py
  python upload_to_weaviate.py --dry-run     # solo logs, no envía
  python upload_to_weaviate.py --url http://localhost:8080
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PARENT = os.path.dirname(_PROJECT_ROOT)
for _p in (_PROJECT_ROOT, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("upload_weaviate")

from infrastructure.adapters.weaviate.batch_operations import WeaviateBatch


def _rfc3339(dt: datetime | None = None) -> str:
    """RFC3339 con Z para Weaviate."""
    dt = dt or datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts_rfc3339(ts_str: str) -> str:
    """Convierte timestamp string 'YYYY-MM-DD HH:MM:SS' a RFC3339."""
    try:
        dt = datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return _rfc3339()


def _make_explanation(summary: dict) -> dict:
    """Convierte un profile (equipo o parámetro) a propiedades MLExplanation."""
    param = summary.get("parameter", "")
    equip = summary.get("equipment_id", "unknown")
    series_id = f"{equip}/{param}" if param else equip
    engine = "equipment_profile" if summary["type"] == "equipment_profile" else "parameter_profile"
    return {
        "seriesId": series_id,
        "domainName": "iot",
        "engineName": engine,
        "explanationText": summary["summary"],
        "trend": "stable",
        "confidenceScore": 1.0,
        "confidenceLevel": "high",
        "predictedValue": summary.get("mean", 0.0),
        "horizonSteps": 0,
        "featureContributions": json.dumps(summary, ensure_ascii=False, default=str),
        "sourceRecordId": 0,
        "auditTraceId": f"alpla-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "createdAt": _rfc3339(),
        "metadata": json.dumps({"source": "ALPLA_dataset", "type": summary["type"]}, ensure_ascii=False),
    }


def _make_anomaly(summary: dict) -> dict:
    """Convierte una anomalía a propiedades AnomalyMemory."""
    equip = summary.get("equipment_id", "unknown")
    ts = summary.get("timestamp", "")
    params = summary.get("anomalous_parameters", [])
    is_counter = summary.get("counter_extreme", False)
    severity = "high" if is_counter else "medium"
    event_code = "COUNTER_EXTREME" if is_counter else "IF_ANOMALY"
    behavior = "spike" if is_counter else "deviation"
    return {
        "seriesId": f"{equip}/{','.join(params)}",
        "domainName": "iot",
        "isAnomaly": True,
        "anomalyScore": 1.0 if is_counter else 0.8,
        "confidence": 0.95 if is_counter else 0.85,
        "severity": severity,
        "explanationText": summary["summary"],
        "methodVotes": json.dumps({"detection": event_code}, ensure_ascii=False),
        "eventCode": event_code,
        "behaviorPattern": behavior,
        "operationalContext": f"Timestamp: {ts}, Equipment: {equip}, Parameters: {', '.join(params)}",
        "sourceRecordId": 0,
        "relatedPredictionId": 0,
        "auditTraceId": f"alpla-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "createdAt": _parse_ts_rfc3339(ts),
        "metadata": json.dumps({"source": "ALPLA_dataset", "type": "anomaly"}, ensure_ascii=False),
    }


def _make_pattern(summary: dict) -> dict:
    """Convierte un patrón temporal a propiedades PatternMemory."""
    equip = summary.get("equipment_id", "unknown")
    param = summary.get("parameter", "unknown")
    trend = summary.get("trend", "stable")
    trend_pct = summary.get("trend_percentage", 0.0)
    return {
        "seriesId": f"{equip}/{param}",
        "domainName": "iot",
        "patternType": "drifting",
        "confidence": min(trend_pct / 100.0 + 0.5, 1.0),
        "descriptionText": summary["summary"],
        "changePointIndex": 0,
        "changeMagnitude": trend_pct,
        "spikeClassification": "normal",
        "regimeName": trend,
        "regimeMeanValue": 0.0,
        "persistenceScore": 0.8,
        "sourceRecordId": 0,
        "auditTraceId": f"alpla-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "createdAt": _rfc3339(),
        "metadata": json.dumps({"source": "ALPLA_dataset", "type": "temporal_pattern"}, ensure_ascii=False),
    }


_TYPE_MAP = {
    "equipment_profile": ("MLExplanation", _make_explanation),
    "parameter_profile": ("MLExplanation", _make_explanation),
    "anomaly": ("AnomalyMemory", _make_anomaly),
    "temporal_pattern": ("PatternMemory", _make_pattern),
}


def main():
    parser = argparse.ArgumentParser(description="Sube resúmenes semánticos a Weaviate")
    parser.add_argument("--url", default="http://localhost:8080", help="Weaviate base URL")
    parser.add_argument("--dry-run", action="store_true", help="Solo logs, no envía")
    parser.add_argument("--batch-size", type=int, default=50, help="Tamaño de batch")
    parser.add_argument("input", nargs="?", default=os.path.join(_SCRIPT_DIR, "weaviate_ready_output.json"),
                        help="Archivo JSON de entrada")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        logger.error("Archivo no encontrado: %s", input_path)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    summaries = data.get("summaries", [])
    if not summaries:
        logger.error("No hay resúmenes en el archivo")
        sys.exit(1)

    print("=" * 80)
    print("  SUBIENDO RESÚMENES A WEAVIATE")
    print("=" * 80)
    print(f"  URL: {args.url}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total resúmenes: {len(summaries)}")

    batch = WeaviateBatch(
        args.url,
        batch_size=args.batch_size,
        enabled=True,
        dry_run=args.dry_run,
        timeout=30,
    )

    counts: dict[str, int] = {}
    with batch:
        for s in summaries:
            stype = s.get("type", "unknown")
            mapping = _TYPE_MAP.get(stype)
            if mapping is None:
                logger.warning("Tipo desconocido: %s — sáltando", stype)
                continue
            class_name, builder = mapping
            props = builder(s)
            batch.add_object(class_name, props)
            counts[class_name] = counts.get(class_name, 0) + 1

    stats = batch.get_stats()
    print(f"\n  Resultados:")
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls}: {cnt}")
    print(f"\n  Total enviados: {stats['total_sent']}")
    print(f"  Exitosos: {stats['total_successful']}")
    print(f"  Fallos: {stats['total_failed']}")
    success = stats["total_failed"] == 0
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
