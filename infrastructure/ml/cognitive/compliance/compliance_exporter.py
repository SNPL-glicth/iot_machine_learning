"""ComplianceExporter — PredictionResult → NDJSON audit sink.

Design (IMP-5):
    * ``build_record`` is pure. Useful in tests / dry-run flows.
    * ``export`` writes the record to ``sink_path`` atomically
      (append + fsync) and returns it. Any exception during write is
      logged at WARNING and swallowed — export failure must never
      corrupt the pipeline's return value.
    * Thread-safe append via an instance-level :class:`threading.Lock`
      so concurrent pipeline runs cannot interleave NDJSON lines.
    * ``verify_record`` is a class-level helper that recomputes the
      HMAC over the canonical body and compares it constant-time.
    * ``clock`` / ``uuid_factory`` are injectable for deterministic
      tests.

Env knobs (consumed by :class:`AssemblyPhase`, not by this class):
    * ``ML_COMPLIANCE_EXPORT_PATH`` — NDJSON sink; enables export.
    * ``ML_COMPLIANCE_HMAC_KEY``    — hex (preferred) or UTF-8 string.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from .compliance_record import ComplianceRecord, _canonical_json

logger = logging.getLogger(__name__)


SCHEMA_VERSION: str = "1.0"


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


def _default_uuid_factory() -> str:
    return uuid.uuid4().hex


def _iso8601_utc(dt: datetime) -> str:
    """ISO-8601 UTC with microsecond precision, ``Z`` suffix."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def load_hmac_key_from_env(env_name: str = "ML_COMPLIANCE_HMAC_KEY") -> Optional[bytes]:
    """Return the HMAC key bytes from the environment, or ``None``.

    Tries hex decoding first; falls back to UTF-8 encoded raw string.
    Logs a one-line INFO with the key length (never the key itself)
    when successfully loaded.
    """
    raw = os.environ.get(env_name)
    if not raw:
        return None
    try:
        key = bytes.fromhex(raw)
        logger.info("compliance_hmac_key_loaded", extra={"length": len(key), "encoding": "hex"})
        return key
    except ValueError:
        key = raw.encode("utf-8")
        logger.info("compliance_hmac_key_loaded", extra={"length": len(key), "encoding": "utf8"})
        return key


class ComplianceExporter:
    """Builds ComplianceRecords and (optionally) appends them to NDJSON."""

    SCHEMA_VERSION: str = SCHEMA_VERSION

    def __init__(
        self,
        *,
        sink_path: Optional[Path] = None,
        fsync_each_write: bool = True,
        clock: Callable[[], datetime] = _default_clock,
        uuid_factory: Callable[[], str] = _default_uuid_factory,
        hmac_key: Optional[bytes] = None,
    ) -> None:
        self._sink_path = Path(sink_path) if sink_path is not None else None
        self._fsync = bool(fsync_each_write)
        self._clock = clock
        self._uuid_factory = uuid_factory
        self._hmac_key = hmac_key
        self._write_lock = Lock()

    # -- record construction ------------------------------------------

    def build_record(self, series_id: str, result: Any) -> ComplianceRecord:
        """Assemble a :class:`ComplianceRecord` from a PredictionResult.

        Raises ``ValueError`` for caller bugs (empty ``series_id`` or
        ``None`` result). Never swallows them because these indicate a
        programming mistake, not a runtime condition.
        """
        if result is None:
            raise ValueError("result is required")
        if not series_id:
            raise ValueError("series_id must be non-empty")

        metadata = dict(getattr(result, "metadata", {}) or {})

        outcome_dict: Dict[str, Any] = {
            "predicted_value": getattr(result, "predicted_value", None),
            "confidence": getattr(result, "confidence", 0.0),
            "trend": getattr(result, "trend", "stable"),
        }
        # If the pipeline embedded a richer Outcome via the explanation,
        # prefer its serialized form (more structured than the envelope).
        explanation = metadata.get("explanation")
        if isinstance(explanation, dict):
            inner = explanation.get("outcome")
            if isinstance(inner, dict):
                outcome_dict = dict(inner)

        explanation_digest = self._digest_explanation(explanation)

        record = ComplianceRecord.build(
            schema_version=self.SCHEMA_VERSION,
            record_id=self._uuid_factory(),
            created_at=_iso8601_utc(self._clock()),
            series_id=series_id,
            outcome=outcome_dict,
            sanitization_flags=list(metadata.get("sanitization_flags") or []),
            fusion_flags=list(metadata.get("fusion_flags") or []),
            engine_failures=list(metadata.get("engine_failures") or []),
            hampel=metadata.get("hampel"),
            pipeline_timing=metadata.get("pipeline_timing"),
            explanation_digest=explanation_digest,
            hmac_key=self._hmac_key,
        )
        return record

    # -- export -------------------------------------------------------

    def export(self, series_id: str, result: Any) -> Optional[ComplianceRecord]:
        """Build + write + return. Returns ``None`` on any write error.

        ``ValueError`` from caller bugs still propagates so they are
        caught during development, not silently dropped in prod.
        """
        record = self.build_record(series_id, result)
        if self._sink_path is None:
            return record
        try:
            self._write(record)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "compliance_export_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )
            return None
        return record

    # -- verification -------------------------------------------------

    @classmethod
    def verify_record(cls, record: ComplianceRecord, key: bytes) -> bool:
        """Return ``True`` iff the HMAC over the canonical body matches.

        Constant-time comparison. Returns ``False`` on missing HMAC or
        when the record's content_hash itself does not re-compute
        (catches both content tampering and wrong-key attempts).
        """
        if record.hmac_sha256 is None or not key:
            return False
        body = {
            "schema_version": record.schema_version,
            "record_id": record.record_id,
            "created_at": record.created_at,
            "series_id": record.series_id,
            "outcome": record.outcome,
            "sanitization_flags": list(record.sanitization_flags),
            "fusion_flags": list(record.fusion_flags),
            "engine_failures": list(record.engine_failures),
            "hampel": record.hampel,
            "pipeline_timing": record.pipeline_timing,
            "explanation_digest": record.explanation_digest,
        }
        canonical = _canonical_json(body).encode("utf-8")
        expected_content_hash = hashlib.sha256(canonical).hexdigest()
        if not hmac.compare_digest(expected_content_hash, record.content_hash):
            return False
        expected_hmac = hmac.new(key, canonical, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected_hmac, record.hmac_sha256)

    # -- helpers ------------------------------------------------------

    @staticmethod
    def _digest_explanation(explanation: Any) -> Optional[str]:
        if not isinstance(explanation, dict):
            return None
        try:
            canonical = _canonical_json(explanation).encode("utf-8")
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(canonical).hexdigest()

    def _write(self, record: ComplianceRecord) -> None:
        if self._sink_path is None:
            return
        line = record.to_json_line() + "\n"
        with self._write_lock:
            self._sink_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._sink_path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                if self._fsync:
                    os.fsync(fh.fileno())


__all__ = [
    "ComplianceExporter",
    "SCHEMA_VERSION",
    "load_hmac_key_from_env",
]
