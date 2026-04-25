"""ComplianceRecord — canonical audit entry for a single prediction.

Schema v1.0. Future additions are optional keys; ``schema_version``
advances to ``"1.1"``, ``"1.2"``, etc. so downstream consumers can
branch if they care.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _canonical_json(payload: Dict[str, Any]) -> str:
    """Deterministic JSON serialization used for hashing.

    * ``sort_keys=True`` — stable key order.
    * ``separators=(",", ":")`` — no whitespace.
    * ``default=str`` — falls back on ``str()`` for non-JSON types
      (defensive; should never fire on a well-formed record).
    * ``ensure_ascii=True`` — escapes all non-ASCII so the bytes are
      byte-identical across locales.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=True,
    )


@dataclass(frozen=True)
class ComplianceRecord:
    """One audit-grade entry for a prediction.

    Fields are immutable (``frozen=True``). ``content_hash`` covers all
    preceding fields; ``hmac_sha256`` (optional) signs the same bytes.
    """

    schema_version: str
    record_id: str
    created_at: str                        # ISO-8601 UTC, µs precision
    series_id: str
    outcome: Dict[str, Any]
    sanitization_flags: List[str] = field(default_factory=list)
    fusion_flags: List[str] = field(default_factory=list)
    engine_failures: List[Dict[str, Any]] = field(default_factory=list)
    hampel: Optional[Dict[str, Any]] = None
    pipeline_timing: Optional[Dict[str, Any]] = None
    explanation_digest: Optional[str] = None
    content_hash: str = ""
    hmac_sha256: Optional[str] = None

    # -- constructors -------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        schema_version: str,
        record_id: str,
        created_at: str,
        series_id: str,
        outcome: Dict[str, Any],
        sanitization_flags: List[str],
        fusion_flags: List[str],
        engine_failures: List[Dict[str, Any]],
        hampel: Optional[Dict[str, Any]],
        pipeline_timing: Optional[Dict[str, Any]],
        explanation_digest: Optional[str],
        hmac_key: Optional[bytes] = None,
    ) -> "ComplianceRecord":
        """Assemble a record with deterministic ``content_hash`` (+ HMAC)."""
        body = {
            "schema_version": schema_version,
            "record_id": record_id,
            "created_at": created_at,
            "series_id": series_id,
            "outcome": outcome,
            "sanitization_flags": list(sanitization_flags),
            "fusion_flags": list(fusion_flags),
            "engine_failures": list(engine_failures),
            "hampel": hampel,
            "pipeline_timing": pipeline_timing,
            "explanation_digest": explanation_digest,
        }
        canonical = _canonical_json(body).encode("utf-8")
        content_hash = hashlib.sha256(canonical).hexdigest()

        hmac_sha256: Optional[str] = None
        if hmac_key:
            hmac_sha256 = hmac.new(
                hmac_key, canonical, hashlib.sha256
            ).hexdigest()

        return cls(
            content_hash=content_hash,
            hmac_sha256=hmac_sha256,
            **body,
        )

    # -- serialization ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json_line(self) -> str:
        """Return a single-line canonical JSON representation.

        Suitable for NDJSON append. No trailing newline — the writer
        adds one.
        """
        return _canonical_json(self.to_dict())


__all__ = ["ComplianceRecord"]
