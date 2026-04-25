"""Tests for ComplianceExporter + ComplianceRecord (IMP-5).

Covers all 13 spec items + 5 HMAC cases + 3 AssemblyPhase integration
cases (see test_compliance_assembly_integration.py for the latter).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.compliance import (
    ComplianceExporter,
    ComplianceRecord,
)
from iot_machine_learning.infrastructure.ml.cognitive.compliance.compliance_exporter import (
    load_hmac_key_from_env,
)
from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult


# -- helpers --------------------------------------------------------------


def _fixed_clock() -> datetime:
    return datetime(2026, 4, 24, 15, 30, 45, 123456, tzinfo=timezone.utc)


def _fixed_uuid() -> str:
    return "deadbeefdeadbeefdeadbeefdeadbeef"


def _make_result(**metadata) -> PredictionResult:
    return PredictionResult(
        predicted_value=42.5,
        confidence=0.87,
        trend="up",
        metadata=metadata,
    )


# =========================================================================
# build_record — spec items 1, 2, 12, 13
# =========================================================================


class TestBuildRecord:
    def test_build_record_minimal_result(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock, uuid_factory=_fixed_uuid)
        result = _make_result()
        record = exporter.build_record("sid-1", result)
        
        assert record.schema_version == "1.0"
        assert record.series_id == "sid-1"
        assert record.record_id == _fixed_uuid()
        assert record.sanitization_flags == []
        assert record.fusion_flags == []
        assert record.engine_failures == []
        assert record.hampel is None
        assert record.pipeline_timing is None
        assert record.explanation_digest is None
        assert record.hmac_sha256 is None
        assert record.content_hash  # non-empty
        assert record.outcome["predicted_value"] == 42.5
        assert record.outcome["confidence"] == 0.87
        assert record.outcome["trend"] == "up"

    def test_build_record_full_result(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock, uuid_factory=_fixed_uuid)
        result = _make_result(
            sanitization_flags=["value_clamped:1"],
            fusion_flags=["hampel_rejected:1"],
            engine_failures=[{"engine": "slow", "reason": "timeout"}],
            hampel={"median": 10.0, "mad": 1.0, "rejected": []},
            pipeline_timing={"total_ms": 123.4},
            explanation={"version": "1.0", "outcome": {"confidence": 0.8}},
        )
        record = exporter.build_record("sid", result)
        
        assert record.sanitization_flags == ["value_clamped:1"]
        assert record.fusion_flags == ["hampel_rejected:1"]
        assert record.engine_failures == [{"engine": "slow", "reason": "timeout"}]
        assert record.hampel == {"median": 10.0, "mad": 1.0, "rejected": []}
        assert record.pipeline_timing == {"total_ms": 123.4}
        assert record.explanation_digest is not None
        # Prefers the richer outcome embedded in the explanation.
        assert record.outcome == {"confidence": 0.8}

    def test_build_record_rejects_empty_series_id(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock)
        with pytest.raises(ValueError, match="series_id"):
            exporter.build_record("", _make_result())

    def test_build_record_rejects_none_result(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock)
        with pytest.raises(ValueError, match="result"):
            exporter.build_record("sid", None)


# =========================================================================
# Hashing / determinism — spec items 3, 4, 5, 6, 7
# =========================================================================


class TestHashing:
    def test_record_fields_immutable(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock, uuid_factory=_fixed_uuid)
        record = exporter.build_record("sid", _make_result())
        with pytest.raises(FrozenInstanceError):
            record.content_hash = "tampered"  # type: ignore[misc]

    def test_content_hash_deterministic(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock, uuid_factory=_fixed_uuid)
        result = _make_result(sanitization_flags=["x"])
        a = exporter.build_record("sid", result)
        b = exporter.build_record("sid", result)
        assert a.content_hash == b.content_hash
        assert a.to_json_line() == b.to_json_line()

    def test_content_hash_changes_on_mutation(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock, uuid_factory=_fixed_uuid)
        a = exporter.build_record("sid", _make_result())
        b = exporter.build_record("sid-other", _make_result())  # different series_id
        assert a.content_hash != b.content_hash

    def test_explanation_digest_present_when_explanation_provided(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock, uuid_factory=_fixed_uuid)
        result = _make_result(explanation={"foo": "bar"})
        record = exporter.build_record("sid", result)
        assert record.explanation_digest is not None
        assert len(record.explanation_digest) == 64  # sha256 hex

    def test_explanation_digest_absent_without_explanation(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock, uuid_factory=_fixed_uuid)
        record = exporter.build_record("sid", _make_result())
        assert record.explanation_digest is None

    def test_to_json_line_is_single_line_canonical(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock, uuid_factory=_fixed_uuid)
        record = exporter.build_record("sid", _make_result())
        line = record.to_json_line()
        assert "\n" not in line
        # Sorted keys → 'content_hash' < 'created_at' alphabetically.
        parsed = json.loads(line)
        assert parsed["series_id"] == "sid"


# =========================================================================
# Export (file writes) — spec items 8, 9, 10, 11
# =========================================================================


class TestExport:
    def test_export_writes_ndjson(self, tmp_path: Path) -> None:
        sink = tmp_path / "audit.ndjson"
        exporter = ComplianceExporter(
            sink_path=sink, clock=_fixed_clock, uuid_factory=_fixed_uuid
        )
        record = exporter.export("sid", _make_result())
        assert record is not None
        content = sink.read_text().strip()
        assert content == record.to_json_line()

    def test_export_appends(self, tmp_path: Path) -> None:
        sink = tmp_path / "audit.ndjson"
        exporter = ComplianceExporter(sink_path=sink, clock=_fixed_clock, uuid_factory=_fixed_uuid)
        exporter.export("a", _make_result())
        exporter.export("b", _make_result())
        lines = sink.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_export_concurrent_writes_no_interleave(self, tmp_path: Path) -> None:
        sink = tmp_path / "audit.ndjson"
        exporter = ComplianceExporter(sink_path=sink, uuid_factory=_fixed_uuid)
        N = 16
        barrier = threading.Barrier(N)
        
        def worker(i: int) -> None:
            barrier.wait()
            exporter.export(f"sid-{i}", _make_result())
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        lines = sink.read_text().strip().split("\n")
        assert len(lines) == N
        # Every line is valid JSON (no interleaving corrupted content).
        for ln in lines:
            parsed = json.loads(ln)
            assert "content_hash" in parsed

    def test_export_swallows_write_errors(self, tmp_path: Path) -> None:
        """Unwritable sink → returns None, does not raise."""
        # Make the parent a file instead of a directory — open() will fail.
        bad = tmp_path / "blocker"
        bad.write_text("i am a file, not a directory")
        sink = bad / "nested" / "audit.ndjson"
        
        exporter = ComplianceExporter(sink_path=sink, clock=_fixed_clock, uuid_factory=_fixed_uuid)
        # ValueError for caller bugs still propagates, but IOError → None.
        result = exporter.export("sid", _make_result())
        assert result is None


# =========================================================================
# HMAC — 5 additional cases
# =========================================================================


class TestHMAC:
    def test_hmac_signature_present_when_key_set(self) -> None:
        key = b"test-key-32-bytes-long-enough-ok!!"
        exporter = ComplianceExporter(
            hmac_key=key, clock=_fixed_clock, uuid_factory=_fixed_uuid
        )
        record = exporter.build_record("sid", _make_result())
        assert record.hmac_sha256 is not None
        assert len(record.hmac_sha256) == 64

    def test_hmac_signature_absent_when_key_unset(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock, uuid_factory=_fixed_uuid)
        record = exporter.build_record("sid", _make_result())
        assert record.hmac_sha256 is None

    def test_hmac_verify_accepts_valid_signature(self) -> None:
        key = b"mykey-0123456789"
        exporter = ComplianceExporter(hmac_key=key, clock=_fixed_clock, uuid_factory=_fixed_uuid)
        record = exporter.build_record("sid", _make_result())
        assert ComplianceExporter.verify_record(record, key) is True

    def test_hmac_verify_rejects_wrong_key(self) -> None:
        key_a = b"key-A"
        key_b = b"key-B"
        exporter = ComplianceExporter(hmac_key=key_a, clock=_fixed_clock, uuid_factory=_fixed_uuid)
        record = exporter.build_record("sid", _make_result())
        assert ComplianceExporter.verify_record(record, key_b) is False

    def test_hmac_verify_rejects_tampered_content(self) -> None:
        """Tampered record → verification fails even with the correct key."""
        from dataclasses import replace
        key = b"key"
        exporter = ComplianceExporter(hmac_key=key, clock=_fixed_clock, uuid_factory=_fixed_uuid)
        record = exporter.build_record("sid", _make_result())
        # Simulate tampering: create a record with a different series_id but
        # the original hmac_sha256/content_hash. Since record is frozen,
        # use dataclasses.replace().
        tampered = replace(record, series_id="evil")
        assert ComplianceExporter.verify_record(tampered, key) is False

    def test_hmac_verify_without_hmac_returns_false(self) -> None:
        exporter = ComplianceExporter(clock=_fixed_clock, uuid_factory=_fixed_uuid)
        record = exporter.build_record("sid", _make_result())
        assert record.hmac_sha256 is None
        assert ComplianceExporter.verify_record(record, b"anykey") is False


# =========================================================================
# Env helpers
# =========================================================================


class TestEnvHelpers:
    def test_load_hmac_key_missing_env(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ML_COMPLIANCE_HMAC_KEY", None)
            assert load_hmac_key_from_env() is None

    def test_load_hmac_key_hex(self) -> None:
        hex_key = "deadbeef" * 4  # 16 bytes
        with patch.dict(os.environ, {"ML_COMPLIANCE_HMAC_KEY": hex_key}):
            key = load_hmac_key_from_env()
            assert key == bytes.fromhex(hex_key)

    def test_load_hmac_key_utf8_fallback(self) -> None:
        # Contains non-hex characters → falls back to UTF-8 encoding.
        with patch.dict(os.environ, {"ML_COMPLIANCE_HMAC_KEY": "my-secret-z"}):
            key = load_hmac_key_from_env()
            assert key == b"my-secret-z"
