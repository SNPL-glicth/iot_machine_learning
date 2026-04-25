"""AssemblyPhase ↔ ComplianceExporter integration (IMP-5 §4).

Covers:
* Export happens when ML_COMPLIANCE_EXPORT_PATH is set.
* Export is skipped when the env var is unset.
* Export failure does not corrupt the returned PredictionResult.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import (
    PipelineContext,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.assembly_phase import (
    AssemblyPhase,
    _reset_compliance_exporter,
)


@pytest.fixture(autouse=True)
def _reset_exporter_singleton():
    _reset_compliance_exporter()
    yield
    _reset_compliance_exporter()


def _build_ctx() -> PipelineContext:
    orch = MagicMock()
    orch._storage = None
    timer = MagicMock()
    timer.to_dict = lambda: {"total_ms": 12.3}
    ctx = PipelineContext(
        orchestrator=orch,
        values=[1.0, 2.0, 3.0],
        timestamps=None,
        series_id="sid-audit",
        flags=None,
        timer=timer,
        fused_value=10.0,
        fused_confidence=0.9,
        fused_trend="up",
    )
    return ctx


class TestComplianceAssemblyIntegration:
    def test_assembly_phase_exports_when_env_set(self, tmp_path: Path) -> None:
        sink = tmp_path / "audit.ndjson"
        with patch.dict(os.environ, {"ML_COMPLIANCE_EXPORT_PATH": str(sink)}):
            result = AssemblyPhase().execute(_build_ctx())
        assert sink.exists()
        content = sink.read_text().strip()
        assert content  # non-empty
        parsed = json.loads(content)
        assert parsed["series_id"] == "sid-audit"
        assert parsed["schema_version"] == "1.0"
        # AssemblyPhase still returned a valid result.
        assert result.predicted_value == 10.0
        assert result.confidence == 0.9

    def test_assembly_phase_skips_export_when_env_unset(self, tmp_path: Path) -> None:
        # Env var unset — no sink file should appear.
        sink = tmp_path / "audit.ndjson"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ML_COMPLIANCE_EXPORT_PATH", None)
            result = AssemblyPhase().execute(_build_ctx())
        assert not sink.exists()
        assert result.predicted_value == 10.0

    def test_assembly_phase_swallows_export_errors(self, tmp_path: Path) -> None:
        """Unwritable sink path: AssemblyPhase still returns a valid result."""
        # Point the sink at a path whose parent is a file, not a dir.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        sink = blocker / "nested" / "audit.ndjson"
        with patch.dict(os.environ, {"ML_COMPLIANCE_EXPORT_PATH": str(sink)}):
            # Must not raise.
            result = AssemblyPhase().execute(_build_ctx())
        assert result.predicted_value == 10.0
        assert result.confidence == 0.9

    def test_metadata_untouched_by_export_hook(self, tmp_path: Path) -> None:
        """The post-export hook must not mutate the returned metadata."""
        sink = tmp_path / "audit.ndjson"
        with patch.dict(os.environ, {"ML_COMPLIANCE_EXPORT_PATH": str(sink)}):
            result = AssemblyPhase().execute(_build_ctx())
        # IMP-1/IMP-2 fields still present in metadata envelope.
        assert "sanitization_flags" in result.metadata
        assert "fusion_flags" in result.metadata
        assert "engine_failures" in result.metadata

    def test_singleton_reused_across_calls(self, tmp_path: Path) -> None:
        """Two predictions append to the SAME sink file."""
        sink = tmp_path / "audit.ndjson"
        with patch.dict(os.environ, {"ML_COMPLIANCE_EXPORT_PATH": str(sink)}):
            AssemblyPhase().execute(_build_ctx())
            AssemblyPhase().execute(_build_ctx())
        lines = sink.read_text().strip().split("\n")
        assert len(lines) == 2
