"""Tests para anomaly_narrator.py.

Verifica lógica de Narrative pura — sin I/O, sin sklearn, sin estado.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.anomaly.narration.builder import (
    build_anomaly_explanation,
)


class TestBuildAnomalyExplanation:
    """Tests para generación de explicación de anomalía."""

    def test_no_anomaly_returns_normal(self) -> None:
        votes = {"z_score": 0.0, "iqr": 0.0}
        assert build_anomaly_explanation(votes) == "Valor normal"

    def test_z_score_high(self) -> None:
        votes = {"z_score": 1.0, "iqr": 0.0}
        result = build_anomaly_explanation(votes, z_score=3.5)
        assert "Z-score alto" in result
        assert "3.5σ" in result

    def test_iqr_out_of_range(self) -> None:
        votes = {"z_score": 0.0, "iqr": 1.0}
        result = build_anomaly_explanation(votes)
        assert "Fuera de rango IQR" in result

    def test_isolation_forest(self) -> None:
        votes = {"z_score": 0.0, "iqr": 0.0, "isolation_forest": 1.0}
        result = build_anomaly_explanation(votes)
        assert "Aislado globalmente (IF)" in result

    def test_lof(self) -> None:
        votes = {"z_score": 0.0, "iqr": 0.0, "local_outlier_factor": 0.8}
        result = build_anomaly_explanation(votes)
        assert "Outlier local (LOF)" in result

    def test_multiple_explanations_joined(self) -> None:
        votes = {"z_score": 1.0, "iqr": 1.0, "isolation_forest": 1.0}
        result = build_anomaly_explanation(votes, z_score=4.0)
        assert " + " in result
        assert "Z-score alto" in result
        assert "Fuera de rango IQR" in result
        assert "Aislado globalmente (IF)" in result

    def test_z_vote_below_threshold_no_text(self) -> None:
        """Z-vote < 1.0 no genera texto de Z-score."""
        votes = {"z_score": 0.5, "iqr": 0.0}
        result = build_anomaly_explanation(votes, z_score=2.5)
        assert "Z-score" not in result

    def test_iqr_vote_below_threshold_no_text(self) -> None:
        votes = {"z_score": 0.0, "iqr": 0.3}
        result = build_anomaly_explanation(votes)
        assert "IQR" not in result

    def test_empty_votes(self) -> None:
        assert build_anomaly_explanation({}) == "Valor normal"

    def test_all_methods_anomalous(self) -> None:
        votes = {
            "z_score": 1.0,
            "iqr": 1.0,
            "isolation_forest": 1.0,
            "local_outlier_factor": 0.9,
        }
        result = build_anomaly_explanation(votes, z_score=5.0)
        parts = result.split(" + ")
        assert len(parts) == 4
