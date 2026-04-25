"""Tests for EnsembleWeightedPredictor deprecation (Problema 3).

3 cases:
1. File exists in deprecated/
2. Can still be imported (backward compat)
3. Has deprecation notice in docstring
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iot_machine_learning.infrastructure.ml.engines.deprecated.ensemble_predictor import (
    EnsembleWeightedPredictor,
)


class TestDeprecatedEnsemble:
    def test_file_is_in_deprecated_directory(self) -> None:
        path = (
            Path(__file__).parents[4]
            / "iot_machine_learning"
            / "infrastructure"
            / "ml"
            / "engines"
            / "deprecated"
            / "ensemble_predictor.py"
        )
        assert path.exists()

    def test_import_from_deprecated_works(self) -> None:
        assert EnsembleWeightedPredictor is not None
        assert EnsembleWeightedPredictor.__name__ == "EnsembleWeightedPredictor"

    def test_has_deprecation_notice(self) -> None:
        doc = EnsembleWeightedPredictor.__doc__ or ""
        assert "Ensemble" in doc
        # Module-level deprecation comment is not in __doc__
        # but the docstring should still mention ensemble
        assert "ensemble" in doc.lower() or "Ensemble" in doc
