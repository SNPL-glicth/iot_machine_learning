"""Tests for core/decorrelation.py."""

import numpy as np
import pytest

from core.ensemble.decorrelation import EnsembleDecorrelator
from core.ensemble.ensemble_correlation import CorrelationLevel, CorrelationResult


class TestEnsembleDecorrelator:
    def test_init_with_defaults(self):
        decorrelator = EnsembleDecorrelator()
        assert decorrelator._correlation_threshold == 0.7
        assert decorrelator._weight_reduction_factor == 0.5

    def test_init_with_custom_params(self):
        decorrelator = EnsembleDecorrelator(
            correlation_threshold=0.8, weight_reduction_factor=0.3
        )
        assert decorrelator._correlation_threshold == 0.8
        assert decorrelator._weight_reduction_factor == 0.3

    def test_adjust_weights_for_correlation_empty(self):
        decorrelator = EnsembleDecorrelator()
        weights = {}
        matrix = np.array([])
        engine_names = []
        result = decorrelator.adjust_weights_for_correlation(weights, matrix, engine_names)
        assert result == {}

    def test_adjust_weights_for_correlation_single_engine(self):
        decorrelator = EnsembleDecorrelator()
        weights = {"engine1": 1.0}
        matrix = np.array([[1.0]])
        engine_names = ["engine1"]
        result = decorrelator.adjust_weights_for_correlation(weights, matrix, engine_names)
        assert result == weights  # No change for single engine

    def test_adjust_weights_for_correlation_high_correlation(self):
        decorrelator = EnsembleDecorrelator()
        weights = {"engine1": 0.5, "engine2": 0.5}
        matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        engine_names = ["engine1", "engine2"]
        result = decorrelator.adjust_weights_for_correlation(weights, matrix, engine_names)
        # engine2 weight should be reduced
        assert result["engine2"] < weights["engine2"]
        # Sum should be preserved at 1.0
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_adjust_weights_for_correlation_low_correlation(self):
        decorrelator = EnsembleDecorrelator()
        weights = {"engine1": 0.5, "engine2": 0.5}
        matrix = np.array([[1.0, 0.1], [0.1, 1.0]])
        engine_names = ["engine1", "engine2"]
        result = decorrelator.adjust_weights_for_correlation(weights, matrix, engine_names)
        # No change for low correlation
        assert result == weights

    def test_apply_if_needed_high_correlation(self):
        decorrelator = EnsembleDecorrelator()
        weights = {"engine1": 0.5, "engine2": 0.5}
        correlation_result = CorrelationResult(
            matrix=np.array([[1.0, 0.8], [0.8, 1.0]]),
            engine_names=["engine1", "engine2"],
            classification={"engine_0": CorrelationLevel.HIGH, "engine_1": CorrelationLevel.HIGH},
            recommendations=["HIGH correlation detected"],
            max_correlation=0.8,
            avg_correlation=0.8,
        )
        result, applied = decorrelator.apply_if_needed(weights, correlation_result)
        assert applied is True
        assert result != weights

    def test_apply_if_needed_low_correlation(self):
        decorrelator = EnsembleDecorrelator()
        weights = {"engine1": 0.5, "engine2": 0.5}
        correlation_result = CorrelationResult(
            matrix=np.array([[1.0, 0.1], [0.1, 1.0]]),
            engine_names=["engine1", "engine2"],
            classification={"engine_0": CorrelationLevel.LOW, "engine_1": CorrelationLevel.LOW},
            recommendations=["LOW correlation"],
            max_correlation=0.1,
            avg_correlation=0.1,
        )
        result, applied = decorrelator.apply_if_needed(weights, correlation_result)
        assert applied is False
        assert result == weights

    def test_compute_diversity_score(self):
        decorrelator = EnsembleDecorrelator()
        correlation_result = CorrelationResult(
            matrix=np.array([[1.0, 0.8], [0.8, 1.0]]),
            engine_names=["engine1", "engine2"],
            classification={},
            recommendations=[],
            max_correlation=0.8,
            avg_correlation=0.8,
        )
        diversity = decorrelator.compute_diversity_score(correlation_result)
        assert abs(diversity - 0.2) < 0.001  # 1 - 0.8 (floating point tolerance)
