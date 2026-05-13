"""Tests for core/ensemble_correlation.py."""

import numpy as np
import pytest

from core.ensemble.ensemble_correlation import (
    EngineCorrelationAnalyzer,
    CorrelationLevel,
    CorrelationResult,
)


class TestEngineCorrelationAnalyzer:
    def test_init_with_defaults(self):
        analyzer = EngineCorrelationAnalyzer()
        assert analyzer._low_threshold == 0.3
        assert analyzer._high_threshold == 0.7

    def test_init_with_custom_thresholds(self):
        analyzer = EngineCorrelationAnalyzer(low_threshold=0.2, high_threshold=0.8)
        assert analyzer._low_threshold == 0.2
        assert analyzer._high_threshold == 0.8

    def test_compute_correlation_matrix_empty(self):
        analyzer = EngineCorrelationAnalyzer()
        matrix = analyzer.compute_correlation_matrix({})
        assert matrix.shape == (0, 0)

    def test_compute_correlation_matrix_single_engine(self):
        analyzer = EngineCorrelationAnalyzer()
        # Need multiple time points for correlation
        predictions = {"engine1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        matrix = analyzer.compute_correlation_matrix(predictions)
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 1.0

    def test_compute_correlation_matrix_multiple_engines(self):
        analyzer = EngineCorrelationAnalyzer()
        predictions = {
            "engine1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "engine2": np.array([1.1, 2.1, 3.1, 4.1, 5.1]),
        }
        matrix = analyzer.compute_correlation_matrix(predictions)
        assert matrix.shape == (2, 2)
        assert np.allclose(matrix[0, 0], 1.0)
        assert np.allclose(matrix[1, 1], 1.0)
        assert matrix[0, 1] > 0.9  # High correlation

    def test_compute_correlation_matrix_insufficient_data(self):
        analyzer = EngineCorrelationAnalyzer()
        predictions = {"engine1": np.array([1.0]), "engine2": np.array([1.0])}
        matrix = analyzer.compute_correlation_matrix(predictions)
        assert matrix.shape == (2, 2)
        assert np.allclose(matrix, np.eye(2))  # Returns identity for insufficient data

    def test_classify_correlation_low(self):
        analyzer = EngineCorrelationAnalyzer()
        matrix = np.array([[1.0, 0.1], [0.1, 1.0]])
        classification = analyzer.classify_correlation(matrix)
        assert classification["engine_0"] == CorrelationLevel.LOW
        assert classification["engine_1"] == CorrelationLevel.LOW

    def test_classify_correlation_moderate(self):
        analyzer = EngineCorrelationAnalyzer()
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        classification = analyzer.classify_correlation(matrix)
        assert classification["engine_0"] == CorrelationLevel.MODERATE
        assert classification["engine_1"] == CorrelationLevel.MODERATE

    def test_classify_correlation_high(self):
        analyzer = EngineCorrelationAnalyzer()
        matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        classification = analyzer.classify_correlation(matrix)
        assert classification["engine_0"] == CorrelationLevel.HIGH
        assert classification["engine_1"] == CorrelationLevel.HIGH

    def test_analyze_complete(self):
        analyzer = EngineCorrelationAnalyzer()
        predictions = {
            "engine1": np.array([1.0, 2.0, 3.0, 4.0]),
            "engine2": np.array([1.1, 2.1, 3.1, 4.1]),
        }
        result = analyzer.analyze(predictions)
        assert isinstance(result, CorrelationResult)
        assert result.max_correlation > 0.9
        assert len(result.recommendations) > 0
        assert result.engine_names == ["engine1", "engine2"]
