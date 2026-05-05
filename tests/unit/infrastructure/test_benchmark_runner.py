"""Unit tests for benchmark system.

Tests dataset loading, metrics computation, and benchmark execution.
"""

import tempfile
from pathlib import Path

import pytest

from iot_machine_learning.infrastructure.ml.benchmark import (
    DatasetLoader,
    DatasetSample,
    BenchmarkMetrics,
    MetricsResult,
    BenchmarkRunner,
    BenchmarkReport,
)


class TestDatasetLoader:
    """Test DatasetLoader class."""
    
    def test_load_csv_basic(self):
        """Should load basic CSV dataset."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,value,label\n")
            f.write("1.0,10.5,0\n")
            f.write("2.0,11.2,0\n")
            f.write("3.0,25.8,1\n")
            f.write("4.0,12.1,0\n")
            temp_path = f.name
        
        try:
            samples = DatasetLoader.load_csv(temp_path)
            
            assert len(samples) == 4
            assert samples[0] == DatasetSample(1.0, 10.5, False)
            assert samples[2] == DatasetSample(3.0, 25.8, True)
        finally:
            Path(temp_path).unlink()
    
    def test_load_nab_format(self):
        """Should load NAB format dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,value,label\n")
            f.write("100.0,5.5,0\n")
            f.write("101.0,50.0,1\n")
            temp_path = f.name
        
        try:
            samples = DatasetLoader.load_nab(temp_path)
            
            assert len(samples) == 2
            assert samples[1].is_anomaly is True
        finally:
            Path(temp_path).unlink()
    
    def test_load_yahoo_format(self):
        """Should load Yahoo format dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,value,is_anomaly\n")
            f.write("1.0,10.0,0\n")
            f.write("2.0,100.0,1\n")
            temp_path = f.name
        
        try:
            samples = DatasetLoader.load_yahoo(temp_path)
            
            assert len(samples) == 2
            assert samples[1].is_anomaly is True
        finally:
            Path(temp_path).unlink()
    
    def test_load_missing_file_raises_error(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader.load_csv("/nonexistent/file.csv")
    
    def test_load_alternative_label_column(self):
        """Should handle alternative label column names."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,value,anomaly\n")
            f.write("1.0,10.0,0\n")
            f.write("2.0,100.0,1\n")
            temp_path = f.name
        
        try:
            samples = DatasetLoader.load_csv(
                temp_path,
                label_col="anomaly",
            )
            
            assert len(samples) == 2
            assert samples[1].is_anomaly is True
        finally:
            Path(temp_path).unlink()


class TestBenchmarkMetrics:
    """Test BenchmarkMetrics class."""
    
    def test_compute_perfect_classification(self):
        """Should compute metrics for perfect classification."""
        predictions = [False, False, True, True, False]
        ground_truth = [False, False, True, True, False]
        
        metrics = BenchmarkMetrics.compute(predictions, ground_truth)
        
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.false_positive_rate == 0.0
        assert metrics.true_positives == 2
        assert metrics.false_positives == 0
        assert metrics.true_negatives == 3
        assert metrics.false_negatives == 0
    
    def test_compute_all_false_positives(self):
        """Should compute metrics for all false positives."""
        predictions = [True, True, True]
        ground_truth = [False, False, False]
        
        metrics = BenchmarkMetrics.compute(predictions, ground_truth)
        
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.false_positive_rate == 1.0
        assert metrics.false_positives == 3
    
    def test_compute_mixed_results(self):
        """Should compute metrics for mixed results."""
        # TP=2, FP=1, TN=2, FN=1
        predictions = [True, False, True, True, False, False]
        ground_truth = [True, False, False, True, True, False]
        
        metrics = BenchmarkMetrics.compute(predictions, ground_truth)
        
        # precision = 2 / (2 + 1) = 0.667
        assert abs(metrics.precision - 0.667) < 0.01
        # recall = 2 / (2 + 1) = 0.667
        assert abs(metrics.recall - 0.667) < 0.01
        # f1 = 2 * 0.667 * 0.667 / (0.667 + 0.667) = 0.667
        assert abs(metrics.f1_score - 0.667) < 0.01
        # fpr = 1 / (1 + 2) = 0.333
        assert abs(metrics.false_positive_rate - 0.333) < 0.01
    
    def test_compute_detection_delay(self):
        """Should compute detection delay correctly."""
        # Anomaly at index 2-3, detected at index 3 (delay=1)
        predictions = [False, False, False, True, False]
        ground_truth = [False, False, True, True, False]
        
        metrics = BenchmarkMetrics.compute(predictions, ground_truth, compute_delay=True)
        
        assert metrics.avg_detection_delay == 1.0
    
    def test_compute_no_detection_delay(self):
        """Should return None when no anomalies detected."""
        predictions = [False, False, False]
        ground_truth = [False, True, False]
        
        metrics = BenchmarkMetrics.compute(predictions, ground_truth, compute_delay=True)
        
        assert metrics.avg_detection_delay is None
    
    def test_compute_mismatched_lengths_raises_error(self):
        """Should raise ValueError for mismatched lengths."""
        predictions = [True, False]
        ground_truth = [True, False, True]
        
        with pytest.raises(ValueError, match="must have same length"):
            BenchmarkMetrics.compute(predictions, ground_truth)
    
    def test_metrics_to_dict(self):
        """Should convert metrics to dictionary."""
        predictions = [True, False, True]
        ground_truth = [True, False, False]
        
        metrics = BenchmarkMetrics.compute(predictions, ground_truth)
        result_dict = metrics.to_dict()
        
        assert "precision" in result_dict
        assert "recall" in result_dict
        assert "f1_score" in result_dict
        assert "false_positive_rate" in result_dict
        assert "confusion_matrix" in result_dict


class TestBenchmarkRunner:
    """Test BenchmarkRunner class."""
    
    def test_runner_initialization(self):
        """Should initialize with valid parameters."""
        runner = BenchmarkRunner(decision_threshold=0.8, window_size=30)
        
        assert runner._decision_threshold == 0.8
        assert runner._window_size == 30
    
    def test_runner_invalid_threshold_raises_error(self):
        """Should raise ValueError for invalid threshold."""
        with pytest.raises(ValueError, match="decision_threshold must be in"):
            BenchmarkRunner(decision_threshold=1.5)
        
        with pytest.raises(ValueError, match="decision_threshold must be in"):
            BenchmarkRunner(decision_threshold=-0.1)
    
    def test_run_synthetic_dataset(self):
        """Should run benchmark on synthetic dataset."""
        # Create synthetic dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,value,label\n")
            # Normal samples
            for i in range(20):
                f.write(f"{i}.0,{10.0 + i * 0.1},0\n")
            # Anomaly samples
            for i in range(20, 25):
                f.write(f"{i}.0,{100.0},1\n")
            # Normal samples
            for i in range(25, 40):
                f.write(f"{i}.0,{10.0 + i * 0.1},0\n")
            temp_path = f.name
        
        try:
            # Mock predict function that returns high confidence for value > 50
            def mock_predict(values, timestamps):
                latest_value = values[-1]
                confidence = 0.9 if latest_value > 50 else 0.3
                return {"fused_confidence": confidence}
            
            runner = BenchmarkRunner(decision_threshold=0.7, window_size=10)
            report = runner.run(
                dataset_path=temp_path,
                predict_fn=mock_predict,
                dataset_format="nab",
            )
            
            assert report.n_samples == 40
            assert report.n_anomalies == 5
            assert report.decision_threshold == 0.7
            assert 0.0 <= report.metrics.precision <= 1.0
            assert 0.0 <= report.metrics.recall <= 1.0
        finally:
            Path(temp_path).unlink()
    
    def test_run_invalid_format_raises_error(self):
        """Should raise ValueError for invalid dataset format."""
        runner = BenchmarkRunner()
        
        def mock_predict(values, timestamps):
            return {"fused_confidence": 0.5}
        
        with pytest.raises(ValueError, match="Invalid dataset_format"):
            runner.run(
                dataset_path="/tmp/test.csv",
                predict_fn=mock_predict,
                dataset_format="invalid",
            )
    
    def test_aggregate_reports(self):
        """Should aggregate metrics from multiple reports."""
        # Create mock reports
        metrics1 = MetricsResult(
            precision=0.8,
            recall=0.7,
            f1_score=0.75,
            false_positive_rate=0.1,
            avg_detection_delay=2.0,
            true_positives=10,
            false_positives=2,
            true_negatives=20,
            false_negatives=3,
        )
        
        metrics2 = MetricsResult(
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            false_positive_rate=0.05,
            avg_detection_delay=1.5,
            true_positives=15,
            false_positives=1,
            true_negatives=25,
            false_negatives=2,
        )
        
        report1 = BenchmarkReport("dataset1", 100, 10, metrics1, 0.7)
        report2 = BenchmarkReport("dataset2", 120, 15, metrics2, 0.7)
        
        aggregated = BenchmarkRunner.aggregate_reports([report1, report2])
        
        assert aggregated["n_datasets"] == 2
        assert aggregated["avg_precision"] == 0.85
        assert aggregated["avg_recall"] == 0.75
        assert abs(aggregated["avg_f1_score"] - 0.8) < 0.01
        assert "datasets" in aggregated
    
    def test_report_to_dict(self):
        """Should convert report to dictionary."""
        metrics = MetricsResult(
            precision=0.8,
            recall=0.7,
            f1_score=0.75,
            false_positive_rate=0.1,
            avg_detection_delay=2.0,
            true_positives=10,
            false_positives=2,
            true_negatives=20,
            false_negatives=3,
        )
        
        report = BenchmarkReport("test_dataset", 100, 10, metrics, 0.7)
        report_dict = report.to_dict()
        
        assert report_dict["dataset"] == "test_dataset"
        assert report_dict["n_samples"] == 100
        assert report_dict["n_anomalies"] == 10
        assert report_dict["decision_threshold"] == 0.7
        assert "precision" in report_dict
        assert "recall" in report_dict
