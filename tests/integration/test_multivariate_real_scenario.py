"""Integration test — real multivariate anomaly detection scenario.

Tests the full pipeline with correlated sensors and synchronized spike.
"""

import numpy as np
import pytest


def test_multivariate_real_scenario():
    """Real scenario: 3 correlated sensors with synchronized spike.
    
    Scenario:
    - 3 sensors normally correlated (temperature, humidity, pressure)
    - Normal operation: slow drift
    - Anomaly: synchronized spike at t=50
    
    Expected:
    - Multivariate detector should score high
    - Confidence calibration should produce valid probability
    - Decision should detect anomaly
    """
    from iot_machine_learning.infrastructure.ml.anomaly.detectors import MultivariateDetector
    from iot_machine_learning.infrastructure.ml.calibration import ConfidenceCalibrator
    
    # Create detector
    detector = MultivariateDetector(
        min_series=3,
        enabled=True,
        pca_components=2,
        baseline_percentile=90.0,
        warmup_samples=20,
    )
    
    # Create calibrator
    calibrator = ConfidenceCalibrator(temperature=1.5)
    
    # Generate correlated normal data (50 samples)
    np.random.seed(42)
    n_normal = 50
    
    # Sensor 1: temperature (base pattern)
    temp_normal = 20.0 + np.linspace(0, 2, n_normal) + np.random.randn(n_normal) * 0.1
    
    # Sensor 2: humidity (correlated with temperature)
    humidity_normal = 60.0 - 0.5 * (temp_normal - 20.0) + np.random.randn(n_normal) * 0.2
    
    # Sensor 3: pressure (correlated with temperature)
    pressure_normal = 1013.0 + 0.3 * (temp_normal - 20.0) + np.random.randn(n_normal) * 0.15
    
    # Train on normal data
    normal_scores = []
    for i in range(10, n_normal):
        # Window of last 10 samples
        temp_window = temp_normal[i-10:i+1].tolist()
        humidity_window = humidity_normal[i-10:i+1].tolist()
        pressure_window = pressure_normal[i-10:i+1].tolist()
        
        # Detect
        score = detector.detect(
            values=temp_window,
            series_id="sensor_temp",
            correlated_series_data={
                "sensor_humidity": humidity_window,
                "sensor_pressure": pressure_window,
            }
        )
        
        normal_scores.append(score)
    
    # Generate anomaly: synchronized spike
    temp_anomaly = temp_normal[-1] + 10.0  # +10°C spike
    humidity_anomaly = humidity_normal[-1] - 5.0  # -5% spike (inverse correlation)
    pressure_anomaly = pressure_normal[-1] + 3.0  # +3 hPa spike
    
    # Detect on anomaly
    temp_window_anomaly = list(temp_normal[-10:]) + [temp_anomaly]
    humidity_window_anomaly = list(humidity_normal[-10:]) + [humidity_anomaly]
    pressure_window_anomaly = list(pressure_normal[-10:]) + [pressure_anomaly]
    
    anomaly_score = detector.detect(
        values=temp_window_anomaly,
        series_id="sensor_temp",
        correlated_series_data={
            "sensor_humidity": humidity_window_anomaly,
            "sensor_pressure": pressure_window_anomaly,
        }
    )
    
    # Calibrate scores
    avg_normal_score = np.mean(normal_scores) if normal_scores else 0.0
    calibrated_normal = calibrator.calibrate(avg_normal_score)
    calibrated_anomaly = calibrator.calibrate(anomaly_score)
    
    # Print results
    print("\n" + "="*60)
    print("MULTIVARIATE REAL SCENARIO RESULTS")
    print("="*60)
    print(f"\n📊 Normal Operation:")
    print(f"   Average multivariate score: {avg_normal_score:.4f}")
    print(f"   Calibrated confidence: {calibrated_normal:.4f}")
    
    print(f"\n🚨 Synchronized Spike:")
    print(f"   Temperature: {temp_normal[-1]:.2f}°C → {temp_anomaly:.2f}°C (+{temp_anomaly - temp_normal[-1]:.2f})")
    print(f"   Humidity: {humidity_normal[-1]:.2f}% → {humidity_anomaly:.2f}% ({humidity_anomaly - humidity_normal[-1]:.2f})")
    print(f"   Pressure: {pressure_normal[-1]:.2f} hPa → {pressure_anomaly:.2f} hPa (+{pressure_anomaly - pressure_normal[-1]:.2f})")
    
    print(f"\n📈 Detection:")
    print(f"   Multivariate score: {anomaly_score:.4f}")
    print(f"   Calibrated confidence: {calibrated_anomaly:.4f}")
    
    # Decision (lower threshold for test)
    decision_threshold = 0.6
    is_anomaly = calibrated_anomaly >= decision_threshold
    
    print(f"\n✅ Decision (threshold={decision_threshold}):")
    print(f"   Anomaly detected: {is_anomaly}")
    print(f"   Score increase: {anomaly_score / (avg_normal_score + 1e-6):.2f}x")
    print("="*60 + "\n")
    
    # Assertions
    assert anomaly_score > 0.0, "Multivariate score should be non-zero"
    assert anomaly_score > avg_normal_score, "Anomaly score should be higher than normal"
    assert 0.0 <= calibrated_anomaly <= 1.0, "Calibrated confidence should be in [0, 1]"
    
    # If sklearn is available, score should be significantly higher
    try:
        import sklearn
        assert anomaly_score > avg_normal_score * 1.5, "Anomaly should be 1.5x higher with sklearn"
        assert is_anomaly, "Should detect anomaly with sklearn"
    except ImportError:
        # Without sklearn, detector returns 0.0 (graceful fallback)
        pass


def test_multivariate_confidence_integration():
    """Test integration between multivariate detector and confidence calibrator.
    
    Verifies that:
    - Raw scores are properly normalized
    - Calibration produces valid probabilities
    - Different regimes affect calibration
    """
    from iot_machine_learning.infrastructure.ml.anomaly.detectors import MultivariateDetector
    from iot_machine_learning.infrastructure.ml.calibration import ConfidenceCalibrator
    
    detector = MultivariateDetector(
        min_series=2,
        enabled=True,
        pca_components=2,
        baseline_percentile=95.0,
        warmup_samples=10,
    )
    
    calibrator = ConfidenceCalibrator(
        temperature=1.5,
        regime_temperatures={
            "VOLATILE": 2.0,
            "STABLE": 1.2,
        }
    )
    
    # Generate simple correlated data
    np.random.seed(123)
    n = 30
    series1 = np.linspace(0, 10, n) + np.random.randn(n) * 0.1
    series2 = series1 * 0.8 + np.random.randn(n) * 0.1
    
    # Collect scores
    scores = []
    for i in range(10, n):
        score = detector.detect(
            values=series1[i-10:i+1].tolist(),
            series_id="s1",
            correlated_series_data={
                "s2": series2[i-10:i+1].tolist(),
            }
        )
        scores.append(score)
    
    # Test calibration with different regimes
    avg_score = np.mean(scores) if scores else 0.5
    
    conf_default = calibrator.calibrate(avg_score, regime=None)
    conf_volatile = calibrator.calibrate(avg_score, regime="VOLATILE")
    conf_stable = calibrator.calibrate(avg_score, regime="STABLE")
    
    print(f"\n📊 Confidence Calibration:")
    print(f"   Raw score: {avg_score:.4f}")
    print(f"   Default (T=1.5): {conf_default:.4f}")
    print(f"   Volatile (T=2.0): {conf_volatile:.4f}")
    print(f"   Stable (T=1.2): {conf_stable:.4f}")
    
    # Assertions
    assert 0.0 <= conf_default <= 1.0
    assert 0.0 <= conf_volatile <= 1.0
    assert 0.0 <= conf_stable <= 1.0
    
    # VOLATILE should be more conservative (lower confidence for same score)
    if avg_score > 0:
        assert conf_volatile < conf_default
        assert conf_stable > conf_default


if __name__ == "__main__":
    # Run tests
    test_multivariate_real_scenario()
    test_multivariate_confidence_integration()
