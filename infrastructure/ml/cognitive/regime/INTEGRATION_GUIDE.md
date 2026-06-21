# Regime Detection Integration Guide

This guide explains how to integrate the new Operational Regime Detection system into ZENIN/UTSAE.

## Quick Start

### 1. Enable Regime Detection in ML Pipeline

```python
from infrastructure.ml.cognitive.regime import RegimeDetectionFactory

# Create regime detection pipeline (minimal for quick wins)
regime_pipeline = RegimeDetectionFactory.create_minimal_pipeline()

# Train classifier with historical data
features_matrix = [...]  # List of [derivative, rolling_std, ...]
timestamps = [...]  # List of corresponding timestamps
regime_pipeline._classifier.train(features_matrix, timestamps)
```

### 2. Configure per Sensor Type

```python
# Register sensor-specific configurations
registry = RegimeDetectionFactory.create_registry_with_presets()

# Override for specific sensor
registry.register_sensor_config(12345, RegimeConfig.for_pressure())
```

### 3. Detect Regime for Sensor Reading

```python
# After computing DynamicFeatures
regime_classification = regime_pipeline.detect_regime(
    sensor_id=sensor_id,
    sensor_type="TEMPERATURE",
    dynamic_features=dynamic_features,
    current_value=current_value,
    current_timestamp=timestamp,
)

# Result includes:
# - regime: "STABLE_NORMAL", "VOLATILE_PEAK", etc.
# - confidence: 0.0 to 1.0
# - previous_regime: Previous regime
# - transition_duration: Duration of transition
```

### 4. Pass Regime to Anomaly Detectors

```python
# When calling anomaly detectors, pass regime
vote = detector.vote(
    value=current_value,
    dynamic_features=dynamic_features,
    regime=regime_classification.regime,  # NEW: Pass regime
    # ... other kwargs
)
```

### 5. Use Contextual Anomaly Router

```python
from infrastructure.ml.cognitive.regime import ContextualAnomalyRouter

# Create router with regime-specific thresholds
router = RegimeDetectionFactory.create_router_with_thresholds()

# Route anomaly with regime context
result = router.route_anomaly(
    sensor_id=sensor_id,
    current_value=current_value,
    regime=regime_classification.regime,
    base_anomaly_score=base_score,
)

# Result includes:
# - contextual_score: Adjusted anomaly score
# - is_anomalous: Whether anomalous in context
# - threshold_used: Threshold used for decision
```

## Architecture Overview

```
DynamicFeatures (derivada, rolling_std, etc.)
    ↓
RegimeDetectionPipeline.detect_regime()
    ↓
OperationalRegimeClassifier.classify() (K-Means/GMM)
    ↓
RegimeStateManager.smooth_transition()
    ↓
RegimeClassification (régimen actual + contexto)
    ↓
Anomaly Detectors (use regime for contextual thresholds)
    ↓
ContextualAnomalyRouter (adjust scores by regime)
    ↓
ContextualAnomalyResult (contextual anomaly decision)
```

## Feature Flags

The system supports gradual rollout via feature flags:

```python
# Phase 1: Disabled by default
ML_ENABLE_REGIME_DETECTION = False

# Phase 2: Enable for specific sensors
ML_ENABLE_REGIME_DETECTION = True
ML_REGIME_DETECTION_WHITELIST = [12345, 67890]

# Phase 3: Enable for all sensors
ML_ENABLE_REGIME_DETECTION = True
ML_REGIME_DETECTION_WHITELIST = []
```

## Backward Compatibility

- **MLFeatures v2.0.0:** With regime detection (regime_classification field)
- **MLFeatures v2.1.0:** With contextual anomaly scoring
- **Engines:** Ignore regime if not supported
- **Detectors:** Fallback to global thresholds if regime not available

## Memory Usage

Estimated memory usage per 1000 sensors:
- Regime state history: ~0.5 MB (100 states × 8 bytes)
- Classifier model: ~1-5 MB (depends on algorithm)
- Total overhead: <10 MB

## Performance Impact

Expected additional latency: <30ms per regime detection
Expected CPU overhead: <5%

## Monitoring

Add these metrics to monitor regime detection:

```python
# Prometheus metrics
regime_detection_total{sensor_type, regime}
regime_detection_latency_seconds{sensor_type}
regime_transition_total{from_regime, to_regime}
regime_duration_seconds{regime}
regime_confidence_score{regime}
```

## Troubleshooting

### Regime not detected
- Check if classifier is trained (`classifier.is_trained`)
- Check if DynamicFeatures are available
- Check if feature vector is valid (not None)

### Frequent regime transitions (flickering)
- Increase `min_regime_duration` in RegimeConfig
- Check if `smooth_transition` is being called
- Verify RegimeStateManager is being used

### High memory usage
- Reduce `max_history_size` in RegimeStateManager
- Reduce number of regimes (n_components)
- Enable cache cleanup

### Slow performance
- Use K-Means instead of GMM/HMM
- Reduce feature vector size
- Disable smooth transitions (if acceptable)

## Next Steps

1. **Phase 2A (Quick Wins):** Enable K-Means + simple contextual thresholds
2. **Phase 2B:** Enable GMM + full contextual routing
3. **Phase 2C:** Enable HDBSCAN/HMM (if needed)

See the architectural plan for detailed implementation phases.
