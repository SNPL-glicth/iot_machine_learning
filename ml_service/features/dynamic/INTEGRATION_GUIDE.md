# Dynamic Features Integration Guide

This guide explains how to integrate the new Dynamic Feature Engineering system into ZENIN/UTSAE.

## Quick Start

### 1. Enable Dynamic Features in MLFeaturesProducer

```python
from ml_service.features.dynamic import DynamicFeatureFactory
from infrastructure.ml.cognitive.dynamic.feature_metadata_registry import FeatureMetadataRegistry

# Create dynamic pipeline (minimal for quick wins)
dynamic_pipeline = DynamicFeatureFactory.create_minimal_pipeline()

# Create feature registry with presets
registry = DynamicFeatureFactory.create_registry_with_presets()

# Initialize FeatureComputer with dynamic pipeline
from ml_service.features.services.feature_computer import FeatureComputer

feature_computer = FeatureComputer(
    registry=existing_registry,  # Your existing registry
    dynamic_pipeline=dynamic_pipeline,  # NEW: Add dynamic pipeline
)
```

### 2. Configure per Sensor Type

```python
# Register sensor-specific configurations
registry = FeatureMetadataRegistry()

# Temperature sensors: enable momentum
registry.register_type_config("TEMPERATURE", FeatureConfig.for_temperature())

# Pressure sensors: enable gradient
registry.register_type_config("PRESSURE", FeatureConfig.for_pressure())

# Vibration sensors: shorter windows
registry.register_type_config("VIBRATION", FeatureConfig.for_vibration())

# Override for specific sensor
registry.register_sensor_config(12345, FeatureConfig.minimal())
```

### 3. Compute Features with Dynamic Pipeline

```python
# When computing features, pass dynamic config
dynamic_config = registry.get_config(sensor_id, sensor_type)

features = feature_computer.compute_features(
    window=sensor_window,
    current_value=current_value,
    current_timestamp=timestamp,
    sensor_type=sensor_type,
    dynamic_config=dynamic_config,  # NEW: Pass dynamic config
)
```

### 4. Pass Dynamic Features to Anomaly Detectors

```python
# When calling anomaly detectors, pass dynamic_features
if features.dynamic_features is not None:
    vote = detector.vote(
        value=current_value,
        dynamic_features=features.dynamic_features,  # NEW: Pass dynamic features
        # ... other kwargs
    )
```

## Architecture Overview

```
MLFeaturesProducer
    ↓
FeatureComputer (with DynamicFeaturePipeline)
    ↓
DynamicFeaturePipeline
    ├─ RollingWindowEngine (1h, 6h, 24h windows)
    ├─ DerivativeCalculator (Δ/Δt, d²/dt²)
    ├─ LagFeatureGenerator (t-1, t-6, t-24)
    └─ CrossFeatureGenerator (sensor correlations)
    ↓
DynamicFeatures (dataclass)
    ↓
MLFeatures.dynamic_features (optional)
    ↓
Anomaly Detectors (use DynamicFeatures if available)
    ├─ VelocityZDetector (uses derivative)
    ├─ AccelerationZDetector (uses second_derivative)
    └─ IsolationForestNDDetector (uses multi-dimensional features)
```

## Feature Flags

The system supports gradual rollout via feature flags:

```python
# Phase 1: Disabled by default
ML_ENABLE_DYNAMIC_FEATURES = False

# Phase 2: Enable for specific sensors
ML_ENABLE_DYNAMIC_FEATURES = True
ML_DYNAMIC_FEATURES_WHITELIST = [12345, 67890]

# Phase 3: Enable for all sensors
ML_ENABLE_DYNAMIC_FEATURES = True
ML_DYNAMIC_FEATURES_WHITELIST = []
```

## Backward Compatibility

- **MLFeatures v1.0.0**: Without DynamicFeatures (dynamic_features=None, model_version="1.0.0")
- **MLFeatures v2.0.0**: With DynamicFeatures (dynamic_features populated, model_version="2.0.0")
- **Engines**: Ignore DynamicFeatures if not supported
- **Detectors**: Fallback to temporal_features if DynamicFeatures not available

## Memory Usage

Estimated memory usage per 1000 sensors:
- Rolling windows: ~2.4 MB (3 windows × 100 points × 8 bytes)
- Total overhead: <5 MB

## Performance Impact

Expected additional latency: <50ms per prediction
Expected CPU overhead: <10%

## Monitoring

Add these metrics to monitor dynamic feature computation:

```python
# Prometheus metrics
dynamic_features_computed_total{sensor_type}
dynamic_features_latency_seconds{sensor_type}
dynamic_features_memory_usage_bytes
dynamic_features_cache_hits_total
dynamic_features_cache_misses_total
```

## Troubleshooting

### Dynamic features not computed
- Check if `dynamic_pipeline` is passed to FeatureComputer
- Check if `dynamic_config` is passed to `compute_features`
- Check logs for "Failed to compute dynamic features" warnings

### High memory usage
- Reduce `rolling_windows` in FeatureConfig
- Reduce `max_size` in RollingWindowEngine
- Enable persistence to Redis for recovery

### Slow performance
- Reduce `smoothing_window` in DerivativeCalculator
- Reduce `rolling_windows` in FeatureConfig
- Disable lag features or cross-features

## Next Steps

1. **Phase 1A (Quick Wins)**: Enable derivatives + 1h rolling stats
2. **Phase 1B**: Add lag features + 6h rolling stats
3. **Phase 1C**: Add cross-features + 24h rolling stats

See the architectural plan for detailed implementation phases.
