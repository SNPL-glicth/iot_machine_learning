# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-21

### Added
- **v1.0 Production Configuration** - Validated anomaly detection configuration
  - RollingZScoreDetector: long_window=400, short_window=10, hysteresis=7, z_threshold=3.5
  - Ensemble threshold: 0.75
  - RollingZ weight: 0.20
  - Validation results on NAB machine_temperature_system_failure:
    - F1: 0.2222 (≥ 0.22 ✓)
    - FP: 36 (≤ 40 ✓)
    - Cliff's delta: 0.7261 (≥ 0.70 ✓)
  - Grid search performed over 243 hyperparameter combinations
  - Results saved to: benchmarks/results/grid_search_v2_real.csv
  - Validation saved to: benchmarks/results/validation_v1_production.csv

### Changed
- Default `AnomalyDetectorConfig.voting_threshold`: 0.5 → 0.75
- Default `RollingZScoreDetector` parameters:
  - `short_window`: 10 → 10 (no change)
  - `long_window`: 50 → 400
  - `hysteresis`: 1 → 7
  - `lower`: STAT_THRESHOLDS.Z_SCORE_LOWER → 3.5
  - `upper`: STAT_THRESHOLDS.Z_SCORE_UPPER → 3.5
- Default `AnomalyDetectorConfig.weights['rolling_z']`: 0.20 (no change, validated)

### Benchmark Scripts
- `benchmarks/rolling_z_grid_search_v2_real.py` - Grid search with real detector pipeline
- `benchmarks/validate_v1_production.py` - Validation script for v1.0 config

## [Unreleased]

### Added
- (Future changes will be documented here)
