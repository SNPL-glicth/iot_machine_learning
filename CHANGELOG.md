# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-06-23

### Added
- **Pipeline 25+ fases** — Evolucion de 15 a 25+ fases cognitivas con nuevos modulos:
  - ContextPhase, PredictionReadinessGate, DriftResponse, CausalPhase, MemoryPhase, ShadowEvaluation, Observability
- **Anomaly Ensemble v2.0** — Clean architecture refactor (F1: 0.164 → 0.2857, FP: 73 → 24)
- **MoE Package** — `infrastructure/ml/moe/` con gating tree, sparse fusion, expert registry, rollout
- **Inference Module** — `infrastructure/ml/inference/` con MLE, Bayesian (prior/likelihood/posterior), Naive Bayes, Platt scaling
- **Optimization Toolkit** — `infrastructure/ml/optimization/` con SGD, L-BFGS, Newton, genetico, PSO
- **Governance System** — 9 componentes: ParameterRegistry, BoundsEnforcer, DynamicTuner, TemperatureScaler, CorrelationAnalyzer, Decorrelator, Watchdog, RecoveryManager, LoopBoundsMonitor
- **Kalman Engine** — `infrastructure/ml/engines/kalman/` con filtro Kalman CV, Q adaptativo
- **Multivariate Engine** — `infrastructure/ml/engines/multivariate/` con PCA online
- **Seasonal Engine** — `infrastructure/ml/engines/seasonal/` con deteccion FFT de ciclos
- **RUL Estimator** — `infrastructure/ml/anomaly/rul/` para estimacion de vida util residual
- **Cognitive Memory** — Integracion con Weaviate para memoria episodica y semantica
- **Dynamic Features** — Pipeline de features dinamicos: ventanas moviles, derivadas, lags, cross-features
- **Warmup System** — Precarga de modelos y cache al iniciar (`ml_service/warmup.py`)
- **Prometheus Metrics** — Endpoint `/metrics` con metricas de sistema y A/B testing
- **Circuit Breaker** — Redis-backed circuit breaker con backoff exponencial

### Changed
- `CONFIDENCE.MIN_CONFIDENCE`: 0.3 → **0.5** (para datos industriales)
- Default `AnomalyDetectorConfig.voting_threshold`: 0.5 → 0.75 (validado NAB)
- Default `RollingZScoreDetector` parameters: `long_window`: 50 → 400, `hysteresis`: 1 → 7, `z_threshold`: 3.0 → 3.5
- Default `AnomalyDetectorConfig.contamination`: 0.1 → 0.005
- EngineFactory: imports FQN → relativos para evitar duplicacion de clases
- Adapters deprecados eliminados (BaselinePredictionAdapter, TaylorPredictionAdapter)
- Confidence floor unificado via `core/parameters/numerical_constants.py`

### Fixed
- **Duplicate EngineFactory** — imports FQN vs relativos creaban dos clases en memoria
- **`confidence` vs `confidence_score`** — AttributeError silencioso en MoE fusion
- **Doble penalizacion** — MoE + runner aplicaban penalizaciones por separado
- **Anomaly v1.0 adaptativo** — pesos recalculados silenciosamente causaban inestabilidad (eliminado en v2.0)
- **Drift coupling en anomalia** — sobreescribia pesos configurados sin advertencia (eliminado en v2.0)

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
