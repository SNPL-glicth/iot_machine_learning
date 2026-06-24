# Anomaly Detection Ensemble — ZENIN v2.0

## Architecture
Clean ensemble following SOLID principles.
5 core components, each with one responsibility:

- `scoring/functions.py`      → pure math only, no state
- `scoring/statistical_methods.py` → statistical scoring methods
- `scoring/temporal.py`       → temporal anomaly scoring
- `scoring/training.py`       → training data preparation
- `voting/strategy.py`        → vote combination, no logic
- `voting/context_builder.py` → context-aware voting
- `core/config.py`            → value object, no methods
- `core/protocol.py`          → SubDetector protocol
- `factory/defaults.py`       → 7 detectors, no conditionals
- `core/detector.py`          → train/detect/is_trained only

## Detectors (7 active)
| Detector             | Weight | Purpose                    |
|----------------------|--------|----------------------------|
| isolation_forest     | 0.25   | Global outlier detection   |
| z_score              | 0.20   | Magnitude deviation        |
| rolling_z            | 0.20   | Gradual drift detection    |
| velocity_z           | 0.15   | Rate of change anomalies   |
| acceleration_z       | 0.10   | Second derivative spikes   |
| iqr                  | 0.05   | Robust range detection     |
| local_outlier_factor | 0.05   | Density-based detection    |

## Production Config (v2.0)
| Parameter          | Value | Reason                              |
|--------------------|-------|-------------------------------------|
| voting_threshold   | 0.75  | Validated on NAB machine temp       |
| z_vote_lower       | 2.5   | Reduces FP on normal fluctuations   |
| z_vote_upper       | 3.0   | Standard 3σ saturation              |
| contamination      | 0.005 | Matches real anomaly rate (0.37%)   |
| rolling_z window   | 150   | Fast regime adaptation              |
| rolling_z hyst     | 3     | Filters transient noise             |

## Performance (NAB machine_temperature_system_failure)
| Version   | F1     | Precision | Recall | FP  | Cliff's delta |
|-----------|--------|-----------|--------|-----|:------------:|
| v1.0      | 0.164  | 0.161     | 0.167  | 73  | — |
| v2.0      | 0.2857 | —         | 0.2143 | 24  | 0.7261 |

RollingZ config v1.0 → v2.0: `long_window=50→400`, `hysteresis=1→7`, `z_threshold=3.0→3.5`
Validacion: Grid search sobre 243 combinaciones de hiperparametros.

## CHANGELOG
Ver `CHANGELOG.md` en este directorio para el historial completo de versiones.

## Critical Rules (never break these)
1. `weighted_vote`: absent detectors excluded from numerator
   AND denominator. Never divide by fixed 1.0.
2. `compute_z_vote`: never shared between detectors.
   Each detector owns its own vote logic.
3. One file changed = one benchmark run.
   F1 drop > 0.01 = immediate revert.

## What was removed in v2.0 (and why)
- Adaptive weights: recalculated silently, caused instability
- Drift coupling: overwrote config weights without warning
- Dynamic contamination: failed silently, used wrong rate
- DB persistence for weights: added unpredictable state
- IsolationForestNDDetector: no weight defined, bled into ensemble
- LOFNDDetector: same issue as above

## Additional Components
### RUL Estimator (`rul/`)
- `estimator.py` — Residual Useful Life estimation from anomaly patterns
- `models.py` — Regression models for RUL
- `narrator.py` — Narrative generation for RUL explanations

### Persistent Detector (`persistent_detector.py`)
- Wraps VotingAnomalyDetector with Redis-backed state persistence
- Enables stateful anomaly detection across service restarts

### Multivariate Detector (`detectors/multivariate/`)
- Multi-sensor correlation-based anomaly detection
- Complementa el detector univariado estándar
