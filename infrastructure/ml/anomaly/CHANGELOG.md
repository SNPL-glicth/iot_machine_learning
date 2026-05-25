# Changelog

## v2.0.0 — Clean Architecture Refactor
**Performance:** F1=0.2857, FP=24, Recall=0.2143

### Changed
- scoring/functions.py: pure math only, removed __getattr__ fallback
- voting/strategy.py: clean weighted_vote contract, removed calibrators
- core/config.py: frozen value object, 7 explicit weights
- factory/defaults.py: exactly 7 detectors, no conditionals
- core/detector.py: removed adaptive weights, drift coupling,
  dynamic contamination, and DB persistence

### Fixed
- weighted_vote denominator bug: was dividing by active weights only
  when temporal detectors returned None, inflating scores by 37%
- IsolationForest contamination: was hardcoded to 0.001, now uses
  config value 0.005 (matches real anomaly rate)
- RollingZ hysteresis: counter was tracking but never gating output
- __getattr__ fallback in functions.py: was silently providing wrong
  weighted_vote implementation to all detectors

### Removed
- Adaptive weight recalculation (every 20 outcomes)
- BayesianWeightTracker integration
- EnsembleCalibrator and DetectionRateMeasurer
- DriftCoupling weight override
- Dynamic contamination estimation in train()
- DB persistence (AnomalyWeightsRepository)
- IsolationForestNDDetector and LOFNDDetector from default ensemble

## v1.0.0 — Initial Production Config
**Performance:** F1=0.164, FP=73, Recall=0.167
- rolling_z enabled with weight=0.20
- voting_threshold=0.75
