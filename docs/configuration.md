## Configuration

### Feature Flags Reference

Flags are loaded via `get_feature_flags()` on every call (hot-reload).
All flags read from environment variables. Safe defaults are conservative
(features off, low resource usage). Set in `.env` or export directly.

#### Core toggles
| Flag | Default | Description |
|------|---------|-------------|
| `ML_ROLLBACK_TO_BASELINE` | `false` | Panic button — force all to baseline |
| `ML_USE_TAYLOR_PREDICTOR` | `false` | Enable Taylor engine |
| `ML_USE_KALMAN_FILTER` | `false` | Enable Kalman filtering |
| `ML_ENABLE_AB_TESTING` | `false` | Compare baseline vs Taylor |
| `ML_ENABLE_PLASTICITY` | `false` | Enable weight learning |
| `ML_ENABLE_ITERATIVE` | `false` | Enable cognitive loop |
| `ML_ENABLE_COGNITIVE_MEMORY` | `false` | Weaviate integration |
| `ML_ENABLE_MEMORY_RECALL` | `false` | Query historical explanations |
| `ML_ENABLE_PREDICTION_CACHE` | `false` | Cache predictions |
| `ML_ENABLE_VOTING_ANOMALY` | `false` | Voting anomaly detection |
| `ML_ENABLE_CHANGE_POINT_DETECTION` | `false` | Change point detection |
| `ML_ENABLE_EXPLAINABILITY` | `false` | Explanation builder |
| `ML_ENABLE_ENSEMBLE_PREDICTOR` | `false` | Model ensemble |
| `ML_ENABLE_DELTA_SPIKE_DETECTION` | `false` | Delta spike detection |
| `ML_ENABLE_REGIME_DETECTION` | `false` | Regime detection |
| `ML_ENABLE_AUDIT_LOGGING` | `false` | Audit logging |

#### Plasticity tuning
| Flag | Default | Description |
|------|---------|-------------|
| `ML_PLASTICITY_ALPHA` | `0.15` | Base EMA smoothing factor |
| `ML_PLASTICITY_MIN_WEIGHT` | `0.05` | Floor to prevent total suppression |
| `ML_PLASTICITY_MAX_REGIMES` | `10` | LRU eviction threshold |
| `ML_PLASTICITY_REGIME_TTL_SECONDS` | `86400.0` | Unused regime decay (1 day) |
| `ML_PLASTICITY_NOISE_THRESHOLD` | `0.3` | Noise penalty activation |
| `ML_PLASTICITY_PERSIST_EVERY_N` | `10` | Batch writes every N updates |
| `ML_PLASTICITY_IMMEDIATE_PERSIST_THRESHOLD` | `0.15` | Accuracy change threshold for immediate persist |
| `ML_PLASTICITY_REDIS_CACHE_TTL_SECONDS` | `60.0` | Redis cache TTL |
| `ML_PLASTICITY_REGIME_ALPHAS` | JSON | Per-regime learning rates |
| `ML_PLASTICITY_LR_FACTORS` | JSON | Per-regime LR multipliers |

#### Decision engine tuning
| Flag | Default | Description |
|------|---------|-------------|
| `ML_ENABLE_DECISION_ENGINE` | `true` | Enable decision engine |
| `ML_DECISION_ENGINE` | `simple` | Engine type: simple/contextual/conservative/aggressive/cost_optimized |
| `ML_DECISION_CONSERVATIVE_THRESHOLD` | `0.8` | Confidence threshold for "intervene" |
| `ML_DECISION_CONSERVATIVE_SAFETY_MARGIN` | `1.2` | Risk multiplier for worst-case analysis |
| `ML_DECISION_CONFIDENCE_FLOOR` | `0.6` | Minimum confidence |
| `ML_DECISION_CONFIDENCE_CEILING` | `0.95` | Maximum confidence |
| `ML_DECISION_ESCALATION_THRESHOLD` | `5` | Consecutive anomalies for escalation |
| `ML_DECISION_ATT_STABLE_DRIFT_THRESHOLD` | `0.10` | Drift threshold for stable attenuator |
| `ML_DECISION_CONFIDENCE_REDUCTION_SPARSE` | `0.9` | Confidence reduction with sparse evidence |
| `ML_DECISION_BASE_SCORES` | JSON | Severity to score mappings |
| `ML_DECISION_AMP_THRESHOLDS` | JSON | Amplifier thresholds |
| `ML_DECISION_AMP_CONSECUTIVE_5` | `1.35` | Multiplier for 5+ consecutive anomalies |
| `ML_DECISION_AMP_CONSECUTIVE_3` | `1.20` | Multiplier for 3+ consecutive anomalies |
| `ML_DECISION_AMP_RATE_HIGH` | `1.20` | Multiplier for high anomaly rate |
| `ML_DECISION_AMP_RATE_MED` | `1.10` | Multiplier for medium anomaly rate |
| `ML_DECISION_AMP_VOLATILE` | `1.15` | Multiplier for volatile regime |
| `ML_DECISION_AMP_NOISY` | `1.10` | Multiplier for noisy regime |
| `ML_DECISION_AMP_DRIFT_HIGH` | `1.20` | Multiplier for high drift |
| `ML_DECISION_AMP_DRIFT_MED` | `1.10` | Multiplier for medium drift |
| `ML_DECISION_ATT_STABLE` | `0.85` | Attenuator for stable regime |
| `ML_DECISION_ATT_LOW_CRITICALITY` | `0.80` | Attenuator for low criticality |
| `ML_DECISION_ATT_NO_CONTEXT` | `0.90` | Attenuator for no recent context |
| `ML_DECISION_SUPPRESSION_WINDOW_MINUTES` | `5.0` | Alert suppression window |
| `ML_DECISION_THRESHOLD_ESCALATE` | `0.85` | Score threshold to escalate |
| `ML_DECISION_THRESHOLD_INVESTIGATE` | `0.65` | Score threshold to investigate |
| `ML_DECISION_THRESHOLD_MONITOR` | `0.40` | Score threshold to monitor |

#### Anomaly tracking
| Flag | Default | Description |
|------|---------|-------------|
| `ML_ANOMALY_TTL_SECONDS` | `7200.0` | Anomaly entry TTL (2 hours) |
| `ML_ANOMALY_MAX_ENTRIES_PER_SERIES` | `500` | Max entries per series |
| `ML_ANOMALY_KEY_TTL_SECONDS` | `3600` | Redis key TTL (1 hour) |
| `ML_ANOMALY_TRACKER_BACKEND` | `memory` | Backend: memory or redis |
| `ML_ANOMALY_VOTING_THRESHOLD` | `0.5` | Voting threshold |
| `ML_ANOMALY_CONTAMINATION` | `0.1` | Contamination factor |

#### Performance & streaming
| Flag | Default | Description |
|------|---------|-------------|
| `ML_BATCH_MAX_WORKERS` | `4` | Parallel batch workers |
| `ML_BATCH_CIRCUIT_BREAKER_THRESHOLD` | `10` | Circuit breaker threshold |
| `ML_STREAM_USE_SLIDING_WINDOW` | `true` | Use in-memory windows |
| `ML_ENTERPRISE_USE_PRELOADED_DATA` | `true` | Reduce DB queries |
| `ML_MQTT_ASYNC_PROCESSING` | `true` | Enable async MQTT processing |
| `ML_MQTT_QUEUE_SIZE` | `1000` | Max queue depth |
| `ML_MQTT_NUM_WORKERS` | `4` | ThreadPool workers |
| `ML_CACHE_TTL_SECONDS` | `60` | Cache TTL |
| `ML_CACHE_MAX_ENTRIES` | `1000` | Max cache entries |
| `ML_SLIDING_WINDOW_MAX_SENSORS` | `1000` | LRU eviction threshold |
| `ML_SLIDING_WINDOW_TTL_SECONDS` | `3600` | TTL eviction (1 hour) |

#### Circuit breaker & infrastructure
| Flag | Default | Description |
|------|---------|-------------|
| `ML_INGEST_CIRCUIT_BREAKER_ENABLED` | `true` | Enable circuit breaker |
| `ML_INGEST_CB_FAILURE_THRESHOLD` | `5` | Failure threshold |
| `ML_INGEST_CB_TIMEOUT_SECONDS` | `30` | Timeout in seconds |
| `ML_PIPELINE_BUDGET_MS` | `500` | Pipeline budget in ms |
| `ML_COHERENCE_CHECK_ENABLED` | `false` | Prediction coherence check |
| `ML_DOMAIN_BOUNDARY_ENABLED` | `false` | Validate input domain |
| `ML_ACTION_GUARD_ENABLED` | `false` | Enable action guards |
| `ML_DECISION_ARBITER_ENABLED` | `false` | Enable decision arbiter |
| `ML_CONFIDENCE_CALIBRATION_ENABLED` | `false` | Confidence calibration |
| `ML_NARRATIVE_UNIFICATION_ENABLED` | `false` | Narrative unification |
| `ML_PREDICT_MAX_WORKERS` | `3` | ThreadPool workers for concurrent engine execution |
| `ML_PREDICT_ENGINE_TIMEOUT_MS` | `400` | Per-engine timeout (ms) within 500ms pipeline budget |
| `ML_HAMPEL_ENABLED` | `true` | Outlier rejection before weighted fusion |
| `ML_HAMPEL_K` | `3.0` | Hampel filter sensitivity (≈3σ Gaussian) |

### JSON Flags (Dict-Type Parameters)

Some flags accept JSON strings to configure dictionaries at runtime:

```bash
# Override all severity→score mappings for the decision engine
export ML_DECISION_BASE_SCORES='{"critical":0.95,"high":0.75,"medium":0.50,"low":0.25,"info":0.05,"warning":0.50}'

# Override amplifier thresholds
export ML_DECISION_AMP_THRESHOLDS='{"count_high":3,"count_medium":2,"ratio_high":0.55,"ratio_low":0.25}'

# Override plasticity regime alphas
export ML_PLASTICITY_REGIME_ALPHAS='{"STABLE":0.08,"TRENDING":0.18,"VOLATILE":0.30,"NOISY":0.05}'
```

JSON flags are parsed with `json.loads()` on each `_get_flags()` call.
Invalid JSON falls back to the hardcoded default silently — validate before deploying.

### Backpressure & Limits

```bash
# MQTT Async Processing
export ML_MQTT_ASYNC_PROCESSING=true           # Enable async processing
export ML_MQTT_QUEUE_SIZE=1000                 # Max queue depth
export ML_MQTT_NUM_WORKERS=4                   # ThreadPool workers

# Sliding Window Limits
export ML_SLIDING_WINDOW_MAX_SENSORS=1000      # LRU eviction threshold
export ML_SLIDING_WINDOW_TTL_SECONDS=3600      # TTL eviction (1 hour)

# Circuit Breaker
export ML_INGEST_CIRCUIT_BREAKER_ENABLED=true
export ML_INGEST_CB_FAILURE_THRESHOLD=5
export ML_INGEST_CB_TIMEOUT_SECONDS=30
```

### Panic Button

```bash
# Force all predictions to baseline engine, bypass cognitive pipeline
export ML_ROLLBACK_TO_BASELINE=true
```

### Architecture Decisions: Why Flags Over Constants

| Decision | Rationale |
|----------|-----------|
| **All numeric thresholds in flags** | Adjustable in <5 min without code change (ISO 27001 A.12.1.2) |
| **JSON flags for dicts** | Compatible with env vars in Kubernetes/docker-compose |
| **`_get_flags()` on every call** | Hot-reload: changes propagate without restart |
| **`RegimeType` enum** | Eliminates magic strings across 10+ files |
| **`RedisKeys` registry** | Single point of change for key patterns; enables access auditing |
| **Fallback to `FeatureFlags()`** | Service stays alive even if config system fails |
