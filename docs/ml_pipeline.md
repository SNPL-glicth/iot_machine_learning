# Pipeline Cognitivo ML — Referencia Técnica

**Última actualización:** 2026-05-12
**Archivo fuente:** `infrastructure/ml/cognitive/orchestration/pipeline_executor.py`
**Fases:** 15 (índices 0–14) + AssemblyPhase final

---

## Índice de Fases

| Índice | Fase | Input del Contexto | Output al Contexto | Flag que la controla |
|--------|------|-------------------|---------------------|-------------------|
| 0 | SanitizePhase | `values`, `timestamps` | `sanitization_flags`, `is_fallback` | `ML_ENABLE_SANITIZE` (implícito) |
| 1 | BoundaryCheckPhase | `values` | `is_fallback`, `fallback_reason` | `ML_DOMAIN_BOUNDARY_ENABLED` |
| 2 | SeasonalDecompositionPhase | `values`, `timestamps` | `seasonal_adjusted_values` | `ML_ENABLE_SEASONALITY` |
| 3 | PerceivePhase | `values`, `timestamps` | `regime`, `noise_ratio`, `stability`, `neighbor_trends` | `ML_ENABLE_REGIME_DETECTION` |
| 4 | DriftDetectionPhase | `regime`, `noise_ratio`, `stability` | `drift_detected`, `drift_magnitude`, `drift_reset_regime` | `ML_ENABLE_DRIFT_DETECTION` |
| 5 | PredictPhase | `values`, `timestamps`, `regime` | `perceptions`, `engine_failures` | `ML_ENABLE_PREDICTION_CACHE` (cache) |
| 6 | AdaptPhase | `perceptions`, `regime` | `resolved_weights`, `inhibition_states` | `ML_ENABLE_PLASTICITY` |
| 7 | InhibitPhase | `perceptions`, `inhibition_states` | `inhibited_perceptions` | `ML_ENABLE_PLASTICITY` |
| 8 | FusePhase | `inhibited_perceptions`, `resolved_weights` | `fused_value`, `confidence`, `trend`, `fusion_flags`, `hampel_diagnostic` | `ML_HAMPEL_ENABLED` |
| 9 | DecisionArbiterPhase | `fused_value`, `confidence`, `anomaly_score` | `decision`, `decision_context` | `ML_DECISION_ARBITER_ENABLED` |
| 10 | CoherenceCheckPhase | `decision`, `neighbor_trends` | `coherence_verdict` | `ML_COHERENCE_CHECK_ENABLED` |
| 11 | ConfidenceCalibrationPhase | `confidence`, `regime` | `calibrated_confidence` | `ML_CONFIDENCE_CALIBRATION_ENABLED` |
| 12 | ExplainPhase | `perceptions`, `fusion_flags`, `decision` | `explanation`, `narrative` | `ML_ENABLE_EXPLAINABILITY` |
| 13 | ActionGuardPhase | `decision`, `coherence_verdict` | `guarded_action`, `action_reason` | `ML_ACTION_GUARD_ENABLED` |
| 14 | NarrativeUnificationPhase | `explanation`, `narrative`, `guarded_action` | `unified_narrative` | `ML_NARRATIVE_UNIFICATION_ENABLED` |
| — | AssemblyPhase | Todo el contexto | `PredictionResult`, `ComplianceRecord` | `ML_COMPLIANCE_EXPORT_PATH` |

---

## Descripción por Fase

### [0] SanitizePhase

**Input:** `values: List[float]`, `timestamps: Optional[List[float]]`

**Procesamiento:**
- NaN/Inf → hard-stop fallback (`nan_or_inf_rejected`).
- Clamp a ±6σ usando ventana local o Redis-backed `SeriesValuesStore`.
- CUSUM two-sided (k=0.5σ, h=4σ) detecta rampas graduales.

**Output:** `sanitization_flags: List[str]` (ej. `["cusum_ramp_up"]`)

**Flag:** Siempre ejecuta; no hay flag de desactivación.

**Early termination:** Si NaN/Inf, retorna `PredictionResult(predicted_value=None, confidence=0.0, trend="unknown", metadata={"is_sanitize_fallback": True})`.

---

### [1] BoundaryCheckPhase

**Input:** `values`

**Procesamiento:** Valida que los valores estén dentro de los límites de dominio configurados para la serie.

**Output:** `is_fallback: bool`, `fallback_reason: "out_of_domain"`

**Flag:** `ML_DOMAIN_BOUNDARY_ENABLED` (default `false`).

---

### [2] SeasonalDecompositionPhase

**Input:** `values`, `timestamps`

**Procesamiento:**
- FFT por defecto (periodo 24h = `ML_SEASONAL_PERIOD_DEFAULT`).
- STL opcional si `ML_SEASONAL_USE_STL=true` (requiere statsmodels).
- Requiere mínimo 48 puntos (`ML_SEASONAL_MIN_POINTS`).

**Output:** `seasonal_adjusted_values: List[float]`

**Flag:** `ML_ENABLE_SEASONALITY` (default `false`).

---

### [3] PerceivePhase

**Input:** `values`, `timestamps`

**Procesamiento:**
- `SignalAnalyzer.analyze()` extrae: regime (`STABLE`, `TRENDING`, `VOLATILE`, `NOISY`, `TRANSITIONAL`), noise_ratio, stability, z_score.
- Correlation enrichment: consulta `CorrelationPort` para tendencias de vecinos correlacionados (max 3).

**Output:** `regime: str`, `noise_ratio: float`, `stability: float`, `neighbor_trends: Dict[str, float]`

**Flag:** `ML_ENABLE_REGIME_DETECTION` (default `false`).

---

### [4] DriftDetectionPhase

**Input:** `regime`, `noise_ratio`, `stability`

**Procesamiento:**
- Page-Hinkley (default): δ=0.005, λ=50, α=0.9999.
- ADWIN (opcional): δ=0.002, max_window=1000.
- Cooldown 300s entre resets por serie.

**Al detectar drift:**
1. Reset de `BayesianWeightTracker` para el régimen afectado.
2. Emite condición indicador ISO 13374 (`DRIFT_MAGNITUDE`).
3. Log de auditoría vía `AuditPort`.

**Output:** `drift_detected: bool`, `drift_magnitude: float`, `drift_reset_regime: Optional[str]`

**Flag:** `ML_ENABLE_DRIFT_DETECTION` (default `true`).

---

### [5] PredictPhase

**Input:** `values`, `timestamps`, `regime`

**Procesamiento:**
- Recolecta percepciones de todos los motores capaces (`can_handle(n_points)`).
- Ejecución concurrente con `ThreadPoolExecutor` (max workers = `ML_PREDICT_MAX_WORKERS`, default 3).
- Timeout por motor = `ML_PREDICT_ENGINE_TIMEOUT_MS` (default 400ms).
- Fallback a secuencial si el executor falla.

**Output:** `perceptions: List[EnginePerception]`, `engine_failures: List[Dict]`

**Flag:** `ML_ENABLE_PREDICTION_CACHE` habilita cacheo de predicciones recientes.

---

### [6] AdaptPhase

**Input:** `perceptions`, `regime`

**Procesamiento:**
- `WeightResolutionService` resuelve pesos:
  1. Pesos adaptativos basados en MAE (si `BayesianWeightTracker` tiene datos).
  2. Plasticidad bayesiana por régimen.
  3. Pesos base como fallback.

**Output:** `resolved_weights: Dict[str, float]`, `inhibition_states: Dict[str, InhibitionState]`

**Flag:** `ML_ENABLE_PLASTICITY` (default `false`).

---

### [7] InhibitPhase

**Input:** `perceptions`, `inhibition_states`

**Procesamiento:**
- `InhibitionGate.compute()` suprime motores con error reciente alto.
- Estados de inhibición se propagan a `FusionPhase`.

**Output:** `inhibited_perceptions: List[EnginePerception]`

**Flag:** `ML_ENABLE_PLASTICITY` (default `false`).

---

### [8] FusePhase

**Input:** `inhibited_perceptions`, `resolved_weights`

**Procesamiento:**
- Hampel filter (k=3.0 × 1.4826 × MAD) sobre `predicted_value` de percepciones.
- Si `<3` percepciones o MAD=0, bypass (no filtra).
- `WeightedFusion.fuse()` genera consenso ponderado.

**Output:** `fused_value: float`, `confidence: float`, `trend: str`, `fusion_flags: List[str]`, `hampel_diagnostic: Optional[Dict]`

**Flag:** `ML_HAMPEL_ENABLED` (default `true`), `ML_HAMPEL_K` (default `3.0`).

---

### [9] DecisionArbiterPhase

**Input:** `fused_value`, `confidence`, `anomaly_score`

**Procesamiento:**
- Arbitra entre decisiones de múltiples estrategias si están registradas.
- Selecciona la decisión con mayor confianza coherente.

**Output:** `decision: Decision`, `decision_context: DecisionContext`

**Flag:** `ML_DECISION_ARBITER_ENABLED` (default `false`).

---

### [10] CoherenceCheckPhase

**Input:** `decision`, `neighbor_trends`

**Procesamiento:** Valida que la decisión sea coherente con tendencias de sensores correlacionados.

**Output:** `coherence_verdict: str` (`"coherent"`, `"inconclusive"`, `"contradictory"`)

**Flag:** `ML_COHERENCE_CHECK_ENABLED` (default `false`).

---

### [11] ConfidenceCalibrationPhase

**Input:** `confidence`, `regime`

**Procesamiento:**
- Calibra confianza con temperatura por régimen:
  - STABLE: temp=1.2 (más confiado)
  - VOLATILE: temp=2.0 (más conservador)
  - NOISY: temp=1.8
  - TRENDING: temp=1.5

**Output:** `calibrated_confidence: float`

**Flag:** `ML_CONFIDENCE_CALIBRATION_ENABLED` (default `false`).

---

### [12] ExplainPhase

**Input:** `perceptions`, `fusion_flags`, `decision`

**Procesamiento:**
- `ExplanationRenderer` genera explicación estructurada.
- `CausalNarrativeBuilder` mapea indicadores técnicos a narrativas humanas.

**Output:** `explanation: Dict`, `narrative: List[str]`

**Flag:** `ML_ENABLE_EXPLAINABILITY` (default `false`).

---

### [13] ActionGuardPhase

**Input:** `decision`, `coherence_verdict`

**Procesamiento:**
- Aplica guardrails: `AUTO` (ejecuta), `ASK` (requiere confirmación), `DENY` (bloquea).
- Severidad CRITICAL puede forzar AUTO independientemente del guardrail general.

**Output:** `guarded_action: str`, `action_reason: str`

**Flag:** `ML_ACTION_GUARD_ENABLED` (default `false`).

---

### [14] NarrativeUnificationPhase

**Input:** `explanation`, `narrative`, `guarded_action`

**Procesamiento:** Unifica narrativa técnica y de acción en un solo bloque legible.

**Output:** `unified_narrative: str`

**Flag:** `ML_NARRATIVE_UNIFICATION_ENABLED` (default `false`).

---

## VotingAnomalyDetector

### Sub-detectores (8 por defecto, 9 con multivariate)

| # | Detector | Peso default | Qué detecta | Por qué importa |
|---|----------|-------------|-------------|-----------------|
| 1 | ZScoreDetector | 0.20 | Spikes de magnitud (>2σ o >3σ) | Clásico, interpretable, rápido |
| 2 | IQRDetector | 0.10 | Outliers robustos (Q1–1.5×IQR, Q3+1.5×IQR) | Inmune a distribuciones asimétricas |
| 3 | IsolationForestDetector | 0.30 | Patrones de aislamiento en feature space | Capta anomalías multidimensionales sutiles |
| 4 | LOFDetector | 0.15 | Outliers por densidad local | Detecta clusters con densidad heterogénea |
| 5 | VelocityZDetector | 0.15 | Cambios súbitos de velocidad (1ª derivada) | Detecta rampas invisibles para detectores de magnitud |
| 6 | AccelerationZDetector | 0.10 | Cambios de aceleración (2ª derivada) | Detecta inflexiones de régimen antes de que el valor explote |
| 7 | IsolationForestNDDetector | — | Isolation Forest n-dimensional (no pesado en config base) | Extensión multivariada nativa |
| 8 | LOFNDDetector | — | LOF n-dimensional (no pesado en config base) | Extensión multivariada nativa |
| 9 | MultivariateDetector | — | PCA-based (si `enable_multivariate=true`) | Detecta correlaciones rotas entre series |

**Pesos adaptativos:** Cada detector mantiene un historial de 50 outcomes. Cuando un detector acierta frecuentemente, su peso en el ensemble aumenta; cuando falla, disminuye. Los pesos se normalizan para sumar 1.0.

**RobustScaler:** Antes del entrenamiento, los valores se normalizan con `RobustScaler` (mediana + IQR) en vez de `StandardScaler` (media + σ). Esto evita que outliers en el set de entrenamiento distorsionen la escala.

---

## BayesianWeightTracker

### Prior Gaussiano con Varianza Empírica por Motor

**Configuración:**
- `sigma2_obs_default = 1.0` (fallback sin muestras suficientes)
- `sigma2_obs_min = 0.01` (evita varianza cero)
- `variance_window = 20` (errores recientes por motor)
- `variance_min_samples = 5` (mínimo para estimar varianza empírica)

**Update conjugado normal-normal:**

```
Prior:     N(μ₀, σ²₀)
Likelihood: N(μ, σ²_obs)   donde σ²_obs = max(0.01, var(errores_recientes))
Posterior:  N(μₙ, σ²ₙ)

μₙ  = (σ²₀·μ₀ + σ²_obs·n·x̄) / (n·σ²₀ + σ²_obs)
σ²ₙ = (σ²₀·σ²_obs) / (n·σ²₀ + σ²_obs)
```

**¿Por qué varianza empírica?** Un motor de temperatura (0–1000°C) tiene errores de escala 10× mayor que uno de vibración (0–1g). Usar `σ²_obs=1.0` para ambos distorsiona el posterior. ZENIN estima `σ²_obs` online por motor.

**Logging estructurado:**

```json
{
  "event": "sigma2_obs_estimated",
  "engine": "taylor_polynomial",
  "sigma2_obs": 12.34,
  "source": "empirical",
  "n_samples": 20
}
```

### Regimes y LRU Eviction

- Máximo 10 régimes (`max_regimes`).
- TTL 86400s (`regime_ttl_seconds`) — régimen no accedido decae a pesos uniformes.
- Coldest régimen (menor `last_access`) se elimina cuando se excede el límite.

### Reset por Drift Confirmado

Cuando `DriftDetectionPhase` detecta drift:
1. `reset_regime(regime, series_id, drift_severity)` borra priors y accuracies del régimen.
2. El tracker re-aprende desde priors uniformes.
3. Log de auditoría ISO 27001 vía `AuditPort`.

---

## ContextualDecisionEngine

### Amplificadores

| Condición | Multiplier default | Descripción |
|-----------|-------------------|-------------|
| `consecutive_anomalies >= 5` | ×1.35 | Patrón persistente — riesgo elevado |
| `consecutive_anomalies >= 3` | ×1.20 | Patrón emergente |
| `recent_anomaly_rate > 0.60` | ×1.20 | Tasa de anomalías alta |
| `recent_anomaly_rate > 0.30` | ×1.10 | Tasa moderada |
| `regime == "VOLATILE"` | ×1.15 | Régimen inestable |
| `regime == "NOISY"` | ×1.10 | Ruido elevado |
| `drift_score > 0.70` | ×1.20 | Drift severo |
| `drift_score > 0.40` | ×1.10 | Drift moderado |

### Atenuadores

| Condición | Multiplier default | Descripción |
|-----------|-------------------|-------------|
| `regime == "STABLE"` y `drift < 0.10` | ×0.85 | Señal confiable — reducir alarma |
| `criticality == "LOW"` | ×0.80 | Sensor no crítico |
| `recent_anomaly_count == 0` | ×0.90 | Primera anomalía aislada — no exagerar |

### Mapeo Score → Acción

| Score final | Acción | Prioridad | Razón típica |
|-------------|--------|-----------|--------------|
| ≥ 0.85 | `ESCALATE` | 1 | Patrón persistente detectado |
| ≥ 0.65 | `INVESTIGATE` | 2 | Anomalía contextual confirmada |
| ≥ 0.40 | `MONITOR` | 3 | Anomalía moderada |
| < 0.40 | `LOG_ONLY` | 5 | Señal débil o aislada |

### Desconexión conocida con ThresholdPolicy

`ThresholdPolicy` clasifica severidad en 4 niveles (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`). `ContextualDecisionEngine` mapea severidad a score base, pero luego aplica amplificadores/atenuadores que pueden mover una `MEDIUM` a `ESCALATE` o una `HIGH` a `MONITOR`. Esta desconexión está documentada como deuda técnica: no hay unificación automática entre los umbrales de `ThresholdPolicy` y los de `DecisionEngine`.
