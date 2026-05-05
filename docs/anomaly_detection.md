# Detección de Anomalías — Ensemble Voting

**Última actualización:** 2026-05-04
**Archivo fuente:** `infrastructure/ml/anomaly/core/detector.py`, `infrastructure/ml/anomaly/factory/defaults.py`

---

## Los 8+ Detectores del Ensemble

### 1. ZScoreDetector

**Qué detecta:** Valores cuya magnitud se desvía significativamente de la media histórica.

**Fórmula:** `z = (x - μ) / σ`

**Voto:** 0 si `|z| < lower` (2.0), 1 si `|z| > upper` (3.0), interpolado linealmente entre ambos.

**Por qué importa:** Detector clásico, interpretable, computacionalmente barato. Funciona bien para spikes de magnitud en señales gaussianas.

---

### 2. IQRDetector

**Qué detecta:** Outliers robustos usando rango intercuartílico.

**Fórmula:** `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`

**Voto:** 1 si fuera del rango, 0 si dentro.

**Por qué importa:** Inmune a distribuciones asimétricas y colas pesadas donde Z-score falla (ej. proceso industrial con arranques/paradas que generan skew).

---

### 3. IsolationForestDetector

**Qué detecta:** Patrones de aislamiento en el espacio de características (unidimensional por defecto).

**Configuración:** 100 árboles (`n_estimators=100`), `contamination=0.1`, `random_state=42`.

**Voto:** Normalizado a [0,1] basado en profundidad media de aislamiento.

**Por qué importa:** No asume distribución. Detecta anomalías sutiles que no son spikes simples (ej. patrones de vibración anómalos en una ventana).

---

### 4. LOFDetector (Local Outlier Factor)

**Qué detecta:** Outliers por densidad local en el espacio de características.

**Configuración:** `max_neighbors=20`, `contamination=0.1`.

**Voto:** Score LOF normalizado.

**Por qué importa:** Detecta puntos que son anómalos respecto a su vecindario, no respecto a la distribución global. Útil cuando la densidad de la señal varía (ej. día vs noche).

---

### 5. VelocityZDetector

**Qué detecta:** Cambios súbitos de velocidad (primera derivada).

**Fórmula:** `velocity = (x_t - x_{t-1}) / Δt`

**Voto:** Z-score de la velocidad con umbrales `lower=2.0`, `upper=3.0`.

**Por qué importa:** Un sensor puede tener valor normal pero estar cambiando demasiado rápido. Esto anticipa fallas antes de que el valor cruce umbrales de magnitud. Ej: temperatura de motor que sube 5°C/min en vez de 1°C/min.

---

### 6. AccelerationZDetector

**Qué detecta:** Cambios de aceleración (segunda derivada) — inflexiones de régimen.

**Fórmula:** `acceleration = (v_t - v_{t-1}) / Δt`

**Voto:** Z-score de la aceleración con umbrales `lower=2.0`, `upper=3.0`.

**Por qué importa:** Detecta cuando un proceso pasa de acelerar a desacelerar (o viceversa). Esto es crítico en IoT industrial donde cambios de régimen (arranque, parada, carga variable) preceden a fallas mecánicas.

---

### 7. IsolationForestNDDetector

**Qué detecta:** Extensión multivariada de Isolation Forest.

**Configuración:** Igual que IsolationForestDetector pero con `min_training_points=50`.

**Voto:** Score normalizado.

**Por qué importa:** Detecta anomalías que solo son visibles cuando se consideran múltiples dimensiones (ej. temperatura + presión + vibración juntas).

---

### 8. LOFNDDetector

**Qué detecta:** Extensión multivariada de LOF.

**Configuración:** `max_neighbors=20`, `contamination=0.1`, `min_training_points=50`.

**Voto:** Score LOF multivariado normalizado.

---

### 9. MultivariateDetector (opcional)

**Qué detecta:** Correlaciones rotas entre series usando PCA incremental.

**Configuración:** `pca_components=2`, `baseline_percentile=95.0`, `warmup_samples=30`.

**Requisito:** `ML_ENABLE_MULTIVARIATE=true` + mínimo 3 series correlacionadas.

**Por qué importa:** Univariados no ven cuando dos sensores correlacionados dejan de moverse juntos (ej. temperatura de entrada y salida de un intercambiador de calor).

---

## Pesos del Ensemble (Default)

```python
weights = {
    "isolation_forest": 0.30,   # Mayor peso — detector más general
    "z_score": 0.20,            # Segundo — rápido e interpretable
    "local_outlier_factor": 0.15, # Tercero — bueno en densidad variable
    "velocity_z": 0.15,         # Detecta cambios de régimen
    "acceleration_z": 0.10,    # Detecta inflexiones
    "iqr": 0.10,                # Robusto a skew
}
```

**Suma:** 1.0 (validado en `AnomalyDetectorConfig.__post_init__`).

---

## Pesos Adaptativos

Cada detector mantiene un `deque(maxlen=50)` de outcomes booleanos (True = acertó, False = falló). Cada 10 outcomes (`_outcome_count % 10 == 0`), se recalculan pesos:

```python
accuracy = sum(outcomes) / len(outcomes)
new_weight = base_weight * (0.5 + accuracy)  # boost si accuracy > 0.5
```

Los pesos se normalizan para sumar 1.0.

---

## Seasonal Decomposition

### FFT (por defecto)

- Periodo: 24h (`ML_SEASONAL_PERIOD_DEFAULT=24`).
- Requiere: mínimo 48 puntos (`ML_SEASONAL_MIN_POINTS=48`).
- Componentes extraídos: tendencia + estacionalidad + residual.
- El residual se pasa a los detectores de anomalías.

### STL (opcional)

- Requiere: `statsmodels` + `ML_SEASONAL_USE_STL=true`.
- Ventaja: mejor manejo de cambios de tendencia no lineales.
- Desventaja: más lento y dependencia externa.

---

## ThresholdPolicy — Niveles de Severidad

| Severidad | Score mínimo (default) | Acción típica |
|-----------|------------------------|---------------|
| CRITICAL | 0.85 | AUTO — intervención inmediata |
| HIGH | 0.70 | ASK — requiere confirmación |
| MEDIUM | 0.45 | MONITOR — alerta operador |
| LOW | 0.25 | LOG_ONLY — registro silencioso |

**Nota:** Los valores exactos dependen de la configuración del `ThresholdPolicy` instanciado. Ver `domain/policies/threshold_policy.py` para la implementación concreta.

---

## Desconexión Conocida: ThresholdPolicy vs ContextualDecisionEngine

`ThresholdPolicy` clasifica severidad del anomaly score en 4 niveles. `ContextualDecisionEngine` toma esa severidad como input, pero luego aplica 8 amplificadores y 3 atenuadores que pueden elevar o reducir la acción final. No existe unificación automática entre los umbrales de `ThresholdPolicy` y los de `ContextualDecisionEngine`.

**Impacto:** Un operador que ajusta `ThresholdPolicy` para ser más permisivo puede ver que `ContextualDecisionEngine` sigue escalando porque los amplificadores (consecutive_anomalies, drift) no dependen de los umbrales de `ThresholdPolicy`.

**Workaround actual:** Ajustar ambos sistemas manualmente. **Deuda técnica:** unificar en un solo sistema de umbrales configurables.
