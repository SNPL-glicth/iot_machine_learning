# Detección de Anomalías — Ensemble Voting v2.0

**Última actualización:** 2026-06-23
**Archivo fuente:** `infrastructure/ml/anomaly/`
**CHANGELOG:** `infrastructure/ml/anomaly/CHANGELOG.md`

---

## Arquitectura (Clean Ensemble v2.0)

```
scoring/functions.py      → pure math, sin estado
scoring/statistical_methods.py → scoring estadístico
scoring/temporal.py       → scoring temporal
scoring/training.py       → preparación de datos de entrenamiento
voting/strategy.py        → combinación de votos, sin lógica
voting/context_builder.py → votación con contexto
core/config.py            → value object frozen
core/protocol.py          → protocolo SubDetector
core/detector.py          → train/detect/is_trained
factory/defaults.py       → 7 detectores, sin condicionales
```

## Los 7 Detectores del Ensemble (v2.0)

### 1. ZScoreDetector

**Qué detecta:** Valores cuya magnitud se desvía significativamente de la media histórica.

**Fórmula:** `z = (x - μ) / σ`

**Voto:** 0 si `|z| < lower` (2.0), 1 si `|z| > upper` (3.0), interpolado linealmente entre ambos.

**Por qué importa:** Detector clásico, interpretable, computacionalmente barato.

---

### 2. RollingZScoreDetector

**Qué detecta:** Desviación Z sobre ventana deslizante con histéresis.

**Configuración v2.0:** `long_window=400`, `short_window=10`, `hysteresis=7`, `z_threshold=3.5`

**Por qué importa:** Detecta deriva gradual que Z-score fijo no captura.

---

### 3. IQRDetector

**Qué detecta:** Outliers robustos usando rango intercuartílico.

**Fórmula:** `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`

**Voto:** 1 si fuera del rango, 0 si dentro.

---

### 4. IsolationForestDetector

**Qué detecta:** Patrones de aislamiento en el espacio de características.

**Configuración:** 100 árboles (`n_estimators=100`), `contamination=0.005`, `random_state=42`.

---

### 5. LOFDetector (Local Outlier Factor)

**Qué detecta:** Outliers por densidad local.

**Configuración:** `max_neighbors=20`, `contamination=0.005`.

---

### 6. VelocityZDetector

**Qué detecta:** Cambios súbitos de velocidad (primera derivada).

**Voto:** Z-score de la velocidad con umbrales `lower=2.5`, `upper=3.0`.

---

### 7. AccelerationZDetector

**Qué detecta:** Cambios de aceleración (segunda derivada) — inflexiones de régimen.

**Voto:** Z-score de la aceleración con umbrales `lower=2.5`, `upper=3.0`.

---

## Pesos del Ensemble v2.0 (Producción)

```python
weights = {
    "isolation_forest": 0.25,   # Detector más general
    "z_score": 0.20,            # Magnitud rápida
    "rolling_z": 0.20,          # Deriva gradual
    "velocity_z": 0.15,         # Cambios de régimen
    "acceleration_z": 0.10,    # Inflexiones
    "iqr": 0.05,                # Robusto a skew
    "local_outlier_factor": 0.05, # Densidad variable
}
```

**Suma:** 1.0 (validado en `AnomalyDetectorConfig.__post_init__`).

## Configuración de Producción v2.0

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `voting_threshold` | 0.75 | Validado en NAB machine temp |
| `z_vote_lower` | 2.5 | Reduce FP en fluctuaciones normales |
| `z_vote_upper` | 3.0 | Saturación 3σ estándar |
| `contamination` | 0.005 | Tasa real de anomalías (0.37%) |
| `rolling_z window` | 150 | Adaptación rápida de régimen |
| `rolling_z hyst` | 3 | Filtra ruido transitorio |

## Rendimiento (NAB machine_temperature_system_failure)

| Versión | F1 | Precision | Recall | FP | Cliff's delta |
|---------|:--:|:---------:|:------:|:--:|:------------:|
| v1.0 | 0.164 | 0.161 | 0.167 | 73 | — |
| v2.0 | **0.2857** | — | 0.2143 | **24** | 0.7261 |

RollingZ: `long_window=50→400`, `hysteresis=1→7`, `z_threshold=3.0→3.5`
Validación: Grid search sobre 243 combinaciones.

## Eliminado en v2.0

- **Pesos adaptativos**: se recalculaban silenciosamente, causaban inestabilidad
- **Drift coupling**: sobreescribía pesos configurados sin advertencia
- **Contaminación dinámica**: fallaba silenciosamente, usaba tasa incorrecta
- **Persistencia DB de pesos**: estado impredecible
- **IsolationForestNDDetector**: sin peso definido, se filtraba al ensemble
- **LOFNDDetector**: mismo problema

## Componentes Adicionales

### RUL Estimator (`rul/`)
- `estimator.py` — Estimación de vida útil residual desde patrones de anomalía
- `models.py` — Modelos de regresión para RUL
- `narrator.py` — Generación de narrativa para explicaciones RUL

### Persistent Detector (`persistent_detector.py`)
- Wrapper con estado Redis-backed para detección stateful entre reinicios

### Multivariate Detector (`detectors/multivariate/`)
- Detección de anomalías multi-sensor por correlación

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

---

## Reglas Críticas (v2.0)

1. `weighted_vote`: detectores ausentes excluidos del numerador Y denominador. Nunca dividir por 1.0 fijo.
2. `compute_z_vote`: nunca compartido entre detectores. Cada detector tiene su propia lógica de voto.
3. Un archivo cambiado = un benchmark run. F1 drop > 0.01 = revertir inmediatamente.
