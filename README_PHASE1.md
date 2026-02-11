# UTSAE Phase 1 — Guía de Operación

## Resumen

Phase 1 implementa el **core matemático** de UTSAE sin modificar el comportamiento actual del sistema. Todo está controlado por **feature flags** y puede desactivarse instantáneamente.

### Componentes nuevos

| Componente | Archivo | Descripción |
|------------|---------|-------------|
| `TaylorPredictionEngine` | `ml/core/taylor_predictor.py` | Predicción por Series de Taylor (diferencias finitas) |
| `KalmanSignalFilter` | `ml/core/kalman_filter.py` | Filtro de Kalman 1D con auto-calibración de R |
| `EngineFactory` | `ml/core/engine_factory.py` | Factory + registro dinámico de motores |
| `FeatureFlags` | `ml_service/config/feature_flags.py` | Control de features vía env vars |
| `ABTester` | `ml_service/metrics/ab_testing.py` | Comparación A/B baseline vs Taylor |

### Qué NO se modificó

- ❌ Schemas de BD (ninguna tabla tocada)
- ❌ Contratos de API (`PredictResponse` sin cambios)
- ❌ Lógica de umbrales/eventos (intacta)
- ❌ Dependencias nuevas (solo numpy, ya presente)
- ❌ Comportamiento por defecto (flags=false → sistema idéntico)

---

## Cómo activar Feature Flags

### Variables de entorno

```bash
# PANIC BUTTON — Fuerza baseline para TODO (rollback instantáneo)
export ML_ROLLBACK_TO_BASELINE=false

# Activar Taylor
export ML_USE_TAYLOR_PREDICTOR=true
export ML_TAYLOR_ORDER=2          # 1=velocidad, 2=aceleración, 3=jerk
export ML_TAYLOR_HORIZON=1        # Pasos adelante

# Activar Kalman (pre-filtro)
export ML_USE_KALMAN_FILTER=true
export ML_KALMAN_Q=0.00001        # Varianza del proceso
export ML_KALMAN_WARMUP_SIZE=10   # Lecturas para calibración

# A/B Testing
export ML_ENABLE_AB_TESTING=true

# Whitelist de sensores (CSV) — vacío = todos
export ML_TAYLOR_SENSOR_WHITELIST=1,5,42

# Motor por defecto
export ML_DEFAULT_ENGINE=baseline_moving_average
```

### Activación gradual recomendada

1. **Día 1:** Activar solo A/B testing (sin cambiar motor)
   ```bash
   export ML_ENABLE_AB_TESTING=true
   ```

2. **Día 2-3:** Activar Taylor para 2-3 sensores piloto
   ```bash
   export ML_USE_TAYLOR_PREDICTOR=true
   export ML_TAYLOR_SENSOR_WHITELIST=1,5
   ```

3. **Semana 2:** Si métricas son buenas, ampliar whitelist
   ```bash
   export ML_TAYLOR_SENSOR_WHITELIST=1,5,10,15,20
   ```

4. **Semana 3:** Si todo OK, abrir para todos
   ```bash
   export ML_TAYLOR_SENSOR_WHITELIST=
   ```

---

## Cómo interpretar resultados de A/B

### Endpoint de monitoreo

```
GET /ml/ab-status
```

### Respuesta ejemplo

```json
{
  "status": "active",
  "total_sensors": 5,
  "taylor_wins": 3,
  "baseline_wins": 1,
  "ties": 1,
  "avg_taylor_mae": 0.234,
  "avg_baseline_mae": 0.456,
  "avg_improvement_pct": 12.5,
  "total_samples": 500
}
```

### Interpretación

| Campo | Significado |
|-------|-------------|
| `taylor_wins` | Sensores donde Taylor tiene MAE ≥5% menor |
| `baseline_wins` | Sensores donde baseline tiene MAE ≥5% menor |
| `ties` | Diferencia < 5% (estadísticamente insignificante) |
| `avg_improvement_pct` | % promedio de mejora de Taylor. Positivo = Taylor mejor |
| `total_samples` | Muestras totales. Mínimo 100 para confianza alta |

### Criterios de decisión

- **Taylor gana en ≥80% de sensores** → Avanzar a Fase 2
- **Taylor empata o pierde** → Ajustar `ML_TAYLOR_ORDER`, revisar datos
- **Taylor diverge (MAE >> baseline)** → Activar panic button

---

## Rollback Plan

### Rollback instantáneo (< 1 segundo)

```bash
export ML_ROLLBACK_TO_BASELINE=true
```

Esto fuerza baseline para **todos** los sensores sin importar otros flags.

### Rollback por sensor

```bash
# Desactivar Taylor solo para sensor 42
# (requiere reiniciar o recargar config)
export ML_TAYLOR_SENSOR_WHITELIST=1,5  # Excluir 42
```

### Rollback completo

```bash
export ML_USE_TAYLOR_PREDICTOR=false
export ML_USE_KALMAN_FILTER=false
export ML_ENABLE_AB_TESTING=false
export ML_ROLLBACK_TO_BASELINE=false
```

El sistema vuelve al comportamiento exacto pre-UTSAE.

---

## Ejecutar Tests

```bash
# Todos los tests
python -m pytest tests/ -v

# Solo Taylor
python -m pytest tests/unit/test_taylor_predictor.py -v

# Solo Kalman
python -m pytest tests/unit/test_kalman_filter.py -v

# Solo A/B integration
python -m pytest tests/integration/test_ab_comparison.py -v

# Con coverage
python -m pytest tests/ --cov=iot_machine_learning.ml.core --cov-report=term-missing
```

---

## Arquitectura de Taylor (diferencias finitas)

```
f(t+h) ≈ f(t) + f'(t)·h + f''(t)·h²/2! + f'''(t)·h³/3!

Donde (diferencias finitas hacia atrás):
  f'(t)   = (f(t) - f(t-1)) / Δt              → velocidad
  f''(t)  = (f(t) - 2f(t-1) + f(t-2)) / Δt²   → aceleración
  f'''(t) = (f(t) - 3f(t-1) + 3f(t-2) - f(t-3)) / Δt³  → jerk
```

**Por qué diferencias finitas y NO polyfit:**
- Polyfit es regresión polinomial (minimiza error cuadrático global)
- Taylor es expansión local (derivadas en el punto actual)
- Para predicción a corto plazo, Taylor captura mejor la dinámica local
- Las derivadas tienen significado físico (velocidad, aceleración)

---

## Arquitectura de Kalman (auto-calibración)

```
Warmup (primeras N lecturas):
  → Acumular valores
  → R = var(warmup_values)  ← auto-calibración
  → x_hat = mean(warmup_values)

Post-warmup:
  x_pred = x_hat
  P_pred = P + Q
  K = P_pred / (P_pred + R)
  x_hat = x_pred + K · (measurement - x_pred)
  P = (1 - K) · P_pred
```

**Por qué auto-calibración de R:**
- R (varianza de medición) depende del sensor específico
- Un sensor de temperatura tiene R diferente a uno de humedad
- Auto-calibrar desde warmup evita configuración manual por sensor

funcion de costo y regresion lineal , minizar error cuadratico  , gradiente de desenso 