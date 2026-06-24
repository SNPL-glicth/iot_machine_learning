# ml_service/metrics

Métricas de rendimiento, observabilidad y A/B testing para comparar motores de predicción.

## Archivos

| Archivo | Líneas | Responsabilidad |
|---|---|---|
| `ab_testing.py` | 283 | `ABTester` — registro de predicciones y singleton global |
| `ab_metrics.py` | 164 | Funciones puras: `compute_mae`, `compute_rmse`, `determine_winner`, `compute_ab_result`, `aggregate_summary` |
| `performance_metrics.py` | 269 | `MetricsCollector` — métricas de sistema (predicciones, errores, latencia) |
| `observability_metrics.py` | — | Métricas de observabilidad expuestas vía Prometheus |
| `prometheus_exporter.py` | — | Exportador de métricas Prometheus para `/metrics` |

## Diseño

`ab_testing.py` contiene el estado (deques thread-safe por sensor) y delega
todo el cálculo numérico a `ab_metrics.py` (sin estado, sin I/O).

```
ABTester.compute_results()
  └── ab_metrics.compute_ab_result(actual, baseline_preds, taylor_preds)
        ├── compute_mae()
        ├── compute_rmse()
        ├── determine_winner()   ← margen 5%
        └── compute_improvement_pct()

ABTester.get_summary()
  └── ab_metrics.aggregate_summary(results)
```

## Prometheus / Observability

Las métricas del sistema se exportan via `prometheus-client` en el endpoint `/metrics`:
- `ml_predictions_total` — contador de predicciones por sensor
- `ml_prediction_errors_total` — contador de errores
- `ml_prediction_latency_seconds` — histograma de latencia
- `ml_ab_tester_*` — métricas de A/B testing

## Uso

```python
from ml_service.metrics.ab_testing import get_ab_tester

tester = get_ab_tester()
tester.record_prediction(sensor_id=1, actual_value=20.0,
                         baseline_pred=20.1, taylor_pred=19.9)
result = tester.compute_results(sensor_id=1)
# result.winner → "taylor" | "baseline" | "tie"
# result.improvement_pct → porcentaje de mejora de Taylor sobre baseline
```

## Thread-safety

`ABTester` usa `threading.Lock`. Puede usarse desde batch y stream runners en paralelo.
Los deques tienen `maxlen=1000` por sensor (FIFO automático).
