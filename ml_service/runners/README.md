# ml_service/runners

Runners de ejecución ML: batch (SQL) y stream (broker en memoria).

## Archivos raíz

| Archivo | Líneas | Responsabilidad |
|---|---|---|
| `ml_batch_runner.py` | 234 | Runner batch interno (dev/test). **No es el runner de producción** — ver `iot_ingest_services/jobs/ml_batch_runner.py` |
| `ml_stream_runner.py` | 260 | `SimpleMlOnlineProcessor` — procesa lecturas online vía `ReadingBroker` |
| `prediction_deviation_checker.py` | 130 | `check_prediction_deviation()` — detecta desviación predicción vs real |

## Subdirectorios

### `common/` — Componentes compartidos batch + stream

| Archivo | Responsabilidad |
|---|---|
| `event_writer.py` | `EventWriter` — lógica de negocio para emitir/resolver eventos ML |
| `event_queries.py` | Queries SQL puras: `query_upsert_event`, `query_resolve_event`, `query_active_threshold` |
| `sensor_processor.py` | `SensorProcessor` — orquesta predicción + anomalía + severidad + narrativa |
| `prediction_writer.py` | `PredictionWriter` — escritura legacy (dev/test) |
| `severity_classifier.py` | `SeverityClassifier` — clasifica severidad por umbrales de usuario |
| `model_manager.py` | `ModelManager` — carga y cachea modelos ML desde BD |
| `prediction_narrator.py` | Genera narrativa textual de predicciones |
| `regression_prediction_service.py` | Servicio de predicción por regresión |

### `models/` — Modelos de datos del runner online

| Archivo | Responsabilidad |
|---|---|
| `sensor_state.py` | `SensorState` — estado actual del sensor (severidad, patrón) |
| `online_analysis.py` | `OnlineAnalysis` — resultado del análisis de ventanas |

### `services/` — Servicios del runner online

| Archivo | Responsabilidad |
|---|---|
| `window_analyzer.py` | `WindowAnalyzer` — analiza estadísticas de ventanas 1s/5s/10s |
| `threshold_validator.py` | `ThresholdValidator` — valida estado operacional y rangos |
| `explanation_builder.py` | `ExplanationBuilder` — construye explicación de severidad |
| `event_persister.py` | `MLEventPersister` — persiste eventos ML en BD |

### `monitoring/` — Auditoría y métricas

| Archivo | Responsabilidad |
|---|---|
| `batch_audit.py` | Auditoría del ciclo batch |
| `ab_metrics.py` | Métricas A/B (re-export desde `ml_service/metrics/`) |

### `wiring/` — Contenedor de dependencias

| Archivo | Responsabilidad |
|---|---|
| `container.py` | `BatchEnterpriseContainer` — wiring de adapters enterprise |

### `adapters/` y `bridge_config/` — Bridges legacy

Adapters de compatibilidad para el runner batch legacy.

## Pipelines

### Batch
```
ml_batch_runner → sensores activos → SensorProcessor
  → predicción (Taylor/Baseline) + anomalía + severidad
  → dbo.predictions + dbo.ml_events
```

### Stream (online)
```
ReadingBroker → SimpleMlOnlineProcessor.handle_reading()
  → MLFeaturesProducer (siempre)
  → ThresholdValidator (estado operacional)
  → SlidingWindowBuffer → WindowAnalyzer
  → ExplanationBuilder → MLEventPersister
  → check_prediction_deviation (autovalidación)
```
