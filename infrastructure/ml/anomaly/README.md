# infrastructure/ml/anomaly

Detección de anomalías mediante ensemble de sub-detectores con voting ponderado.

## Módulos raíz

| Archivo | Líneas | Responsabilidad |
|---|---|---|
| `voting_anomaly_detector.py` | 241 | `VotingAnomalyDetector` — orquestador del ensemble |
| `detector_factory.py` | 77 | `create_default_detectors()` — factory de los 8 sub-detectores por defecto |
| `vote_context_builder.py` | 109 | `build_vote_context()`, `extract_vel_z()`, `extract_acc_z()` — contexto puro para votos |
| `voting_strategy.py` | 72 | `VotingStrategy` — combina votos en score final |
| `detector_protocol.py` | 159 | `SubDetector` ABC + `DetectorRegistry` + `@register_detector` |
| `anomaly_narrator.py` | 75 | `build_anomaly_explanation()` — narrativa textual |
| `anomaly_config.py` | 56 | `AnomalyDetectorConfig` — configuración centralizada |
| `scoring_functions.py` | 137 | Funciones puras: `compute_z_score`, `compute_iqr_score` |
| `statistical_methods.py` | 37 | Métodos estadísticos auxiliares |
| `training_stats.py` | 52 | `TrainingStats` + `compute_training_stats()` |
| `temporal_stats.py` | 116 | `TemporalTrainingStats` + `compute_temporal_training_stats()` |

## Sub-detectores (`detectors/`)

| Detector | Tipo | Descripción |
|---|---|---|
| `ZScoreDetector` | Magnitud | Z-score sobre distribución histórica |
| `IQRDetector` | Magnitud | Rango intercuartílico |
| `IsolationForestDetector` | Magnitud | Isolation Forest 1D |
| `LOFDetector` | Magnitud | Local Outlier Factor 1D |
| `VelocityZDetector` | Temporal | Z-score de velocidad (Δvalue/Δt) |
| `AccelerationZDetector` | Temporal | Z-score de aceleración (Δvelocity/Δt) |
| `IsolationForestNDDetector` | Temporal | Isolation Forest 3D [value, vel, acc] |
| `LOFNDDetector` | Temporal | LOF 3D [value, vel, acc] |

## Arquitectura

```
VotingAnomalyDetector.detect(window)
  ├── build_vote_context(window, temporal_stats)
  │     ├── temporal_features (si hay datos temporales)
  │     └── nd_features [value, vel, acc] (si window.size >= 3)
  ├── SubDetector[].vote(value, **ctx)  → Dict[str, float]
  ├── VotingStrategy.combine(votes)     → score final
  └── build_anomaly_explanation(votes, z, vel_z, acc_z)
```

## Extensibilidad (DI)

```python
# Inyectar detectores custom
from infrastructure.ml.anomaly import create_default_detectors, VotingAnomalyDetector

custom = create_default_detectors(config) + [MyCustomDetector()]
detector = VotingAnomalyDetector(sub_detectors=custom)

# Registrar detector nuevo
from infrastructure.ml.anomaly import register_detector

@register_detector("my_detector")
def create_my_detector(config):
    return MyCustomDetector(config)
```

## Pesos por defecto

| Detector | Peso |
|---|---|
| IsolationForest | 0.30 |
| Z-Score | 0.20 |
| VelocityZ | 0.15 |
| LOF | 0.15 |
| IQR | 0.10 |
| AccelerationZ | 0.10 |
