# infrastructure/persistence

Capa de persistencia del sistema ML: SQL Server y Weaviate (vector DB).

## Estructura

```
persistence/
├── sql/
│   └── storage/              # Queries SQL modularizadas
│       ├── connection_manager.py
│       ├── base_queries.py
│       ├── prediction_queries.py
│       ├── anomaly_queries.py
│       ├── plasticity_queries.py
│       └── performance_queries.py
└── vector/
    ├── schema/               # Schema builder para Weaviate
    │   ├── schema_builder.py
    │   ├── class_definitions.py
    │   ├── property_definitions.py
    │   ├── property_builder.py
    │   └── migration_runner.py
    └── cognitive/            # Adaptadores de memoria cognitiva
        ├── memory_adapter.py
        ├── search_adapter.py
        └── indexing_adapter.py
```

## sql/storage/

| Archivo | Responsabilidad |
|---|---|
| `connection_manager.py` | Pool de conexiones SQL Server |
| `base_queries.py` | Queries base compartidas |
| `prediction_queries.py` | `save_prediction`, `get_latest_prediction`, `load_sensor_window` |
| `anomaly_queries.py` | `save_anomaly_event`, `get_anomaly_history` |
| `plasticity_queries.py` | `record_prediction_error`, `get_contextual_weights` |
| `performance_queries.py` | `get_rolling_performance`, métricas de rendimiento |

El adaptador principal `infrastructure/adapters/sqlserver_storage.py` orquesta estos módulos.

## vector/cognitive/

| Archivo | Responsabilidad |
|---|---|
| `memory_adapter.py` | `WeaviateCognitiveAdapter` — almacena y recupera memorias cognitivas |
| `search_adapter.py` | Búsqueda semántica por similitud |
| `indexing_adapter.py` | Indexación de explicaciones y anomalías |

Controlado por `ML_ENABLE_COGNITIVE_MEMORY` + `ML_COGNITIVE_MEMORY_URL` en `FeatureFlags`.

## vector/schema/

Builder modular para el schema de Weaviate. Ejecutar una sola vez al inicializar:

```bash
python scripts/create_weaviate_schema.py
```

## Uso

```python
# SQL
from infrastructure.adapters.sqlserver_storage import SqlServerStorageAdapter
adapter = SqlServerStorageAdapter(engine)
adapter.save_prediction(prediction)

# Weaviate (cognitivo)
from infrastructure.persistence.vector.cognitive import WeaviateCognitiveAdapter
cognitive = WeaviateCognitiveAdapter(url="http://localhost:8080")
cognitive.store_explanation(explanation)
```

## Tests

```bash
python -m pytest tests/ -k "storage or weaviate" -v
```
