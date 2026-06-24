# infrastructure/persistence

Capa de persistencia del sistema ML: SQL Server, Redis y Weaviate (vector DB).

## Estructura

```
persistence/
├── sql/                          # Queries SQL modularizadas
│   └── storage/
│       ├── connection_manager.py
│       ├── base_queries.py
│       ├── prediction_queries.py
│       ├── anomaly_queries.py
│       ├── plasticity_queries.py
│       ├── performance_queries.py
│       ├── zenin_db_connection.py
│       └── zenin_ml_storage.py
├── redis/                        # Redis persistence layer
│   ├── circuit_breaker.py
│   ├── connection_manager.py
│   ├── pools.py
│   ├── sliding_window_store.py
│   ├── tsdb_adapter.py
│   ├── redis_cache.py
│   └── redis_connection_manager.py
├── vector/                       # Weaviate vector DB
│   ├── schema/
│   │   ├── schema_builder.py
│   │   ├── class_definitions.py
│   │   ├── property_definitions.py
│   │   ├── property_builder.py
│   │   └── migration_runner.py
│   └── cognitive/
│       ├── memory_adapter.py
│       ├── search_adapter.py
│       └── indexing_adapter.py
├── cache.py / cache_decorators.py
├── factory.py
├── sliding_window.py
├── circuit_breaker.py
└── adapters/
    └── analysis_result_adapter.py
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
| `zenin_db_connection.py` | Conexion Zenin DB (config desde .env) |
| `zenin_ml_storage.py` | Almacenamiento ML unificado |

El adaptador principal `infrastructure/adapters/sqlserver_storage.py` orquesta estos módulos.

## redis/

Persistencia de estado en Redis para:
- **Sliding windows** — ventanas deslizantes de lecturas por sensor
- **Plasticity** — almacenamiento de pesos bayesianos aprendidos
- **Circuit breaker** — estado de circuit breakers del sistema
- **Time-series** — datos temporales via Redis TSDB
- **Cache** — cache de predicciones y configuraciones

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
python infrastructure/weaviate/setup.py --host http://localhost:8080
```

## Uso

```python
# SQL
from infrastructure.adapters.sqlserver_storage import SqlServerStorageAdapter
adapter = SqlServerStorageAdapter(engine)
adapter.save_prediction(prediction)

# Redis
from infrastructure.persistence.redis.sliding_window_store import SlidingWindowStore
store = SlidingWindowStore(redis_client)
store.append(sensor_id, value, timestamp)

# Weaviate (cognitivo)
from infrastructure.persistence.vector.cognitive import WeaviateCognitiveAdapter
cognitive = WeaviateCognitiveAdapter(url="http://localhost:8080")
cognitive.store_explanation(explanation)
```

## Tests

```bash
python -m pytest tests/ -k "storage or weaviate" -v
```
