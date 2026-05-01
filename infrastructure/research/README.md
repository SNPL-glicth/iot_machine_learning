# infrastructure/research/

Adaptadores opcionales — no activos en el path crítico de producción.

## Contenido

- `weaviate/` — Cognitive memory adapter (Weaviate).
  Activar con `ML_ENABLE_COGNITIVE_MEMORY=true` + `ML_COGNITIVE_MEMORY_URL=<url>`.
  Default: **desactivado**.
- `cognitive_memory_adapter.py` — Puerto de dominio hacia Weaviate.

## Regla

```python
# ✓ Correcto — importar desde la nueva ubicación
from iot_machine_learning.infrastructure.research.weaviate.weaviate_cognitive import WeaviateCognitiveAdapter

# El shim en infrastructure/adapters/weaviate/ también funciona (emite DeprecationWarning)
```

## Activación

```env
ML_ENABLE_COGNITIVE_MEMORY=true
ML_COGNITIVE_MEMORY_URL=http://weaviate:8080
```
