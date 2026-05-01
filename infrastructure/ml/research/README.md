# research/

Este directorio contiene código experimental **no usado en producción**.

No importar desde el pipeline de producción.

## Contenido

- `neural/` — Módulos de redes neuronales (SNN, attention, competition, plasticity).
  Nunca conectados al pipeline de inferencia principal.
  Preservados para cuando haya datos de entrenamiento suficientes.

## Regla

```
# ✗ Incorrecto — no hacer esto desde código de producción
from infrastructure.ml.research.neural.attention import AttentionContextCollector

# ✓ Correcto — usar try/except con fallback si se necesita
try:
    from infrastructure.ml.research.neural.attention import AttentionContextCollector
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
```
