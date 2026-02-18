"""Feature flags para control de features UTSAE - Facade principal.

Todos los flags tienen valores por defecto SEGUROS (desactivados).
Esto garantiza que el sistema funciona igual que antes si no se
configuran variables de entorno.

Carga desde variables de entorno con prefijo ``ML_``.
Ejemplo:
    export ML_USE_TAYLOR_PREDICTOR=true
    export ML_TAYLOR_ORDER=2
    export ML_TAYLOR_SENSOR_WHITELIST=1,5,42

Decisiones de diseño:
- Pydantic BaseModel para validación automática de tipos.
- ``from_env()`` como factory method (no en __init__) para separar
  la lógica de parsing de env vars de la construcción del objeto.
- ``ML_ROLLBACK_TO_BASELINE`` es el "panic button": si está activo,
  TODO el sistema usa baseline sin importar otros flags.
- Whitelist de sensores como string CSV para facilitar configuración
  vía env vars (los dicts no son prácticos en env vars).

Modular implementation:
    - parsers: Parsing utilities (parse_bool, parse_int_set)
    - flags: FeatureFlags model with validators
    - loader: Environment loader and singleton management
    - feature_flags: Main facade (this file - backward compatibility)
"""

from __future__ import annotations

from .flags import FeatureFlags
from .loader import get_feature_flags, reset_feature_flags

__all__ = [
    "FeatureFlags",
    "get_feature_flags",
    "reset_feature_flags",
]
