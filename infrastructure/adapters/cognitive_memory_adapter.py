"""Backward-compatibility shim.

cognitive_memory_adapter.py movido a:
    infrastructure.research.cognitive_memory_adapter

No añadir código nuevo aquí.
"""

import warnings

warnings.warn(
    "infrastructure.adapters.cognitive_memory_adapter está deprecado. "
    "Importar desde infrastructure.research.cognitive_memory_adapter",
    DeprecationWarning,
    stacklevel=2,
)

from iot_machine_learning.infrastructure.research.cognitive_memory_adapter import *  # noqa: F401,F403,E402
