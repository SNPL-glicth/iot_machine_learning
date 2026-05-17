"""RolloutDecider — determina si un sensor está en el grupo de tratamiento MoE.

Usa hash determinista de series_id para garantizar que:
- El mismo sensor siempre cae en el mismo grupo.
- El porcentaje de activación es exacto y configurable.

Configuración: env var MOE_ROLLOUT_PERCENT (0-100).
"""

from __future__ import annotations

import hashlib
import os


class RolloutDecider:
    """Decide rollout gradual por sensor_id hash.

    Args:
        percent: Porcentaje de sensores en tratamiento (0-100).
                 Si None, lee de env var MOE_ROLLOUT_PERCENT (default 0).
    """

    def __init__(self, percent: int | None = None) -> None:
        if percent is None:
            try:
                percent = int(os.getenv("MOE_ROLLOUT_PERCENT", "0"))
            except ValueError:
                percent = 0
        self._percent = max(0, min(100, percent))

    @property
    def percent(self) -> int:
        return self._percent

    def is_enabled(self, series_id: str) -> bool:
        """True si este sensor_id cae en el grupo de tratamiento."""
        if self._percent <= 0:
            return False
        if self._percent >= 100:
            return True
        # Hash determinista MD5 → entero 0-99
        h = hashlib.md5(series_id.encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) % 100
        return bucket < self._percent
