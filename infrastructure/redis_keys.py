"""Redis Key Registry — centralized key pattern management.

ISO 27001 A.12.4.1: All Redis key patterns are centralized and auditable.
DRY: Single source of truth for all Redis keys in the system.

Usage:
    from iot_machine_learning.infrastructure.redis_keys import RedisKeys
    key = RedisKeys.plasticity("VOLATILE")
    key = RedisKeys.last_alert("series_123")
"""

from __future__ import annotations


class RedisKeys:
    """Registro central de todos los key patterns de Redis.

    Modificar aquí afecta a todo el sistema. Auditables.
    """

    # --- Plasticity keys ---
    @staticmethod
    def plasticity(regime: str) -> str:
        """Key for plasticity weights per regime.

        Format: plasticity:{regime}
        Type: Redis Hash (engine_name -> accuracy)
        """
        return f"plasticity:{regime}"

    # --- Error history keys ---
    @staticmethod
    def error_history(series_id: str, engine_name: str) -> str:
        """Key for error history per series and engine.

        Format: error_history:{series_id}:{engine_name}
        Type: Redis List (maxlen configured in flags)
        """
        return f"error_history:{series_id}:{engine_name}"

    # --- Anomaly tracking keys ---
    @staticmethod
    def anomaly_track(series_id: str) -> str:
        """Key for anomaly tracking (sorted set by timestamp).

        Format: anomaly_track:{series_id}
        Type: Redis SortedSet (timestamp -> anomaly_data)
        """
        return f"anomaly_track:{series_id}"

    @staticmethod
    def anomaly_consecutive(series_id: str) -> str:
        """Key for consecutive anomaly counter.

        Format: anomaly_consecutive:{series_id}
        Type: Redis String (INCR/DEL operations)
        """
        return f"anomaly_consecutive:{series_id}"

    # --- Alert suppression keys ---
    @staticmethod
    def last_alert(series_id: str) -> str:
        """Key for last emitted alert per series.

        Format: last_alert:{series_id}
        Type: Redis String with TTL (JSON blob)
        """
        return f"last_alert:{series_id}"

    @staticmethod
    def suppressed(series_id: str) -> str:
        """Key for suppressed alert counter per series.

        Format: suppressed:{series_id}
        Type: Redis String (counter with TTL)
        """
        return f"suppressed:{series_id}"

    # --- Pattern registry for wildcards ---
    @staticmethod
    def pattern_all_error_history(series_id: str) -> str:
        """Pattern for all error history keys of a series.

        For use with KEYS/SCAN operations (not for production queries).
        """
        return f"error_history:{series_id}:*"

    @staticmethod
    def pattern_all_plasticity() -> str:
        """Pattern for all plasticity regime keys.

        For use with KEYS/SCAN operations (not for production queries).
        """
        return "plasticity:*"
