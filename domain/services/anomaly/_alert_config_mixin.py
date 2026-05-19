"""Config mixin for AlertSuppressor.

Reads hot-reloadable parameters from feature flags.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class _AlertConfigMixin:
    """Mixin that provides hot-reloadable config getters."""

    def _get_key_ttl(self: Any) -> int:
        """Leer TTL de keys desde flags (hot-reload)."""
        try:
            from ....ml_service.config.feature_flags import get_feature_flags
            return int(get_feature_flags().ML_ANOMALY_KEY_TTL_SECONDS)
        except (ImportError, AttributeError, ValueError, TypeError) as e:
            logger.warning(
                "feature_flag_read_failed",
                extra={
                    "flag": "ML_ANOMALY_KEY_TTL_SECONDS",
                    "error_type": type(e).__name__,
                    "fallback": 3600,
                },
            )
            return 3600  # fallback: 1 hora

    def _get_window_minutes(self: Any) -> float:
        """Leer ventana de supresión desde flags (hot-reload)."""
        try:
            from ....ml_service.config.feature_flags import get_feature_flags
            flags = get_feature_flags()
            return flags.ML_DECISION_SUPPRESSION_WINDOW_MINUTES
        except (ImportError, AttributeError, TypeError) as e:
            logger.warning(
                "feature_flag_read_failed",
                extra={
                    "flag": "ML_DECISION_SUPPRESSION_WINDOW_MINUTES",
                    "error_type": type(e).__name__,
                    "fallback": 5.0,
                },
            )
            return 5.0  # fallback seguro

    def _get_escalation_threshold(self: Any) -> int:
        """Leer umbral de escalación desde flags (hot-reload)."""
        try:
            from ....ml_service.config.feature_flags import get_feature_flags
            return int(get_feature_flags().ML_DECISION_ESCALATION_THRESHOLD)
        except (ImportError, AttributeError, ValueError, TypeError) as e:
            logger.warning(
                "feature_flag_read_failed",
                extra={
                    "flag": "ML_DECISION_ESCALATION_THRESHOLD",
                    "error_type": type(e).__name__,
                    "fallback": 5,
                },
            )
            return 5  # fallback
