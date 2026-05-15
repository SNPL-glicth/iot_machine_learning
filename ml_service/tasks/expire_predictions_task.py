"""Periodic task for expiring stale pending predictions."""

from __future__ import annotations

import logging

from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.prediction_verification_repository import (
    PredictionVerificationRepository,
)

logger = logging.getLogger(__name__)


def expire_stale_predictions(
    repo: PredictionVerificationRepository,
    max_age_seconds: int = 3600,
) -> int:
    """Expire pending predictions older than max_age_seconds.

    Args:
        repo: Repository for prediction verifications.
        max_age_seconds: Maximum age in seconds before a pending prediction
            is considered stale and marked as expired.

    Returns:
        Number of predictions that were expired.
    """
    expired = repo.expire_old(max_age_seconds)
    logger.info(
        "predictions_expired",
        extra={
            "count": expired,
            "max_age_seconds": max_age_seconds,
        },
    )
    return expired
