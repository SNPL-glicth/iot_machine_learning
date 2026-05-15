"""Tests for expire_predictions_task."""

from unittest.mock import MagicMock

import pytest

from iot_machine_learning.ml_service.tasks.expire_predictions_task import (
    expire_stale_predictions,
)


class MockRepo:
    def __init__(self, return_value: int = 0):
        self._return_value = return_value
        self.calls: list = []

    def expire_old(self, max_age_seconds: int) -> int:
        self.calls.append({"max_age_seconds": max_age_seconds})
        return self._return_value


def test_expire_calls_repo_expire_old():
    repo = MockRepo(return_value=5)
    expire_stale_predictions(repo, max_age_seconds=3600)
    assert len(repo.calls) == 1
    assert repo.calls[0]["max_age_seconds"] == 3600


def test_expire_returns_count():
    repo = MockRepo(return_value=42)
    result = expire_stale_predictions(repo, max_age_seconds=1800)
    assert result == 42


def test_expire_returns_zero():
    repo = MockRepo(return_value=0)
    result = expire_stale_predictions(repo, max_age_seconds=3600)
    assert result == 0


def test_expire_uses_default_max_age():
    repo = MockRepo(return_value=1)
    expire_stale_predictions(repo)
    assert repo.calls[0]["max_age_seconds"] == 3600


def test_expire_logs_count(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="iot_machine_learning.ml_service.tasks.expire_predictions_task")
    repo = MockRepo(return_value=7)
    expire_stale_predictions(repo, max_age_seconds=3600)
    assert "predictions_expired" in caplog.text
    # Verify extra fields via log records
    records = [r for r in caplog.records if "predictions_expired" in r.message]
    assert len(records) == 1
    assert records[0].count == 7
    assert records[0].max_age_seconds == 3600
