"""Tests for cognitive/plasticity.py — regime-contextual weight learning."""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.plasticity import (
    PlasticityTracker,
)


class TestPlasticityBasic:

    def test_no_history_uniform_weights(self) -> None:
        pt = PlasticityTracker()
        w = pt.get_weights("stable", ["a", "b"])
        assert w["a"] == pytest.approx(0.5)
        assert w["b"] == pytest.approx(0.5)

    def test_has_history_false_initially(self) -> None:
        pt = PlasticityTracker()
        assert pt.has_history("stable") is False

    def test_update_creates_history(self) -> None:
        pt = PlasticityTracker()
        pt.update("stable", "a", 1.0)
        assert pt.has_history("stable") is True

    def test_low_error_engine_gets_higher_weight(self) -> None:
        pt = PlasticityTracker(alpha=1.0)  # instant learning
        # Engine "a" has low error, "b" has high error
        for _ in range(5):
            pt.update("stable", "a", 0.1)
            pt.update("stable", "b", 10.0)
        w = pt.get_weights("stable", ["a", "b"])
        assert w["a"] > w["b"]

    def test_weights_sum_to_one(self) -> None:
        pt = PlasticityTracker()
        pt.update("stable", "a", 1.0)
        pt.update("stable", "b", 2.0)
        pt.update("stable", "c", 3.0)
        w = pt.get_weights("stable", ["a", "b", "c"])
        assert sum(w.values()) == pytest.approx(1.0)

    def test_regime_isolation(self) -> None:
        """Weights in one regime don't affect another."""
        pt = PlasticityTracker(alpha=1.0)
        pt.update("stable", "a", 0.1)
        pt.update("stable", "b", 10.0)
        pt.update("noisy", "a", 10.0)
        pt.update("noisy", "b", 0.1)

        w_stable = pt.get_weights("stable", ["a", "b"])
        w_noisy = pt.get_weights("noisy", ["a", "b"])

        assert w_stable["a"] > w_stable["b"]
        assert w_noisy["b"] > w_noisy["a"]


class TestPlasticityReset:

    def test_reset_all(self) -> None:
        pt = PlasticityTracker()
        pt.update("stable", "a", 1.0)
        pt.update("noisy", "b", 2.0)
        pt.reset()
        assert pt.has_history("stable") is False
        assert pt.has_history("noisy") is False

    def test_reset_single_regime(self) -> None:
        pt = PlasticityTracker()
        pt.update("stable", "a", 1.0)
        pt.update("noisy", "b", 2.0)
        pt.reset("stable")
        assert pt.has_history("stable") is False
        assert pt.has_history("noisy") is True


class TestPlasticityEMA:

    def test_ema_smoothing(self) -> None:
        """With alpha < 1, old observations still influence weights."""
        pt = PlasticityTracker(alpha=0.3)
        # First: "a" is great
        for _ in range(10):
            pt.update("stable", "a", 0.1)
            pt.update("stable", "b", 5.0)
        w1 = pt.get_weights("stable", ["a", "b"])
        assert w1["a"] > w1["b"]

        # Now "b" becomes great, but EMA means "a" still has residual
        for _ in range(3):
            pt.update("stable", "a", 5.0)
            pt.update("stable", "b", 0.1)
        w2 = pt.get_weights("stable", ["a", "b"])
        # "a" should still have some weight from history
        assert w2["a"] > 0.05
