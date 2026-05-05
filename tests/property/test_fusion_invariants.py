"""Property-based tests for WeightedFusion and related components.

Validates invariants:
- weights sum to 1
- no negative probabilities
- stable outputs under noise

No external frameworks (hypothesis) — uses randomized loops with fixed seeds.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.fusion.engine_selector import (
    WeightedFusion,
)
from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception, InhibitionState,
)
from iot_machine_learning.infrastructure.ml.cognitive.fusion.contextual_weight_calculator import (
    compute_inverse_mae_weights,
    resolve_contextual_weights,
)


class TestWeightedFusionInvariants:
    """Property: fused weights always sum to 1.0 (within fp tolerance)."""

    def _random_perceptions(
        self,
        rng: random.Random,
        n_engines: int,
    ) -> List[EnginePerception]:
        trends = ["up", "down", "stable"]
        return [
            EnginePerception(
                engine_name=f"engine_{i}",
                predicted_value=rng.gauss(20.0, 5.0),
                confidence=rng.uniform(0.0, 1.0),
                trend=rng.choice(trends),
            )
            for i in range(n_engines)
        ]

    def _random_inhibitions(
        self,
        rng: random.Random,
        names: List[str],
    ) -> List[InhibitionState]:
        return [
            InhibitionState(
                engine_name=name,
                base_weight=1.0 / len(names),
                inhibited_weight=max(0.0, rng.gauss(1.0 / len(names), 0.05)),
                suppression_factor=rng.uniform(0.0, 0.5),
            )
            for name in names
        ]

    @pytest.mark.parametrize("seed", range(5))
    def test_weights_sum_to_one(self, seed: int) -> None:
        rng = random.Random(seed)
        fusion = WeightedFusion()
        for _ in range(500):
            n = rng.randint(1, 8)
            perceptions = self._random_perceptions(rng, n)
            inhibitions = self._random_inhibitions(rng, [p.engine_name for p in perceptions])
            result = fusion.fuse(perceptions, inhibitions)
            weights: Dict[str, float] = result[3]
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"weights_sum={total} seed={seed}"

    @pytest.mark.parametrize("seed", range(5))
    def test_no_negative_probabilities(self, seed: int) -> None:
        rng = random.Random(seed)
        fusion = WeightedFusion()
        for _ in range(500):
            n = rng.randint(1, 8)
            perceptions = self._random_perceptions(rng, n)
            inhibitions = self._random_inhibitions(rng, [p.engine_name for p in perceptions])
            result = fusion.fuse(perceptions, inhibitions)
            _, confidence, _, weights, _, _ = result
            assert confidence >= 0.0, f"negative confidence={confidence}"
            assert confidence <= 1.0, f"confidence > 1: {confidence}"
            for w in weights.values():
                assert w >= -1e-9, f"negative weight={w}"

    @pytest.mark.parametrize("seed", range(5))
    def test_stable_output_under_noise(self, seed: int) -> None:
        """Property: small perturbations in predicted_value produce small output changes."""
        rng = random.Random(seed)
        fusion = WeightedFusion()
        base_perceptions = [
            EnginePerception("e1", 20.0, 0.8, "up"),
            EnginePerception("e2", 21.0, 0.7, "up"),
            EnginePerception("e3", 19.5, 0.6, "stable"),
        ]
        inhibitions = [
            InhibitionState("e1", 0.33, 0.33, 0.0),
            InhibitionState("e2", 0.33, 0.33, 0.0),
            InhibitionState("e3", 0.34, 0.34, 0.0),
        ]
        base = fusion.fuse(base_perceptions, inhibitions)
        base_value = base[0]

        for _ in range(200):
            noisy = [
                EnginePerception(
                    p.engine_name,
                    p.predicted_value + rng.gauss(0, 0.1),
                    max(0.0, min(1.0, p.confidence + rng.gauss(0, 0.02))),
                    p.trend,
                )
                for p in base_perceptions
            ]
            result = fusion.fuse(noisy, inhibitions)
            assert abs(result[0] - base_value) < 2.0, (
                f"output unstable: base={base_value} noisy={result[0]}"
            )

    def test_empty_perceptions_returns_safe_defaults(self) -> None:
        fusion = WeightedFusion()
        result = fusion.fuse([], [])
        value, confidence, trend, weights, selected, reason = result
        assert value == 0.0
        assert confidence == 0.0
        assert trend == "stable"
        assert weights == {}
        assert selected == "none"
        assert reason == "no_engines"


class TestContextualWeightCalculatorInvariants:
    """Property: inverse MAE weights sum to 1 and are non-negative."""

    @pytest.mark.parametrize("seed", range(5))
    def test_inverse_mae_weights_sum_to_one(self, seed: int) -> None:
        rng = random.Random(seed)
        for _ in range(500):
            n = rng.randint(1, 6)
            maes = {f"e{i}": rng.uniform(0.01, 5.0) for i in range(n)}
            weights = compute_inverse_mae_weights(maes, epsilon=0.1)
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"sum={total}"
            for w in weights.values():
                assert w >= -1e-9

    def test_all_none_returns_none(self) -> None:
        result = resolve_contextual_weights(
            {"a": None, "b": None}, ["a", "b"], epsilon=0.1
        )
        assert result is None

    def test_partial_none_returns_none(self) -> None:
        result = resolve_contextual_weights(
            {"a": 0.5, "b": None}, ["a", "b"], epsilon=0.1
        )
        assert result is None
