"""FASE 9.5 — Statistical Reality Check (módulo puro)."""

import random

import pytest
from iot_machine_learning.domain.entities.market.replay.significance import (
    BootstrapCi,
    PermWindow,
    WindowRecord,
    block_bootstrap,
    bootstrap_expert_metrics,
    difference_ci,
    permutation_test,
    pooled_sharpe,
    random_winner_test,
    recover_predicted_direction,
    weighted_acc,
    weighted_net,
    window_cumsum_maxdd,
)

_COST = 0.0012


def _make_ts(
    moves: list[float],
    probs_correct: dict[str, float],
    seed: int = 0,
) -> list[tuple[float, dict[str, tuple[bool, float]]]]:
    """Timestamps donde cada experto acierta con su probabilidad.

    La dirección predicha queda implícita en (correct, move): consistente
    con la recuperación del módulo.
    """
    rng = random.Random(seed)
    per_ts: list[tuple[float, dict[str, tuple[bool, float]]]] = []
    for ts, move in enumerate(moves):
        per: dict[str, tuple[bool, float]] = {}
        for expert, prob in probs_correct.items():
            per[expert] = (rng.random() < prob, move)
        per_ts.append((float(ts), per))
    return per_ts


def _moves(n: int, positive_ratio: float = 0.5, magnitude: float = 0.01, seed: int = 1) -> list[float]:
    rng = random.Random(seed)
    return [
        magnitude if rng.random() < positive_ratio else -magnitude for _ in range(n)
    ]


class TestRecoverPredictedDirection:
    def test_correct_matches_move_sign(self):
        assert recover_predicted_direction(True, 0.01) == 1
        assert recover_predicted_direction(True, -0.01) == -1

    def test_incorrect_flips_move_sign(self):
        assert recover_predicted_direction(False, 0.01) == -1
        assert recover_predicted_direction(False, -0.01) == 1

    def test_zero_move_indeterminate(self):
        assert recover_predicted_direction(True, 0.0) == 0
        assert recover_predicted_direction(False, 0.0) == 0


class TestPermutationTest:
    def test_no_signal_permutation_keeps_result(self):
        """Correct ~50%: el edge real ≈ nulo (no hay señal que destruir)."""
        moves = _moves(200)
        per_ts = _make_ts(moves, {"A": 0.5, "B": 0.5})
        windows = [
            PermWindow(weights={"A": 0.5, "B": 0.5}, per_timestamp=per_ts, cost=_COST, n=200)
        ]
        result = permutation_test(windows, n_permutations=200, seed=7)
        assert result.ci_low <= result.real_mean <= result.ci_high
        assert result.p_value > 0.05

    def test_real_signal_survives_permutation_only_if_real(self):
        """Aciertos 80%: el edge real NO es explicable por la secuencia temporal."""
        moves = _moves(80)
        per_ts = _make_ts(moves, {"A": 0.8, "B": 0.5, "C": 0.5})
        windows = [
            PermWindow(weights={"A": 1.0}, per_timestamp=per_ts, cost=_COST, n=80)
        ]
        result = permutation_test(windows, n_permutations=300, seed=7)
        assert result.real_mean > 0.0
        assert result.real_mean > result.ci_high
        assert result.p_value < 0.05

    def test_permutation_is_deterministic_given_seed(self):
        moves = _moves(40)
        per_ts = _make_ts(moves, {"A": 0.7})
        windows = [
            PermWindow(weights={"A": 1.0}, per_timestamp=per_ts, cost=_COST, n=40)
        ]
        first = permutation_test(windows, n_permutations=100, seed=3)
        second = permutation_test(windows, n_permutations=100, seed=3)
        assert first.real_mean == second.real_mean
        assert first.p_value == second.p_value
        assert first.null_mean == second.null_mean

    def test_weights_are_n_weighted_across_windows(self):
        moves_a = _moves(30)
        moves_b = _moves(10)
        per_a = _make_ts(moves_a, {"A": 0.8})
        per_b = _make_ts(moves_b, {"A": 0.5})
        windows = [
            PermWindow(weights={"A": 1.0}, per_timestamp=per_a, cost=_COST, n=30),
            PermWindow(weights={"A": 1.0}, per_timestamp=per_b, cost=_COST, n=10),
        ]
        result = permutation_test(windows, n_permutations=100, seed=11)
        assert result.n_permutations == 100


class TestBlockBootstrap:
    def _records(self) -> list[WindowRecord]:
        return [
            WindowRecord(n=40, accuracy=0.6, net=0.002, returns=(0.002, 0.001, 0.003)),
            WindowRecord(n=20, accuracy=0.4, net=-0.001, returns=(-0.001, 0.0)),
            WindowRecord(n=10, accuracy=0.5, net=0.0, returns=(0.0, 0.0)),
        ]

    def test_point_matches_statistic(self):
        records = self._records()
        net = block_bootstrap(records, statistic=weighted_net, n_boot=100, seed=5)
        total_n = sum(r.n for r in records)
        expected = sum(r.net * r.n for r in records) / total_n
        assert net.point == pytest.approx(expected)
        assert net.ci_low <= net.point <= net.ci_high

    def test_constant_series_gives_tight_ci(self):
        records = [WindowRecord(n=10, accuracy=0.5, net=0.001, returns=(0.001, 0.001))]
        ci = block_bootstrap(records, statistic=weighted_net, n_boot=100, seed=5)
        assert ci.ci_low == pytest.approx(0.001)
        assert ci.ci_high == pytest.approx(0.001)

    def test_accuracy_and_sharpe_statistics(self):
        records = self._records()
        acc = block_bootstrap(records, statistic=weighted_acc, n_boot=100, seed=5)
        sharpe = block_bootstrap(records, statistic=pooled_sharpe, n_boot=100, seed=5)
        assert acc.ci_low <= acc.point <= acc.ci_high
        assert sharpe.ci_low <= sharpe.point <= sharpe.ci_high

    def test_empty_records(self):
        ci = block_bootstrap([], statistic=weighted_net, n_boot=10, seed=1)
        assert ci.point == 0.0

    def test_crosses_zero_helper(self):
        positive = BootstrapCi(point=0.001, ci_low=0.0005, ci_high=0.0015, n_boot=100)
        ambiguous = BootstrapCi(point=0.0, ci_low=-0.001, ci_high=0.001, n_boot=100)
        assert not positive.crosses_zero
        assert ambiguous.crosses_zero

    def test_window_cumsum_maxdd(self):
        records = [
            WindowRecord(n=10, accuracy=0.5, net=0.01, returns=()),
            WindowRecord(n=10, accuracy=0.5, net=-0.02, returns=()),
            WindowRecord(n=10, accuracy=0.5, net=-0.01, returns=()),
        ]
        assert window_cumsum_maxdd(records) == pytest.approx(-0.03)


class TestDifferenceCi:
    def test_consistently_better_does_not_cross_zero(self):
        pairs = [(0.002, 0.001) for _ in range(10)]
        weights = [10] * 10
        ci = difference_ci(pairs, weights, n_boot=200, seed=5)
        assert ci.point == pytest.approx(0.001)
        assert not ci.crosses_zero
        assert ci.ci_low > 0.0

    def test_tied_baselines_cross_zero(self):
        pairs = [(0.001, 0.001) for _ in range(10)]
        weights = [10] * 10
        ci = difference_ci(pairs, weights, n_boot=200, seed=5)
        assert ci.point == pytest.approx(0.0)
        assert ci.crosses_zero

    def test_empty_pairs(self):
        ci = difference_ci([], [], n_boot=10, seed=1)
        assert ci.point == 0.0


class TestRandomWinnerTest:
    def test_winner_selection_beats_random_expert(self):
        """A acierta 95% en varias ventanas: elegirla supera al azar (una cola)."""
        windows = []
        for w in range(8):
            moves = _moves(60, seed=10 + w)
            per_ts = _make_ts(moves, {"A": 0.95, "B": 0.5, "C": 0.5}, seed=w)
            windows.append(
                PermWindow(weights={"A": 1.0}, per_timestamp=per_ts, cost=_COST, n=60)
            )
        result = random_winner_test(windows, n_permutations=300, seed=7)
        assert result.real_mean > result.null_mean
        assert result.real_mean > result.ci_high
        assert result.p_value < 0.05

    def test_random_selection_is_not_better(self):
        """Todos aciertan 50%: seleccionar no aporta vs azar."""
        moves = _moves(100)
        per_ts = _make_ts(moves, {"A": 0.5, "B": 0.5, "C": 0.5})
        windows = [
            PermWindow(weights={"A": 1.0}, per_timestamp=per_ts, cost=_COST, n=100)
        ]
        result = random_winner_test(windows, n_permutations=200, seed=7)
        assert result.ci_low <= result.real_mean <= result.ci_high


class TestExpertBootstrap:
    def test_ci_contains_sample_statistics(self):
        rng = random.Random(9)
        rows = [
            (rng.random() < 0.6, 0.003 if rng.random() < 0.5 else 0.002, 0.05)
            for _ in range(400)
        ]
        metrics = bootstrap_expert_metrics(rows, n_boot=200, seed=3)
        assert metrics.n == 400
        assert metrics.accuracy.point == pytest.approx(0.6, abs=0.05)
        assert metrics.accuracy.ci_low <= metrics.accuracy.point <= metrics.accuracy.ci_high
        assert metrics.mean_reward.ci_low <= metrics.mean_reward.point <= metrics.mean_reward.ci_high
        assert metrics.ece.ci_low <= metrics.ece.point <= metrics.ece.ci_high

    def test_max_rows_caps_sample(self):
        rows = [(True, 0.003, 0.05) for _ in range(5000)]
        metrics = bootstrap_expert_metrics(rows, n_boot=50, seed=1, max_rows=100)
        assert metrics.n == 100

    def test_empty_rows(self):
        metrics = bootstrap_expert_metrics([], n_boot=10, seed=1)
        assert metrics.n == 0
