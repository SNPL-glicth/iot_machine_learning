from __future__ import annotations

import pytest
from domain.entities.cognitive.failure_taxonomy import (
    FailureReason,
    diagnose_prediction_failure,
)
from domain.entities.cognitive.metacognitive_tracker import (
    MetacognitiveTracker,
)
from domain.entities.cognitive.post_mortem_evaluator import (
    PostMortemEvaluator,
    PostMortemRecord,
)
from infrastructure.ml.cognitive.metacognitive_coordinator import (
    MetacognitiveCoordinator,
)
from infrastructure.ml.cognitive.plasticity.regime_plasticity_manager import (
    RegimePlasticityManager,
)


class TestFailureTaxonomy:
    def test_successful_prediction_classified_as_none(self):
        diag = diagnose_prediction_failure(
            predicted_val=105.0,
            actual_val=104.8,
            prev_val=100.0,
            confidence=0.8,
            lambda_t=0.1,
            expert_variance=0.05,
        )
        assert diag.reason == FailureReason.NONE
        assert not diag.is_failure
        assert diag.severity == 0.0

    def test_outlier_classified_as_regime_disruption(self):
        diag = diagnose_prediction_failure(
            predicted_val=101.0,
            actual_val=125.0,
            prev_val=100.0,
            confidence=0.7,
            lambda_t=0.1,
            expert_variance=0.05,
            is_outlier=True,
        )
        assert diag.reason == FailureReason.REGIME_DISRUPTION
        assert diag.is_failure
        assert diag.severity == 1.0

    def test_directional_reversal_classified_as_inertia_collapse(self):
        diag = diagnose_prediction_failure(
            predicted_val=110.0,
            actual_val=90.0,
            prev_val=100.0,
            confidence=0.8,
            lambda_t=0.1,
            expert_variance=0.1,
        )
        assert diag.reason == FailureReason.INERTIA_COLLAPSE
        assert diag.is_failure
        assert diag.expected_direction == 1
        assert diag.actual_direction == -1

    def test_high_variance_classified_as_arbiter_overconfidence(self):
        diag = diagnose_prediction_failure(
            predicted_val=105.0,
            actual_val=98.0,
            prev_val=100.0,
            confidence=0.85,
            lambda_t=0.1,
            expert_variance=0.45,
            variance_threshold=0.25,
        )
        assert diag.reason == FailureReason.ARBITER_OVERCONFIDENCE
        assert diag.is_failure


class TestPostMortemEvaluator:
    def test_queue_and_evaluation_at_horizon(self):
        evaluator = PostMortemEvaluator()
        evaluator.record_prediction(
            prediction_id="p1",
            prev_value=100.0,
            predicted_value=105.0,
            expert_predictions={"kalman": 104.0, "taylor": 106.0, "statistical": 99.0},
            regime="bull_trend",
            confidence=0.7,
            horizon_steps=2,
        )

        # Step 1: aún no madura
        results_1 = evaluator.step(current_value=101.0)
        assert len(results_1) == 0

        # Step 2: madura
        results_2 = evaluator.step(current_value=104.2)
        assert len(results_2) == 1
        record = results_2[0]

        assert record.prediction_id == "p1"
        assert record.winning_expert == "kalman"  # Error 0.2 vs taylor 1.8
        assert record.directional_correctness["kalman"] is True
        assert record.directional_correctness["taylor"] is True
        assert record.directional_correctness["statistical"] is False


class TestMetacognitiveTracker:
    def test_systemic_blindness_triggers_lambda_penalty(self):
        tracker = MetacognitiveTracker(window_size=10, min_samples=3)

        # Simular 4 fallos correlacionados consecutivos
        for i in range(4):
            diag = diagnose_prediction_failure(
                predicted_val=105.0,
                actual_val=95.0,
                prev_val=100.0,
                confidence=0.8,
                lambda_t=0.1,
                expert_variance=0.1,
            )
            record = PostMortemRecord(
                prediction_id=f"p_{i}",
                regime="turbulent",
                actual_value=95.0,
                diagnostic=diag,
                expert_errors={"k": 10.0, "t": 12.0},
                winning_expert="k",
                directional_correctness={"k": False, "t": False},
            )
            tracker.record_outcome(record)

        status = tracker.get_status("turbulent")
        assert status.systemic_blindness_detected is True
        assert status.meta_competence_score <= 0.2
        # La modulación debe forzar lambda a 1.0
        modulated = tracker.modulate_exploration_factor(base_lambda=0.1, regime="turbulent")
        assert modulated == 1.0


class TestRegimePlasticityManager:
    def test_weight_shift_towards_accurate_expert(self):
        manager = RegimePlasticityManager(
            expert_names=["kalman", "taylor"], learning_rate=0.5
        )
        initial_w = manager.get_weights("trend")
        assert initial_w["kalman"] == initial_w["taylor"] == 0.5

        diag = diagnose_prediction_failure(102.0, 102.0, 100.0, 0.7, 0.1, 0.05)
        record = PostMortemRecord(
            prediction_id="p1",
            regime="trend",
            actual_value=102.0,
            diagnostic=diag,
            expert_errors={"kalman": 0.1, "taylor": 5.0},
            winning_expert="kalman",
            directional_correctness={"kalman": True, "taylor": False},
        )
        manager.update_from_post_mortem(record)
        updated_w = manager.get_weights("trend")

        # Kalman debe tener significativamente más peso que Taylor
        assert updated_w["kalman"] > updated_w["taylor"]
        assert pytest.approx(sum(updated_w.values())) == 1.0

    def test_circuit_breaker_inhibition(self):
        manager = RegimePlasticityManager(
            expert_names=["kalman", "taylor"], max_consecutive_failures=2
        )
        diag = diagnose_prediction_failure(105.0, 95.0, 100.0, 0.8, 0.1, 0.1)

        for i in range(2):
            rec = PostMortemRecord(
                prediction_id=f"p_{i}",
                regime="volatile",
                actual_value=95.0,
                diagnostic=diag,
                expert_errors={"kalman": 1.0, "taylor": 10.0},
                winning_expert="kalman",
                directional_correctness={"kalman": True, "taylor": False},
            )
            manager.update_from_post_mortem(rec)

        assert manager.is_inhibited("taylor", "volatile") is True
        w = manager.get_weights("volatile")
        assert w["taylor"] == 0.0
        assert w["kalman"] == 1.0

    def test_state_persistence_roundtrip(self):
        manager = RegimePlasticityManager(expert_names=["kalman", "taylor"])
        diag = diagnose_prediction_failure(102.0, 102.0, 100.0, 0.7, 0.1, 0.05)
        rec = PostMortemRecord("p1", "reg1", 102.0, diag, {"kalman": 0.1, "taylor": 4.0}, "kalman", {"kalman": True, "taylor": False})
        manager.update_from_post_mortem(rec)

        state = manager.export_state()
        new_manager = RegimePlasticityManager(expert_names=["kalman", "taylor"])
        new_manager.import_state(state)

        assert new_manager.get_weights("reg1") == manager.get_weights("reg1")


class TestMetacognitiveCoordinator:
    def test_coordinator_end_to_end(self):
        coord = MetacognitiveCoordinator(expert_names=["kalman", "taylor"], horizon_steps=1)
        coord.register_inference(
            prediction_id="step_0",
            prev_value=100.0,
            predicted_value=102.0,
            expert_predictions={"kalman": 102.0, "taylor": 98.0},
            regime="normal",
            confidence=0.8,
            horizon_steps=1,
        )

        # Madura en step 1
        records = coord.process_step(current_value=102.5)
        assert len(records) == 1
        assert records[0].winning_expert == "kalman"

        weights = coord.get_expert_weights("normal")
        assert weights["kalman"] > weights["taylor"]
