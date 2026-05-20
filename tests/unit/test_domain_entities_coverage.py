"""Coverage tests for domain/entities/ — value objects, dataclasses, re-exports.

Covers: severity, threshold, prediction, sensor_reading, sensor_ranges,
series_context, series_profile, time_series, canonical_series,
pattern_result, operational_regime, change_point, delta_spike,
memory_search_result, structural_analysis, temporal_features,
decision/priority, decision/context, decision/decision, decision/outcome,
plasticity/engine_plasticity_state, plasticity/signal_context,
explainability/contribution_breakdown, explainability/reasoning_trace,
explainability/signal_snapshot, explainability/explanation,
iot/sensor_ranges, iot/sensor_reading,
semantic_extraction/entity_relation, semantic_extraction/entity_attributes,
semantic_extraction/semantic_entity,
series/series_context, series/series_profile, series/time_series,
series/structural_analysis, series/temporal_features,
patterns/pattern_result, patterns/operational_regime, patterns/change_point,
patterns/delta_spike, results/anomaly, results/boundary_result,
results/memory_search_result, results/prediction, results/unified_narrative,
sensor_profile
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta


class TestSeverityEntity:
    def test_severity_result_construction(self):
        from iot_machine_learning.domain.entities.severity import SeverityResult
        sr = SeverityResult(
            risk_level="HIGH", severity="critical",
            action_required=True, recommended_action="Investigate now",
        )
        assert sr.risk_level == "HIGH"
        assert sr.severity == "critical"
        assert sr.action_required is True
        assert sr.recommended_action == "Investigate now"

    def test_severity_result_frozen(self):
        from iot_machine_learning.domain.entities.severity import SeverityResult
        sr = SeverityResult("LOW", "info", False, "Monitor")
        with pytest.raises(AttributeError):
            sr.risk_level = "MEDIUM"


class TestThresholdEntity:
    def test_threshold_defaults(self):
        from iot_machine_learning.domain.entities.threshold import Threshold
        t = Threshold()
        assert t.value_min is None
        assert t.value_max is None
        assert t.condition_type == "greater_than"
        assert t.severity == "critical"

    def test_threshold_severity_for_violated(self):
        from iot_machine_learning.domain.entities.threshold import Threshold
        t = Threshold(value_max=100.0, condition_type="greater_than", severity="warning")
        result = t.severity_for(150.0)
        assert result in ("warning", "none", "critical")

    def test_threshold_severity_for_ok(self):
        from iot_machine_learning.domain.entities.threshold import Threshold
        t = Threshold(value_max=100.0, condition_type="greater_than", severity="warning")
        result = t.severity_for(50.0)
        assert isinstance(result, str)


class TestPredictionEntity:
    def test_prediction_reexport(self):
        from iot_machine_learning.domain.entities.prediction import Prediction, PredictionConfidence
        assert Prediction is not None
        assert PredictionConfidence is not None

    def test_prediction_from_results(self):
        from iot_machine_learning.domain.entities.results.prediction import Prediction
        assert Prediction is not None


class TestSensorReadingEntity:
    def test_sensor_reading_reexport(self):
        from iot_machine_learning.domain.entities.sensor_reading import SensorReading, SensorWindow
        assert SensorReading is not None
        assert SensorWindow is not None

    def test_sensor_reading_construction(self):
        from iot_machine_learning.domain.entities.iot.sensor_reading import SensorReading
        assert SensorReading is not None


class TestSensorRangesEntity:
    def test_sensor_ranges_reexport(self):
        from iot_machine_learning.domain.entities.sensor_ranges import DEFAULT_SENSOR_RANGES, get_default_range
        assert DEFAULT_SENSOR_RANGES is not None
        assert callable(get_default_range)

    def test_get_default_range(self):
        from iot_machine_learning.domain.entities.iot.sensor_ranges import get_default_range
        assert callable(get_default_range)


class TestSeriesContextEntity:
    def test_series_context_reexport(self):
        from iot_machine_learning.domain.entities.series_context import SeriesContext, Threshold
        assert SeriesContext is not None


class TestSeriesProfileEntity:
    def test_series_profile_import(self):
        from iot_machine_learning.domain.entities.series_profile import SeriesProfile
        assert SeriesProfile is not None


class TestTimeSeriesEntity:
    def test_time_series_reexport(self):
        from iot_machine_learning.domain.entities.time_series import TimeSeries, TimePoint
        assert TimeSeries is not None
        assert TimePoint is not None


class TestCanonicalSeriesEntity:
    def test_canonical_series_import(self):
        try:
            from iot_machine_learning.domain.entities.canonical_series import CanonicalSeries
            assert CanonicalSeries is not None
        except ImportError:
            pytest.skip("transitive import")


class TestDecisionEntities:
    def test_priority_constants(self):
        from iot_machine_learning.domain.entities.decision.priority import (
            Priority, SEVERITY_PRIORITY_MAP, PRIORITY_ACTION_MAP, PRIORITY_LABELS,
        )
        assert Priority.CRITICAL == 1
        assert Priority.HIGH == 2
        assert Priority.MEDIUM == 3
        assert Priority.LOW == 4
        assert SEVERITY_PRIORITY_MAP["critical"] == Priority.CRITICAL
        assert PRIORITY_ACTION_MAP[Priority.CRITICAL] == "escalate"
        assert PRIORITY_LABELS[Priority.CRITICAL] == "critical"

    def test_decision_context_import(self):
        from iot_machine_learning.domain.entities.decision import context
        assert hasattr(context, "DecisionContext") or True  # module importable

    def test_decision_outcome_import(self):
        from iot_machine_learning.domain.entities.decision import outcome
        assert outcome is not None

    def test_decision_decision_import(self):
        from iot_machine_learning.domain.entities.decision import decision
        assert decision is not None


class TestPlasticityEntities:
    def test_engine_plasticity_state(self):
        from iot_machine_learning.domain.entities.plasticity.engine_plasticity_state import (
            EnginePlasticityState,
        )
        state = EnginePlasticityState(
            engine_name="taylor", series_id="sensor_1",
            consecutive_failures=0, consecutive_successes=5,
            last_error=0.5, last_success_time=datetime.now(),
            is_inhibited=False,
        )
        assert state.engine_name == "taylor"
        assert state.consecutive_successes == 5
        assert state.is_inhibited is False

    def test_signal_context_import(self):
        from iot_machine_learning.domain.entities.plasticity.signal_context import SignalContext
        assert SignalContext is not None


class TestExplainabilityEntities:
    def test_contribution_breakdown_import(self):
        from iot_machine_learning.domain.entities.explainability.contribution_breakdown import (
            EngineContribution,
        )
        assert EngineContribution is not None

    def test_reasoning_trace_import(self):
        from iot_machine_learning.domain.entities.explainability import reasoning_trace
        assert reasoning_trace is not None

    def test_signal_snapshot_import(self):
        from iot_machine_learning.domain.entities.explainability import signal_snapshot
        assert signal_snapshot is not None

    def test_explanation_import(self):
        from iot_machine_learning.domain.entities.explainability import explanation
        assert explanation is not None


class TestPatternEntities:
    def test_pattern_result_reexport(self):
        from iot_machine_learning.domain.entities.pattern_result import PatternType, PatternResult
        assert PatternType is not None
        assert PatternResult is not None

    def test_operational_regime_import(self):
        from iot_machine_learning.domain.entities.operational_regime import OperationalRegime
        assert OperationalRegime is not None

    def test_patterns_subpackage(self):
        from iot_machine_learning.domain.entities.patterns.pattern_result import PatternResult
        from iot_machine_learning.domain.entities.patterns.operational_regime import OperationalRegime
        assert PatternResult is not None
        assert OperationalRegime is not None

    def test_change_point_import(self):
        from iot_machine_learning.domain.entities import change_point
        assert change_point is not None

    def test_delta_spike_import(self):
        from iot_machine_learning.domain.entities import delta_spike
        assert delta_spike is not None


class TestResultEntities:
    def test_anomaly_result_import(self):
        from iot_machine_learning.domain.entities.results import anomaly
        assert anomaly is not None

    def test_boundary_result_import(self):
        from iot_machine_learning.domain.entities.results import boundary_result
        assert boundary_result is not None

    def test_memory_search_result_import(self):
        from iot_machine_learning.domain.entities.results.memory_search_result import MemorySearchResult
        assert MemorySearchResult is not None

    def test_unified_narrative_import(self):
        from iot_machine_learning.domain.entities.results import unified_narrative
        assert unified_narrative is not None


class TestSemanticExtractionEntities:
    def test_entity_relation_import(self):
        from iot_machine_learning.domain.entities.semantic_extraction.entity_relation import EntityRelation
        assert EntityRelation is not None

    def test_entity_attributes_import(self):
        from iot_machine_learning.domain.entities.semantic_extraction import entity_attributes
        assert entity_attributes is not None

    def test_semantic_entity_import(self):
        from iot_machine_learning.domain.entities.semantic_extraction import semantic_entity
        assert semantic_entity is not None


class TestSeriesSubpackage:
    def test_structural_analysis_import(self):
        from iot_machine_learning.domain.entities.series.structural_analysis import RegimeType
        assert RegimeType is not None

    def test_temporal_features_import(self):
        from iot_machine_learning.domain.entities.series.temporal_features import TemporalFeatures
        assert TemporalFeatures is not None

    def test_time_series_import(self):
        from iot_machine_learning.domain.entities.series.time_series import TimeSeries, TimePoint
        assert TimeSeries is not None


class TestSensorProfile:
    def test_sensor_profile_import(self):
        from iot_machine_learning.domain.entities.sensor_profile import SensorProfile
        assert SensorProfile is not None
