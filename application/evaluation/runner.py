"""Evaluation runner for A/B testing semantic enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from iot_machine_learning.infrastructure.ml.cognitive.universal import (
    UniversalAnalysisEngine,
)
from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.types import (
    UniversalContext,
    UniversalInput,
)

from .dataset import TestCase, TestDataset
from .metrics import ComparisonMetrics, MetricCalculator, PipelineOutput
from .quality_score import QualityReport, ReasoningQualityScore


@dataclass
class EvaluationResult:
    """Complete result for a single test case."""
    
    test_case: TestCase
    control_output: PipelineOutput
    treatment_output: PipelineOutput
    metrics: ComparisonMetrics
    quality_report: QualityReport


class EvaluationRunner:
    """Execute A/B evaluation comparing pipelines with/without enrichment."""
    
    def __init__(
        self,
        budget_ms: float = 2000.0,
        deterministic_mode: bool = True,
    ):
        self._budget_ms = budget_ms
        self._deterministic = deterministic_mode
        self._results: List[EvaluationResult] = []
    
    def run_single(
        self,
        test_case: TestCase,
    ) -> EvaluationResult:
        """Execute both pipelines for a single test case."""
        
        ctx = UniversalContext(
            series_id=test_case.id,
            domain_hint="industrial" if test_case.category == "industrial" else "general",
        )
        
        # Run control (no enrichment)
        control_engine = UniversalAnalysisEngine(
            enable_semantic_enrichment=False,
            budget_ms=self._budget_ms,
            deterministic_mode=self._deterministic,
        )
        control_result = control_engine.analyze(test_case.text, ctx)
        control_output = MetricCalculator.extract_output(control_result)
        
        # Run treatment (with enrichment)
        treatment_engine = UniversalAnalysisEngine(
            enable_semantic_enrichment=True,
            budget_ms=self._budget_ms,
            deterministic_mode=self._deterministic,
        )
        treatment_result = treatment_engine.analyze(test_case.text, ctx)
        treatment_output = MetricCalculator.extract_output(treatment_result)
        
        # Calculate metrics
        metrics = MetricCalculator.compare(control_output, treatment_output)
        
        # Calculate quality scores
        quality = ReasoningQualityScore.compare(
            control_output,
            treatment_output,
            expected_critical=test_case.has_critical,
        )
        
        result = EvaluationResult(
            test_case=test_case,
            control_output=control_output,
            treatment_output=treatment_output,
            metrics=metrics,
            quality_report=quality,
        )
        
        self._results.append(result)
        return result
    
    def run_all(
        self,
        dataset: Optional[List[TestCase]] = None,
    ) -> List[EvaluationResult]:
        """Run evaluation on all test cases."""
        cases = dataset or TestDataset.all_cases()
        
        results = []
        for case in cases:
            result = self.run_single(case)
            results.append(result)
        
        return results
    
    def run_by_category(
        self,
        category: str,
    ) -> List[EvaluationResult]:
        """Run evaluation on specific category."""
        cases = TestDataset.get_by_category(category)
        return self.run_all(cases)
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self._results:
            return {"error": "No results available"}
        
        reports = [r.quality_report for r in self._results]
        quality_summary = ReasoningQualityScore.aggregate_reports(reports)
        
        # Additional pipeline metrics
        conf_improvements = [r.metrics.confidence_delta for r in self._results]
        entity_improvements = [r.metrics.entity_count_delta for r in self._results]
        
        # Category breakdown
        industrial_results = [r for r in self._results if r.test_case.category == "industrial"]
        neutral_results = [r for r in self._results if r.test_case.category == "neutral"]
        noise_results = [r for r in self._results if r.test_case.category == "noise"]
        
        return {
            "quality": quality_summary,
            "pipeline_metrics": {
                "avg_confidence_delta": sum(conf_improvements) / len(conf_improvements),
                "avg_entity_delta": sum(entity_improvements) / len(entity_improvements),
                "cases_with_semantic_perceptions": sum(
                    1 for r in self._results if r.metrics.has_semantic_perceptions
                ),
                "cases_with_equipment_pairs": sum(
                    1 for r in self._results if r.metrics.has_equipment_metric_pairs
                ),
            },
            "by_category": {
                "industrial": {
                    "count": len(industrial_results),
                    "avg_improvement": (
                        sum(r.quality_report.improvement for r in industrial_results) / 
                        max(len(industrial_results), 1)
                    ),
                },
                "neutral": {"count": len(neutral_results), "avg_improvement": (sum(r.quality_report.improvement for r in neutral_results) / max(len(neutral_results), 1))},
                "noise": {"count": len(noise_results), "avg_improvement": (sum(r.quality_report.improvement for r in noise_results) / max(len(noise_results), 1))},
            },
        }
    @property
    def results(self) -> List[EvaluationResult]:
        return list(self._results)
    def clear(self) -> None:
        self._results.clear()
