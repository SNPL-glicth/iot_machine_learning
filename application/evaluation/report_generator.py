"""Report generator for semantic enrichment evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .runner import EvaluationResult
from .dataset import TestDataset


@dataclass
class EvaluationSummary:
    """High-level summary of evaluation."""
    timestamp: str
    total_cases: int
    overall_improvement_rate: float
    significant_improvement_rate: float
    avg_quality_improvement: float
    recommendation: str
    key_findings: List[str]


class ReportGenerator:
    """Generate human-readable and JSON reports."""
    
    def __init__(self, runner_results: List[EvaluationResult]):
        self._results = runner_results
    
    def generate_per_document_report(
        self,
        result: EvaluationResult,
    ) -> Dict[str, Any]:
        """Generate detailed report for single document."""
        tc = result.test_case
        ctrl = result.control_output
        treat = result.treatment_output
        m = result.metrics
        q = result.quality_report
        
        return {
            "document_id": tc.id,
            "category": tc.category,
            "description": tc.description,
            "input_text": tc.text[:100] + "..." if len(tc.text) > 100 else tc.text,
            
            "control_pipeline": {
                "severity": ctrl.severity,
                "confidence": round(ctrl.confidence, 4),
                "entities_detected": ctrl.entity_count,
                "equipment_metric_pairs": len(ctrl.equipment_metric_pairs),
                "perceptions": len(ctrl.perceptions),
                "quality_score": q.control_score,
            },
            
            "treatment_pipeline": {
                "severity": treat.severity,
                "confidence": round(treat.confidence, 4),
                "entities_detected": treat.entity_count,
                "equipment_metric_pairs": [
                    {
                        "equipment": p.get("equipment"),
                        "metric": p.get("metric"),
                        "is_anomaly": p.get("is_anomaly"),
                    }
                    for p in treat.equipment_metric_pairs
                ],
                "perceptions": len(treat.perceptions),
                "semantic_perceptions_present": m.has_semantic_perceptions,
                "quality_score": q.treatment_score,
            },
            
            "comparison": {
                "severity_changed": m.severity_changed,
                "severity_direction": m.severity_delta,
                "confidence_delta": round(m.confidence_delta, 4),
                "confidence_delta_pct": round(m.confidence_delta_pct, 2),
                "entity_delta": m.entity_count_delta,
                "explanation_length_delta": m.explanation_length_delta,
            },
            
            "quality_assessment": {
                "score_improvement": q.improvement,
                "improvement_pct": round(q.improvement_pct, 2),
                "assessment": q.assessment,
                "is_significant": q.is_significant,
                "component_breakdown": {
                    "critical_entity_points": q.critical_entity_points,
                    "relation_points": q.relation_points,
                    "technical_context_points": q.technical_context_points,
                    "confidence_points": q.confidence_points,
                    "severity_precision_points": q.severity_precision_points,
                },
            },
            
            "technical_context_detected": {
                "mentions_equipment": m.mentions_equipment,
                "mentions_metrics": m.mentions_metrics,
                "mentions_anomaly": m.mentions_anomaly,
            },
        }
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate complete evaluation report."""
        if not self._results:
            return {"error": "No results to report"}
        
        # Per-document reports
        document_reports = [
            self.generate_per_document_report(r) for r in self._results
        ]
        
        # Aggregate statistics
        improvements = [r.quality_report.improvement for r in self._results]
        avg_improvement = sum(improvements) / len(improvements)
        
        significant_count = sum(1 for r in self._results if r.quality_report.is_significant)
        improvement_count = sum(1 for r in self._results if r.quality_report.is_improvement)
        categories = {}
        for cat in ["industrial", "neutral", "noise"]:
            cat_results = [r for r in self._results if r.test_case.category == cat]
            if cat_results:
                cat_improvements = [r.quality_report.improvement for r in cat_results]
                categories[cat] = {"count": len(cat_results), "avg_improvement": round(sum(cat_improvements) / len(cat_improvements), 2), "improvement_rate": sum(1 for i in cat_improvements if i > 0) / len(cat_improvements)}
        moe_changed_count = sum(1 for r in self._results if r.metrics.moe_selection_changed)
        # Generate recommendation
        if significant_count >= len(self._results) * 0.5:
            recommendation, findings = "MAINTAIN", [f"Significant improvement in {significant_count}/{len(self._results)} cases", "Semantic enrichment adds measurable value to reasoning quality"]
        elif improvement_count >= len(self._results) * 0.6:
            recommendation, findings = "ADJUST", [f"Improvement in {improvement_count}/{len(self._results)} cases but not significant", "Consider tuning entity extraction thresholds"]
        else:
            recommendation, findings = "ROLLBACK", [f"Only {improvement_count}/{len(self._results)} cases show improvement", "Semantic enrichment may not justify overhead"]
        summary = EvaluationSummary(timestamp=datetime.now().isoformat(), total_cases=len(self._results), overall_improvement_rate=improvement_count / len(self._results), significant_improvement_rate=significant_count / len(self._results), avg_quality_improvement=avg_improvement, recommendation=recommendation, key_findings=findings)
        return {"summary": asdict(summary), "by_category": categories, "moe_impact": {"phases_changed_count": moe_changed_count, "phases_changed_rate": moe_changed_count / len(self._results)}, "documents": document_reports}
    
    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string."""
        report = self.generate_full_report()
        return json.dumps(report, indent=indent, ensure_ascii=False)
    
    def print_summary(self) -> None:
        """Print human-readable summary to stdout."""
        report = self.generate_full_report()
        summary = report.get("summary", {})
        print("\n" + "=" * 70)
        print("SEMANTIC ENRICHMENT EVALUATION REPORT")
        print("=" * 70)
        print(f"Timestamp: {summary.get('timestamp', 'N/A')}")
        print(f"Total Test Cases: {summary.get('total_cases', 0)}\n")
        print("OVERALL RESULTS:")
        print(f"  • Improvement Rate: {summary.get('overall_improvement_rate', 0)*100:.1f}%")
        print(f"  • Significant Improvement Rate: {summary.get('significant_improvement_rate', 0)*100:.1f}%")
        print(f"  • Avg Quality Improvement: {summary.get('avg_quality_improvement', 0):.1f} points\n")
        print("BY CATEGORY:")
        for cat, data in report.get("by_category", {}).items():
            print(f"  • {cat.upper()}: {data['count']} cases, avg: {data['avg_improvement']:.1f}, rate: {data['improvement_rate']*100:.1f}%")
        print()
        moe = report.get("moe_impact", {})
        print("MOE IMPACT:")
        print(f"  • Cases with changed phases: {moe.get('phases_changed_rate', 0)*100:.1f}%\n")
        print("RECOMMENDATION:")
        print(f"  >>> {summary.get('recommendation', 'UNKNOWN')} <<<\n")
        print("KEY FINDINGS:")
        for finding in summary.get("key_findings", []):
            print(f"  • {finding}")
        print("=" * 70 + "\n")
