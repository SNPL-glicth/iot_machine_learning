"""Reasoning quality scoring algorithm for semantic enrichment evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .metrics import ComparisonMetrics, PipelineOutput


@dataclass
class QualityReport:
    """Detailed quality assessment for a single comparison."""
    
    control_score: int
    treatment_score: int
    improvement: int  # treatment - control
    improvement_pct: float
    
    # Component breakdown
    critical_entity_points: int
    relation_points: int
    technical_context_points: int
    confidence_points: int
    severity_precision_points: int
    
    # Assessment
    is_improvement: bool
    is_significant: bool  # improvement > 30
    assessment: str  # "major_improvement", "improvement", "neutral", "regression"


class ReasoningQualityScore:
    """Calculate composite quality score for reasoning output.
    
    Scoring weights:
    +30: Detects critical entities (equipment + anomaly)
    +25: Detects relations (equipment → metric)
    +20: Explanation mentions technical context
    +15: Confidence increases (up to 15 pts)
    +10: Severity becomes more precise/correct
    """
    
    MAX_SCORE = 100
    SIGNIFICANCE_THRESHOLD = 30
    
    @classmethod
    def calculate(
        cls,
        output: PipelineOutput,
        expected_critical: bool = False,
    ) -> Dict[str, int]:
        """Calculate quality score components for a single output."""
        score = 0
        components = {
            "critical_entity": 0,
            "relations": 0,
            "technical_context": 0,
            "confidence": 0,
            "severity_precision": 0,
        }
        
        # 1. Critical entities (+30 if equipment + anomaly detected)
        has_equipment = output.entity_count > 0 and any(
            p for p in output.perceptions 
            if isinstance(p, dict) and p.get("engine_name") == "semantic_entities"
        )
        has_critical_alert = output.has_semantic_alert
        
        if has_equipment and has_critical_alert:
            components["critical_entity"] = 30
            score += 30
        elif has_equipment:
            components["critical_entity"] = 15  # Partial credit
            score += 15
        
        # 2. Relations (+25 if equipment-metric pairs exist)
        if len(output.equipment_metric_pairs) > 0:
            components["relations"] = 25
            score += 25
        
        # 3. Technical context in explanation (+20)
        expl_lower = output.explanation_text.lower()
        tech_terms = [
            "compresor", "bomba", "generador", "válvula", "motor", "reactor",
            "transformador", "ventilador", "c-", "p-", "m-", "gen-", "r-",
            "psi", "bar", "°c", "rpm", "kv", "gpm", "presión", "temperatura",
            "flujo", "voltaje", "anomalía", "falla", "rango"
        ]
        tech_mentions = sum(1 for term in tech_terms if term in expl_lower)
        if tech_mentions >= 3:
            components["technical_context"] = 20
            score += 20
        elif tech_mentions >= 1:
            components["technical_context"] = 10
            score += 10
        # 4. Confidence (+15 max, scaled 0.6-0.95)
        if output.confidence >= 0.9:
            components["confidence"] = 15
            score += 15
        elif output.confidence >= 0.8:
            components["confidence"] = 10
            score += 10
        elif output.confidence >= 0.7:
            components["confidence"] = 5
            score += 5
        # 5. Severity precision (+10 if correct for context)
        # Correct: critical when expected, not critical when not expected
        is_critical = output.severity in ["high", "critical"]
        
        if expected_critical and is_critical:
            components["severity_precision"] = 10
            score += 10
        elif not expected_critical and not is_critical:
            components["severity_precision"] = 10
            score += 10
        elif not expected_critical and is_critical:
            # False positive - small penalty in precision
            components["severity_precision"] = 0
        
        components["total"] = score
        return components
    @classmethod
    def compare(
        cls,
        control: PipelineOutput,
        treatment: PipelineOutput,
        expected_critical: bool = False,
    ) -> QualityReport:
        """Generate quality comparison report."""
        
        control_scores = cls.calculate(control, expected_critical)
        treatment_scores = cls.calculate(treatment, expected_critical)
        control_total = control_scores["total"]
        treatment_total = treatment_scores["total"]
        improvement = treatment_total - control_total
        improvement_pct = (improvement / max(control_total, 1)) * 100
        
        if improvement >= cls.SIGNIFICANCE_THRESHOLD:
            assessment, is_sig = "major_improvement", True
        elif improvement > 10:
            assessment, is_sig = "improvement", False
        elif improvement >= -10:
            assessment, is_sig = "neutral", False
        else:
            assessment, is_sig = "regression", False
        
        return QualityReport(
            control_score=control_total, treatment_score=treatment_total,
            improvement=improvement, improvement_pct=improvement_pct,
            critical_entity_points=treatment_scores["critical_entity"],
            relation_points=treatment_scores["relations"],
            technical_context_points=treatment_scores["technical_context"],
            confidence_points=treatment_scores["confidence"],
            severity_precision_points=treatment_scores["severity_precision"],
            is_improvement=improvement > 0, is_significant=is_sig, assessment=assessment,
        )
    @staticmethod
    def aggregate_reports(reports: List[QualityReport]) -> Dict[str, any]:
        """Aggregate multiple quality reports into summary statistics."""
        if not reports:
            return {}
        
        improvements = [r.improvement for r in reports]
        
        return {
            "n_tests": len(reports),
            "avg_improvement": sum(improvements) / len(improvements),
            "max_improvement": max(improvements), "min_improvement": min(improvements),
            "major_improvements": sum(1 for r in reports if r.assessment == "major_improvement"),
            "improvements": sum(1 for r in reports if r.assessment == "improvement"),
            "neutral": sum(1 for r in reports if r.assessment == "neutral"),
            "regressions": sum(1 for r in reports if r.assessment == "regression"),
            "significant_rate": sum(1 for r in reports if r.is_significant) / len(reports),
        }
