"""Comparison metrics for A/B testing pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.types import (
    UniversalResult,
)


@dataclass
class PipelineOutput:
    """Structured extraction from UniversalResult."""
    
    severity: str
    confidence: float
    explanation_text: str
    entity_count: int
    equipment_metric_pairs: List[Dict]
    perceptions: List[Dict]
    has_semantic_alert: bool
    semantic_richness: float
    phases_executed: List[str]
    raw_result: UniversalResult


@dataclass
class ComparisonMetrics:
    """Delta metrics between control (A) and treatment (B)."""
    
    # Severity analysis
    severity_changed: bool
    severity_delta: str  # "none", "up", "down"
    
    # Confidence analysis
    confidence_delta: float  # Absolute difference
    confidence_delta_pct: float  # Percentage change
    
    # Entity detection
    entity_count_delta: int
    has_equipment_metric_pairs: bool
    pair_count_delta: int
    
    # Explanation quality
    explanation_length_delta: int
    explanation_density_delta: float  # Entities per 100 chars
    
    # Perception analysis
    perception_count_delta: int
    has_semantic_perceptions: bool
    
    # MoE analysis
    moe_selection_changed: bool
    moe_weights_changed: bool
    
    # Technical context
    mentions_equipment: bool
    mentions_metrics: bool
    mentions_anomaly: bool


class MetricCalculator:
    """Calculate comparison metrics between two pipeline outputs."""
    
    SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "warning": 3, "high": 4, "critical": 5}
    
    @staticmethod
    def extract_output(result: UniversalResult) -> PipelineOutput:
        """Extract structured data from UniversalResult."""
        analysis = result.analysis or {}
        explanation = result.explanation
        
        # Build explanation text from outcome if available
        explanation_text = ""
        if explanation and hasattr(explanation, 'outcome') and explanation.outcome:
            outcome = explanation.outcome
            explanation_text = f"kind={outcome.kind}, confidence={outcome.confidence:.2f}, trend={outcome.trend}"
        
        # Extract perceptions
        perceptions = analysis.get("perceptions", [])
        
        # Check for semantic perceptions
        semantic_perceptions = [
            p for p in perceptions 
            if isinstance(p, dict) and "semantic" in str(p.get("engine_name", ""))
        ]
        
        # Extract enrichment data if available
        enrichment = analysis.get("semantic_enrichment", {})
        
        # Extract phases from explanation trace if available
        phases = []
        if explanation and hasattr(explanation, 'trace') and explanation.trace:
            phases = [p.kind.value for p in explanation.trace.phases] if explanation.trace.phases else []
        
        return PipelineOutput(
            severity=result.severity,
            confidence=result.confidence,
            explanation_text=explanation_text,
            entity_count=enrichment.get("entity_count", 0),
            equipment_metric_pairs=enrichment.get("equipment_metric_pairs", []),
            perceptions=perceptions,
            has_semantic_alert=len(semantic_perceptions) > 0,
            semantic_richness=analysis.get("semantic_richness", 0.0),
            phases_executed=phases,
            raw_result=result,
        )
    
    @classmethod
    def compare(
        cls,
        control: PipelineOutput,
        treatment: PipelineOutput,
    ) -> ComparisonMetrics:
        """Compare control (no enrichment) vs treatment (with enrichment)."""
        
        # Severity comparison
        ctrl_sev_val = cls.SEVERITY_ORDER.get(control.severity, 0)
        treat_sev_val = cls.SEVERITY_ORDER.get(treatment.severity, 0)
        severity_changed = ctrl_sev_val != treat_sev_val
        severity_delta = "up" if treat_sev_val > ctrl_sev_val else (
            "down" if treat_sev_val < ctrl_sev_val else "none"
        )
        
        # Confidence delta
        conf_delta = treatment.confidence - control.confidence
        conf_delta_pct = (conf_delta / control.confidence * 100) if control.confidence > 0 else 0
        
        # Entity counts
        entity_delta = treatment.entity_count - control.entity_count
        pair_delta = len(treatment.equipment_metric_pairs) - len(control.equipment_metric_pairs)
        
        # Explanation quality
        expl_len_delta = len(treatment.explanation_text) - len(control.explanation_text)
        
        ctrl_density = control.entity_count / max(len(control.explanation_text) / 100, 1)
        treat_density = treatment.entity_count / max(len(treatment.explanation_text) / 100, 1)
        density_delta = treat_density - ctrl_density
        
        # Perception counts
        percep_delta = len(treatment.perceptions) - len(control.perceptions)
        
        # MoE analysis (check if phases or weights differ)
        moe_changed = control.phases_executed != treatment.phases_executed
        
        # Technical context detection (simple string checks)
        expl_lower = treatment.explanation_text.lower()
        mentions_equip = any(e in expl_lower for e in ["compresor", "bomba", "generador", "válvula", "motor", "c-", "p-", "gen-"])
        mentions_met = any(m in expl_lower for m in ["psi", "bar", "°c", "rpm", "kv", "gpm"])
        mentions_anom = any(a in expl_lower for a in ["anomal", "falla", "crític", "emergenc", "fuera de rango", "excedido"])
        
        return ComparisonMetrics(
            severity_changed=severity_changed,
            severity_delta=severity_delta,
            confidence_delta=conf_delta,
            confidence_delta_pct=conf_delta_pct,
            entity_count_delta=entity_delta,
            has_equipment_metric_pairs=len(treatment.equipment_metric_pairs) > 0,
            pair_count_delta=pair_delta,
            explanation_length_delta=expl_len_delta,
            explanation_density_delta=density_delta,
            perception_count_delta=percep_delta,
            has_semantic_perceptions=treatment.has_semantic_alert,
            moe_selection_changed=moe_changed,
            moe_weights_changed=False,  # Requires deeper analysis
            mentions_equipment=mentions_equip,
            mentions_metrics=mentions_met,
            mentions_anomaly=mentions_anom,
        )
