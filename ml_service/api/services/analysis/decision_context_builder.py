"""Decision context builder for document analysis.

Extracted from document_analyzer.py as part of refactoring Paso B.
Builds DecisionContext from UniversalResult with contextual enrichment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DecisionContextBuilder:
    """Builds DecisionContext from analysis results with contextual enrichment.
    
    Handles extraction of fields, pattern conversion, anomaly tracking integration,
    and contextual data enrichment from SignalProfile.
    
    Args:
        anomaly_tracker: Optional tracker for anomaly statistics
    """
    
    def __init__(self, anomaly_tracker: Optional[Any] = None) -> None:
        """Initialize builder with optional anomaly tracker."""
        self._anomaly_tracker = anomaly_tracker
    
    def build(
        self,
        universal_result: object,
        document_id: str,
    ) -> "DecisionContext":
        """Build DecisionContext from UniversalResult.
        
        Args:
            universal_result: UniversalResult from ML pipeline
            document_id: Document identifier
            
        Returns:
            DecisionContext for decision engine
        """
        # Extract fields safely with defaults
        severity = getattr(universal_result, "severity", None)
        confidence = getattr(universal_result, "confidence", 0.0)
        domain = getattr(universal_result, "domain", "")
        patterns = getattr(universal_result, "patterns", [])
        
        # Convert patterns to dict format
        pattern_dicts = self._convert_patterns(patterns)
        
        # Extract outcome from explanation
        outcome_data = self._extract_outcome(universal_result)
        
        # Get anomaly statistics
        anomaly_stats = self._get_anomaly_stats(document_id, outcome_data["is_anomaly"])
        
        # Record to tracker
        self._record_to_tracker(
            document_id=document_id,
            is_anomaly=outcome_data["is_anomaly"],
            anomaly_score=outcome_data["anomaly_score"],
            regime=outcome_data["current_regime"],
        )
        
        # Import here to avoid circular dependencies
        from iot_machine_learning.domain.entities.decision import DecisionContext
        
        return DecisionContext(
            series_id=document_id,
            severity=severity,
            confidence=confidence,
            is_anomaly=outcome_data["is_anomaly"],
            anomaly_score=outcome_data["anomaly_score"] or 0.0,
            patterns=pattern_dicts,
            predicted_value=outcome_data["predicted_value"],
            trend=outcome_data["trend"],
            domain=domain,
            audit_trace_id=outcome_data["audit_trace_id"],
            # Contextual enrichment
            recent_anomaly_count=anomaly_stats["recent_anomaly_count"],
            recent_anomaly_rate=anomaly_stats["recent_anomaly_rate"],
            consecutive_anomalies=anomaly_stats["consecutive_anomalies"],
            current_regime=outcome_data["current_regime"],
            drift_score=outcome_data["drift_score"],
        )
    
    def _convert_patterns(self, patterns: List[Any]) -> List[Dict[str, Any]]:
        """Convert pattern objects to dict format."""
        pattern_dicts = []
        for p in patterns:
            if hasattr(p, "to_dict"):
                pattern_dicts.append(p.to_dict())
            elif isinstance(p, dict):
                pattern_dicts.append(p)
            else:
                pattern_dicts.append({
                    "pattern_type": getattr(p, "pattern_type", "unknown"),
                    "severity_hint": getattr(p, "severity_hint", "info"),
                    "confidence": getattr(p, "confidence", 0.0),
                })
        return pattern_dicts
    
    def _extract_outcome(self, universal_result: object) -> Dict[str, Any]:
        """Extract outcome data from universal result explanation."""
        explanation = getattr(universal_result, "explanation", None)
        
        outcome_data = {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "predicted_value": None,
            "trend": "stable",
            "audit_trace_id": None,
            "current_regime": "STABLE",
            "drift_score": 0.0,
        }
        
        if explanation is None:
            return outcome_data
        
        outcome = getattr(explanation, "outcome", None)
        if outcome is not None:
            outcome_data["is_anomaly"] = getattr(outcome, "is_anomaly", False)
            outcome_data["anomaly_score"] = getattr(outcome, "anomaly_score", 0.0)
            outcome_data["predicted_value"] = getattr(outcome, "predicted_value", None)
            outcome_data["trend"] = getattr(outcome, "trend", "stable")
        
        outcome_data["audit_trace_id"] = getattr(explanation, "audit_trace_id", None)
        
        # Extract SignalProfile data
        signal_profile = getattr(explanation, "signal", None)
        if signal_profile is not None:
            regime_attr = getattr(signal_profile, "regime", None)
            if regime_attr is not None:
                outcome_data["current_regime"] = getattr(regime_attr, "value", str(regime_attr))
            outcome_data["drift_score"] = getattr(signal_profile, "drift_score", 0.0)
        
        return outcome_data
    
    def _get_anomaly_stats(self, document_id: str, is_anomaly: bool) -> Dict[str, Any]:
        """Get anomaly statistics from tracker."""
        stats = {
            "recent_anomaly_count": 0,
            "consecutive_anomalies": 0,
            "recent_anomaly_rate": 0.0,
        }
        
        if self._anomaly_tracker is None:
            return stats
        
        try:
            stats["recent_anomaly_count"] = self._anomaly_tracker.get_count_last_n_minutes(
                document_id, 120
            )
            stats["consecutive_anomalies"] = self._anomaly_tracker.get_consecutive_count(
                document_id
            )
            stats["recent_anomaly_rate"] = self._anomaly_tracker.get_anomaly_rate(
                document_id, 120
            )
        except Exception as e:
            logger.debug(f"anomaly_stats_query_failed: {e}")
        
        return stats
    
    def _record_to_tracker(
        self,
        document_id: str,
        is_anomaly: bool,
        anomaly_score: float,
        regime: str,
    ) -> None:
        """Record anomaly or normal to tracker."""
        if self._anomaly_tracker is None:
            return
        
        try:
            if is_anomaly:
                self._anomaly_tracker.record_anomaly(
                    document_id, anomaly_score, regime=regime
                )
            else:
                self._anomaly_tracker.record_normal(document_id)
        except Exception as e:
            logger.debug(f"tracker_record_failed: {e}")
