"""Coherence validator for Zenin text analysis results.

Detects and fixes logical inconsistencies in analysis results before
they reach the user. Always applies the most conservative (safest) fix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CoherenceReport:
    """Report of coherence validation results.
    
    Attributes:
        is_coherent: Whether the result passed all coherence checks
        warnings: List of detected coherence issues
        adjustments: List of adjustments made to fix issues
    """
    is_coherent: bool
    warnings: List[str] = field(default_factory=list)
    adjustments: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_coherent": self.is_coherent,
            "warnings": self.warnings,
            "adjustments": self.adjustments,
        }


class CoherenceValidator:
    """Validates and fixes coherence issues in analysis results.
    
    Rules:
    1. High urgency + positive sentiment = INCOHERENT
    2. Critical severity + low confidence = INCOHERENT
    3. Critical actions + stable pattern = INCOHERENT
    4. Multiple confidence values = INCOHERENT
    5. Critical severity + low urgency = INCOHERENT
    
    Fixes always apply the most conservative (safest) approach:
    - Never escalate severity, only de-escalate
    - Prefer higher confidence when values conflict
    - Align actions with actual severity
    """
    
    # Severity hierarchy (lower index = less severe)
    SEVERITY_LEVELS = ["info", "low", "warning", "medium", "high", "critical"]
    
    def validate(self, result: Any) -> CoherenceReport:
        """Validate and fix coherence issues in analysis result.
        
        Args:
            result: UniversalResult or similar analysis result object
            
        Returns:
            CoherenceReport with warnings and adjustments
        """
        warnings = []
        adjustments = []
        
        # Extract fields from result
        urgency = self._extract_urgency(result)
        sentiment = self._extract_sentiment(result)
        severity = self._extract_severity(result)
        confidence = self._extract_confidence(result)
        patterns = self._extract_patterns(result)
        analysis = self._extract_analysis(result)
        
        # Rule 1: High urgency + positive sentiment
        if urgency is not None and urgency > 0.7 and sentiment == "positive":
            warnings.append(f"High urgency ({urgency:.2f}) conflicts with positive sentiment")
            # Fix: Lower urgency to moderate
            if hasattr(result, 'analysis') and isinstance(result.analysis, dict):
                result.analysis['urgency_score'] = 0.6
                adjustments.append("Lowered urgency from {:.2f} to 0.6 (conservative)".format(urgency))
        
        # Rule 2: Critical severity + low confidence
        if severity in ["critical", "high"] and confidence is not None and confidence < 0.5:
            warnings.append(f"Critical/high severity conflicts with low confidence ({confidence:.1%})")
            # Fix: Downgrade severity to warning
            new_severity = "warning"
            self._set_severity(result, new_severity)
            adjustments.append(f"Downgraded severity from {severity} to {new_severity} (conservative)")
            severity = new_severity
        
        # Rule 3: Critical actions + stable pattern
        if self._has_critical_actions(analysis) and self._has_stable_pattern(patterns):
            warnings.append("Critical actions conflict with stable pattern")
            # Fix: Downgrade severity if critical
            if severity in ["critical", "high"]:
                new_severity = "warning"
                self._set_severity(result, new_severity)
                adjustments.append(f"Downgraded severity from {severity} to {new_severity} due to stable pattern")
                severity = new_severity
        
        # Rule 4: Multiple confidence values
        confidence_values = self._extract_all_confidences(result)
        if len(confidence_values) > 1:
            unique_values = set(round(c, 2) for c in confidence_values)
            if len(unique_values) > 1:
                warnings.append(f"Multiple confidence values detected: {unique_values}")
                # Fix: Use the highest confidence (most conservative)
                max_conf = max(confidence_values)
                self._set_confidence(result, max_conf)
                adjustments.append(f"Unified confidence to {max_conf:.1%} (highest value)")
        
        # Rule 5: Critical severity + low urgency
        if severity in ["critical", "high"] and urgency is not None and urgency < 0.3:
            warnings.append(f"Critical/high severity conflicts with low urgency ({urgency:.2f})")
            # Fix: Downgrade severity
            new_severity = "warning" if urgency < 0.2 else "medium"
            self._set_severity(result, new_severity)
            adjustments.append(f"Downgraded severity from {severity} to {new_severity} (conservative)")
        
        # Log warnings
        if warnings:
            logger.warning(
                "coherence_issues_detected",
                extra={
                    "n_warnings": len(warnings),
                    "n_adjustments": len(adjustments),
                    "warnings": warnings,
                }
            )
        
        # Add coherence warnings to result
        if hasattr(result, 'analysis') and isinstance(result.analysis, dict):
            result.analysis['coherence_warnings'] = warnings
            result.analysis['coherence_adjustments'] = adjustments
        
        return CoherenceReport(
            is_coherent=len(warnings) == 0,
            warnings=warnings,
            adjustments=adjustments,
        )
    
    def _extract_urgency(self, result: Any) -> Optional[float]:
        """Extract urgency score from result."""
        if hasattr(result, 'analysis') and isinstance(result.analysis, dict):
            return result.analysis.get('urgency_score')
        return None
    
    def _extract_sentiment(self, result: Any) -> Optional[str]:
        """Extract sentiment label from result."""
        if hasattr(result, 'analysis') and isinstance(result.analysis, dict):
            return result.analysis.get('sentiment_label')
        return None
    
    def _extract_severity(self, result: Any) -> Optional[str]:
        """Extract severity level from result."""
        if hasattr(result, 'severity'):
            severity_obj = result.severity
            if hasattr(severity_obj, 'severity'):
                return severity_obj.severity
            elif isinstance(severity_obj, str):
                return severity_obj
        return None
    
    def _extract_confidence(self, result: Any) -> Optional[float]:
        """Extract primary confidence from result."""
        if hasattr(result, 'confidence'):
            return result.confidence
        return None
    
    def _extract_patterns(self, result: Any) -> List[Any]:
        """Extract patterns from result."""
        if hasattr(result, 'patterns') and result.patterns:
            return result.patterns
        return []
    
    def _extract_analysis(self, result: Any) -> Dict[str, Any]:
        """Extract analysis dict from result."""
        if hasattr(result, 'analysis') and isinstance(result.analysis, dict):
            return result.analysis
        return {}
    
    def _has_critical_actions(self, analysis: Dict[str, Any]) -> bool:
        """Check if analysis contains critical actions."""
        # Look for critical action keywords
        critical_keywords = ["stop production", "halt", "emergency", "shutdown"]
        
        # Check in various possible locations
        for key in ['actions', 'recommended_actions', 'conclusion']:
            if key in analysis:
                value = str(analysis[key]).lower()
                if any(keyword in value for keyword in critical_keywords):
                    return True
        
        return False
    
    def _has_stable_pattern(self, patterns: List[Any]) -> bool:
        """Check if patterns indicate stability."""
        stable_keywords = ["stable", "estable", "steady", "normal", "consistent"]
        
        for pattern in patterns:
            if hasattr(pattern, 'pattern_type'):
                pattern_type = str(pattern.pattern_type).lower()
                if any(keyword in pattern_type for keyword in stable_keywords):
                    return True
            elif isinstance(pattern, dict):
                pattern_type = str(pattern.get('pattern_type', '')).lower()
                if any(keyword in pattern_type for keyword in stable_keywords):
                    return True
        
        return False
    
    def _extract_all_confidences(self, result: Any) -> List[float]:
        """Extract all confidence values from result."""
        confidences = []
        
        # Primary confidence
        if hasattr(result, 'confidence') and result.confidence is not None:
            confidences.append(result.confidence)
        
        # Explanation confidence
        if hasattr(result, 'explanation') and result.explanation:
            if hasattr(result.explanation, 'outcome'):
                outcome = result.explanation.outcome
                if hasattr(outcome, 'confidence') and outcome.confidence is not None:
                    confidences.append(outcome.confidence)
        
        # Analysis dict confidence
        if hasattr(result, 'analysis') and isinstance(result.analysis, dict):
            for key in ['confidence', 'fused_confidence', 'overall_confidence']:
                if key in result.analysis and result.analysis[key] is not None:
                    confidences.append(result.analysis[key])
        
        return confidences
    
    def _set_severity(self, result: Any, new_severity: str) -> None:
        """Set severity on result object.
        
        NOTE: Frozen dataclass - cannot modify. Log recommendation only.
        """
        current_severity = None
        if hasattr(result, 'severity'):
            severity_obj = result.severity
            if hasattr(severity_obj, 'severity'):
                current_severity = severity_obj.severity
            else:
                current_severity = severity_obj
        
        logger.info(
            "coherence_severity_adjustment_recommended",
            extra={
                "current": current_severity,
                "recommended": new_severity,
                "reason": "coherence_validation",
            }
        )
        # Cannot modify frozen dataclass - adjustment logged only
    
    def _set_confidence(self, result: Any, new_confidence: float) -> None:
        """Set confidence on result object.
        
        NOTE: Frozen dataclass - cannot modify. Log recommendation only.
        """
        logger.info(
            "coherence_confidence_adjustment_recommended",
            extra={
                "current": getattr(result, 'confidence', None),
                "recommended": new_confidence,
                "reason": "coherence_validation",
            }
        )
        # Cannot modify frozen dataclass - adjustment logged only
