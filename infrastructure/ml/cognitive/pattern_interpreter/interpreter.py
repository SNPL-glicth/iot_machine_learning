"""PatternInterpreter - Main class for human-readable pattern interpretation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .text_patterns import interpret_text_patterns
from .numeric_patterns import interpret_numeric_patterns
from .types import InterpretedPattern

logger = logging.getLogger(__name__)


class PatternInterpreter:
    """Human-readable interpreter of detected patterns."""
    
    def __init__(self) -> None:
        self._text_patterns = interpret_text_patterns
        self._numeric_patterns = interpret_numeric_patterns
    
    def interpret(
        self,
        raw_patterns: Dict[str, Any],
        input_type: str,
        domain: str,
        urgency_score: float = 0.0,
        sentiment_label: str = "",
    ) -> List[InterpretedPattern]:
        """Interpret detected patterns into human-readable format."""
        try:
            if input_type == "text":
                return self._text_patterns(raw_patterns, domain, urgency_score, sentiment_label)
            elif input_type == "numeric":
                return self._numeric_patterns(raw_patterns, domain, urgency_score, sentiment_label)
            elif input_type == "universal":
                return self._merge_patterns(raw_patterns, domain, urgency_score, sentiment_label)
            else:
                logger.warning(f"unknown_input_type", extra={"input_type": input_type})
                return []
        except Exception as e:
            logger.error("pattern_interpretation_failed", extra={"input_type": input_type, "domain": domain, "error": str(e)}, exc_info=True)
            return []
    
    def _merge_patterns(self, raw_patterns: Dict[str, Any], domain: str, urgency_score: float, sentiment_label: str) -> List[InterpretedPattern]:
        """Merge text and numeric pattern results for universal input."""
        text_results = self._text_patterns(raw_patterns, domain, urgency_score, sentiment_label)
        numeric_results = self._numeric_patterns(raw_patterns, domain, urgency_score, sentiment_label)
        
        # Deduplicate by pattern_type
        seen_types = set()
        merged = []
        for pattern in text_results + numeric_results:
            if pattern.pattern_type not in seen_types:
                merged.append(pattern)
                seen_types.add(pattern.pattern_type)
        
        # Sort by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        merged.sort(key=lambda p: (severity_order.get(p.severity_hint, 3), -p.confidence))
        return merged
    
    def get_primary_pattern(self, patterns: List[InterpretedPattern]) -> Optional[InterpretedPattern]:
        """Get the most severe pattern from the list."""
        if not patterns:
            return None
        
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        return min(patterns, key=lambda p: (severity_order.get(p.severity_hint, 3), -p.confidence))
    
    def format_for_conclusion(self, patterns: List[InterpretedPattern], domain: str) -> str:
        """Format patterns as human-readable conclusion string."""
        if not patterns:
            return f"No se detectaron patrones significativos en dominio {domain}."
        
        primary = self.get_primary_pattern(patterns)
        if not primary:
            return "Análisis de patrones no disponible."
        
        conclusion_parts = [f"{primary.short_name}: {primary.description}"]
        
        if primary.domain_context:
            conclusion_parts.append(f"Contexto: {primary.domain_context}")
        
        # Add other critical patterns (max 2)
        critical_patterns = [p for p in patterns if p.severity_hint == "critical" and p.pattern_type != primary.pattern_type][:2]
        if critical_patterns:
            conclusion_parts.append("Otros patrones críticos:")
            conclusion_parts.extend(f"- {p.short_name}" for p in critical_patterns)
        
        conclusion_parts.append(f"Confianza: {int(primary.confidence * 100)}%")
        return "\n".join(conclusion_parts)
    
    def get_pattern_summary(self, patterns: List[InterpretedPattern]) -> Dict[str, Any]:
        """Get summary statistics of interpreted patterns."""
        if not patterns:
            return {"total_patterns": 0, "severity_breakdown": {"critical": 0, "warning": 0, "info": 0}, "data_types": [], "primary_pattern": None}
        
        severity_counts = {"critical": 0, "warning": 0, "info": 0}
        data_types = set()
        
        for pattern in patterns:
            severity_counts[pattern.severity_hint] += 1
            data_types.add(pattern.data_type)
        
        return {
            "total_patterns": len(patterns),
            "severity_breakdown": severity_counts,
            "data_types": list(data_types),
            "primary_pattern": self.get_primary_pattern(patterns),
        }
