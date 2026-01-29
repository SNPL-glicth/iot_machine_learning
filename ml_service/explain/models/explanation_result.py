"""Explanation result model for contextual explainer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExplanationResult:
    """Resultado de la explicación contextual."""
    severity: str
    explanation: str
    possible_causes: list[str]
    recommended_action: str
    confidence: float
    source: str  # 'llm', 'template', 'fallback'
    generated_at: datetime
    
    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "explanation": self.explanation,
            "possible_causes": self.possible_causes,
            "recommended_action": self.recommended_action,
            "confidence": self.confidence,
            "source": self.source,
            "generated_at": self.generated_at.isoformat(),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
