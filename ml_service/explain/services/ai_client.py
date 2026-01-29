"""AI Explainer client service.

Extracts AI/LLM communication logic from ContextualExplainer.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from ..models.enriched_context import EnrichedContext
from ..models.explanation_result import ExplanationResult

logger = logging.getLogger(__name__)


class AIExplainerClient:
    """Client for communicating with AI Explainer service."""
    
    DEFAULT_TIMEOUT = 2.0  # seconds
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        self._base_url = base_url or os.getenv("AI_EXPLAINER_URL", "http://localhost:8003")
        self._timeout = timeout
    
    async def explain_async(self, context: EnrichedContext) -> Optional[ExplanationResult]:
        """Call AI Explainer service asynchronously."""
        url = f"{self._base_url.rstrip('/')}/explain/anomaly"
        
        payload = {
            "context": "industrial_iot_monitoring",
            "model_output": {
                "metric": context.sensor_type,
                "observed_value": context.predicted_value,
                "expected_range": self._format_expected_range(context),
                "anomaly_score": context.anomaly_score,
                "model": "sklearn_regression_iforest",
                "model_version": "1.0.0",
            },
            "enriched_context": context.to_dict(),
        }
        
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            
            return ExplanationResult(
                severity=data.get("severity", "MEDIUM"),
                explanation=data.get("explanation", ""),
                possible_causes=data.get("possible_causes", []),
                recommended_action=data.get("recommended_action", ""),
                confidence=data.get("confidence", 0.5),
                source="llm",
                generated_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.warning(
                "[AI_CLIENT] AI Explainer call failed: %s",
                str(e),
            )
            return None
    
    def _format_expected_range(self, context: EnrichedContext) -> str:
        """Formatea el rango esperado para el AI Explainer."""
        if context.user_threshold_min is not None and context.user_threshold_max is not None:
            return f"{context.user_threshold_min}-{context.user_threshold_max}"
        if context.user_threshold_min is not None:
            return f">= {context.user_threshold_min}"
        if context.user_threshold_max is not None:
            return f"<= {context.user_threshold_max}"
        if context.recent_min is not None and context.recent_max is not None:
            return f"{context.recent_min:.2f}-{context.recent_max:.2f} (histórico)"
        return "unknown"
