"""UniversalComparativeEngine — compare current vs historical."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .types import ComparisonContext, ComparisonResult
from .memory_comparator import fetch_similar_from_memory
from .similarity_scorer import compute_similarity_metrics
from .delta_analyzer import build_delta_conclusion, estimate_resolution


logger = logging.getLogger(__name__)


class UniversalComparativeEngine:
    """Compare current analysis vs historical similar incidents.

    Pipeline:
        1. Recall top 3 similar past analyses (via CognitiveMemoryPort)
        2. Compute severity/urgency/topic deltas
        3. Estimate resolution probability and time
        4. Build human-readable comparison conclusion

    Graceful-fail: Returns None if memory unavailable or no matches.
    """

    def compare(
        self,
        ctx: ComparisonContext,
    ) -> Optional[ComparisonResult]:
        """Run comparative analysis.

        Args:
            ctx: ComparisonContext with current result + cognitive memory

        Returns:
            ComparisonResult or None if insufficient data
        """
        if not ctx.cognitive_memory:
            logger.debug("No cognitive_memory provided, skipping comparison")
            return None
        
        try:
            historical = self._fetch_historical(ctx)
            
            if not historical:
                logger.debug("No similar historical incidents found")
                return None
            
            deltas = self._compute_deltas(
                ctx.current_result.analysis, historical
            )
            
            conclusion = self._build_conclusion(
                deltas, historical, ctx.domain
            )
            
            resolution_prob, time_est = self._estimate_resolution(historical)
            
            return ComparisonResult(
                severity_delta_pct=deltas["severity_delta_pct"],
                urgency_delta_pct=deltas["urgency_delta_pct"],
                topic_overlap_pct=deltas["topic_overlap_pct"],
                top_similar=historical,
                delta_conclusion=conclusion,
                resolution_probability=resolution_prob,
                estimated_resolution_time=time_est,
            )
        
        except Exception as e:
            logger.error(f"comparative_analysis_failed: {e}", exc_info=True)
            return None

    def _fetch_historical(
        self,
        ctx: ComparisonContext,
    ) -> List[Dict[str, Any]]:
        """Fetch and parse historical matches."""
        query = self._build_query_from_result(ctx.current_result)
        
        return fetch_similar_from_memory(
            cognitive_memory=ctx.cognitive_memory,
            query=query,
            series_id=ctx.series_id if ctx.series_id != "unknown" else None,
            domain=ctx.domain,
            limit=3,
            min_certainty=0.7,
        )

    def _compute_deltas(
        self,
        current_analysis: Dict[str, Any],
        historical: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute severity/urgency/topic deltas."""
        return compute_similarity_metrics(current_analysis, historical)

    def _build_conclusion(
        self,
        deltas: Dict[str, float],
        historical: List[Dict[str, Any]],
        domain: str,
    ) -> str:
        """Build human-readable comparison."""
        return build_delta_conclusion(
            severity_delta_pct=deltas["severity_delta_pct"],
            urgency_delta_pct=deltas["urgency_delta_pct"],
            topic_overlap_pct=deltas["topic_overlap_pct"],
            top_similar=historical,
            domain=domain,
        )

    def _estimate_resolution(
        self,
        historical: List[Dict[str, Any]],
    ) -> tuple:
        """Estimate resolution probability and time."""
        return estimate_resolution(historical)

    def _build_query_from_result(self, result) -> str:
        """Extract query string from UniversalResult."""
        analysis = result.analysis
        
        if "full_text" in analysis:
            return str(analysis["full_text"])[:500]
        
        if "conclusion" in analysis:
            return str(analysis["conclusion"])[:500]
        
        explanation_dict = result.explanation.to_dict()
        signal = explanation_dict.get("signal", {})
        
        return f"domain:{result.domain} regime:{signal.get('regime', 'unknown')}"
