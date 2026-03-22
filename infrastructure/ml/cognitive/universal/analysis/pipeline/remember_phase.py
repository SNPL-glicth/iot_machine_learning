"""Remember phase: recall similar past analyses."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.types import UniversalContext

logger = logging.getLogger(__name__)


class RememberPhase:
    """Phase 3: Recall similar past analyses via CognitiveMemoryPort."""

    def execute(
        self,
        raw_data: Any,
        domain: str,
        ctx: UniversalContext,
        timing: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Execute remember phase.

        Args:
            raw_data: Any input
            domain: Classified domain
            ctx: Pipeline configuration and environment
            timing: Pipeline timing dict

        Returns:
            Recall context dict or None
        """
        t0 = time.monotonic()
        
        recall_ctx = None
        
        if ctx.cognitive_memory:
            try:
                query = str(raw_data)[:500] if isinstance(raw_data, str) else ""
                
                if hasattr(ctx.cognitive_memory, 'recall_similar_explanations'):
                    results = ctx.cognitive_memory.recall_similar_explanations(
                        query=query,
                        series_id=ctx.series_id if ctx.series_id != "unknown" else None,
                        limit=3,
                        min_certainty=0.7,
                    )
                    
                    if results:
                        recall_ctx = {
                            "n_matches": len(results),
                            "top_score": round(results[0].score, 3) if results else 0.0,
                            "has_context": True,
                        }
            except Exception as e:
                logger.debug(f"memory_recall_failed: {e}")
        
        timing["remember"] = (time.monotonic() - t0) * 1000
        
        return recall_ctx
