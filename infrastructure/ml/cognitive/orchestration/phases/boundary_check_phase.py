"""Boundary Check Phase — MED-1 Refactoring.

Validates that input data is within acceptable domain boundaries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PipelineContext

from ......domain.services.domain_boundary_checker import DomainBoundaryChecker
from ......domain.entities.results.boundary_result import BoundaryResult
from ....interfaces import PredictionResult

logger = logging.getLogger(__name__)


class BoundaryCheckPhase:
    """Phase 0: Domain boundary validation."""
    
    @property
    def name(self) -> str:
        return "boundary_check"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute boundary check if enabled."""
        flags = ctx.flags
        
        if not flags.ML_DOMAIN_BOUNDARY_ENABLED:
            return ctx
        
        try:
            checker = DomainBoundaryChecker()
            noise_ratio = 0.0
            boundary_result = checker.check(
                values=ctx.values,
                timestamps=ctx.timestamps,
                noise_ratio=noise_ratio,
            )
            
            if not boundary_result.within_domain:
                logger.warning("domain_boundary_violation", extra={
                    "series_id": ctx.series_id,
                    "rejection_reason": boundary_result.rejection_reason,
                    "n_points": len(ctx.values),
                })
                # Return early result (this will terminate pipeline)
                return ctx.with_field(
                    boundary_result=boundary_result,
                    is_fallback=True,
                    fallback_reason="out_of_domain",
                )
            
            return ctx.with_field(boundary_result=boundary_result)
            
        except Exception as e:
            logger.debug(f"boundary_check_skipped: {e}")
            return ctx
    
    def create_early_result(self, ctx: PipelineContext) -> PredictionResult:
        """Create early out-of-domain result."""
        boundary = ctx.boundary_result
        return PredictionResult(
            predicted_value=None,
            confidence=0.0,
            trend="unknown",
            metadata={
                "is_out_of_domain": True,
                "rejection_reason": boundary.rejection_reason,
                "boundary_check": {
                    "within_domain": False,
                    "rejection_reason": boundary.rejection_reason,
                    "data_quality_score": 0.0,
                    "warnings": [],
                },
            },
        )
