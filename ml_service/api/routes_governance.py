"""Governance Observability Endpoints - FASE-9.

Endpoints for monitoring the governance system:
- /governance/status - Complete governance state
- /governance/parameters - List all registered parameters
- /governance/history - Audit trail of parameter changes
- /governance/reset-convergence - Reset convergence detectors
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/governance", tags=["governance"])


@router.get("/status")
async def get_governance_status(request: Request) -> dict:
    """Return complete governance system status.

    Response includes:
    {
        "registry": {
            "total_parameters": 14,
            "by_category": {...},
            "total_changes": 0
        },
        "convergence": {
            "ML_BAYES_ALPHA": {"status": "converging", ...},
            ...
        },
        "bounds_violations": [...],
        "ensemble": {
            "correlation_threshold": 0.3,
            "decorrelation_threshold": 0.7
        },
        "temperature_scaling": {
            "default_temperature": 1.5,
            "floor": 0.05,
            "ceiling": 0.95
        }
    }
    """
    governance = getattr(request.app.state, "governance", None)
    if not governance:
        raise HTTPException(status_code=503, detail="Governance not initialized")

    try:
        status = governance.get_status()
        return status
    except Exception as e:
        logger.error("governance_status_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/parameters")
async def get_parameters(request: Request) -> dict:
    """List all registered parameters with metadata.

    Returns:
        {
            "parameters": [
                {
                    "name": "ML_BAYES_ALPHA",
                    "category": "LEARNING_RATE",
                    "scope": "REGIME",
                    "current_value": 0.1,
                    "metadata": {...}
                },
                ...
            ]
        }
    """
    governance = getattr(request.app.state, "governance", None)
    if not governance:
        raise HTTPException(status_code=503, detail="Governance not initialized")

    try:
        registry = governance.registry
        parameters = []
        for name, meta in registry._parameters.items():
            parameters.append(
                {
                    "name": name,
                    "category": meta.category.value,
                    "scope": meta.scope.value,
                    "current_value": registry.get_value(name),
                    "description": meta.description,
                }
            )
        return {"parameters": parameters}
    except Exception as e:
        logger.error("governance_parameters_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to get parameters: {str(e)}")


@router.get("/history")
async def get_parameter_history(
    request: Request,
    parameter_name: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """Return audit trail of parameter changes.

    Args:
        parameter_name: Filter by specific parameter name (optional).
        limit: Maximum number of history entries to return (default 50).

    Returns:
        {
            "history": [
                {
                    "parameter_name": "ML_BAYES_ALPHA",
                    "old_value": 0.1,
                    "new_value": 0.08,
                    "timestamp": "...",
                    "reason": "convergence_detected"
                },
                ...
            ]
        }
    """
    governance = getattr(request.app.state, "governance", None)
    if not governance:
        raise HTTPException(status_code=503, detail="Governance not initialized")

    try:
        registry = governance.registry
        history = registry._changes

        # Filter by parameter name if provided
        if parameter_name:
            history = [h for h in history if h.parameter_name == parameter_name]

        # Limit results
        history = history[-limit:] if len(history) > limit else history

        # Format for response
        formatted_history = []
        for change in history:
            formatted_history.append(
                {
                    "parameter_name": change.parameter_name,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "timestamp": change.timestamp.isoformat(),
                    "reason": change.reason,
                }
            )

        return {"history": formatted_history}
    except Exception as e:
        logger.error("governance_history_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.post("/reset-convergence")
async def reset_convergence(request: Request) -> dict:
    """Reset convergence detectors (useful post-deploy).

    Returns:
        {
            "status": "reset_complete",
            "detectors_reset": ["ML_BAYES_ALPHA", ...]
        }
    """
    governance = getattr(request.app.state, "governance", None)
    if not governance:
        raise HTTPException(status_code=503, detail="Governance not initialized")

    try:
        dynamic_tuner = governance.dynamic_tuner
        if hasattr(dynamic_tuner, "_convergence_detector"):
            dynamic_tuner._convergence_detector.reset()
            logger.info("governance_convergence_reset")
            return {
                "status": "reset_complete",
                "detectors_reset": ["ML_BAYES_ALPHA"],
            }
        else:
            return {"status": "no_detectors_to_reset"}
    except Exception as e:
        logger.error("governance_reset_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to reset convergence: {str(e)}")
