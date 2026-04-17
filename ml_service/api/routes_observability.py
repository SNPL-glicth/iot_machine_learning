"""Observability routes for runtime validation.

Exposes metrics and safe feature activation endpoints.
Restriction: < 180 lines.
"""
from __future__ import annotations
import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from ..config.feature_flags import get_feature_flags, FeatureFlags
from ..metrics.observability import get_observability
from .dependencies import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/ml/observability/metrics")
async def observability_metrics(_: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Runtime metrics: fallback rate, engine usage, semantic stats, silent failures."""
    obs = get_observability()
    return {
        "observability": obs.to_dict(),
        "feature_flags_active": {
            "ML_MOE_ENABLED": getattr(get_feature_flags(), 'ML_MOE_ENABLED', False),
            "ML_USE_TAYLOR_PREDICTOR": getattr(get_feature_flags(), 'ML_USE_TAYLOR_PREDICTOR', False),
            "ML_BATCH_USE_ENTERPRISE": getattr(get_feature_flags(), 'ML_BATCH_USE_ENTERPRISE', False),
            "ML_COHERENCE_CHECK_ENABLED": getattr(get_feature_flags(), 'ML_COHERENCE_CHECK_ENABLED', False),
        }
    }


@router.get("/ml/observability/fallback-rate")
async def fallback_rate(_: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Enterprise fallback rate. Returns CRITICAL if > 10%."""
    obs = get_observability()
    fallback_metrics = obs.fallback
    total_preds = fallback_metrics.total + obs.engine_usage.baseline + obs.engine_usage.taylor + obs.engine_usage.moe
    rate = obs.get_fallback_rate(total_preds)
    
    status = "ok"
    if rate > 0.20:
        status = "critical"
    elif rate > 0.10:
        status = "warning"
    
    return {
        "fallback_rate": round(rate, 4),
        "total_fallbacks": fallback_metrics.total,
        "affected_sensors": len(fallback_metrics.per_sensor),
        "status": status,
    }


@router.get("/ml/observability/engine-distribution")
async def engine_distribution(_: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Real engine usage distribution in production."""
    obs = get_observability()
    total = obs.engine_usage.baseline + obs.engine_usage.taylor + obs.engine_usage.moe + obs.engine_usage.fallback
    
    if total == 0:
        return {"status": "no_data", "message": "No predictions tracked yet"}
    
    return {
        "total_predictions": total,
        "distribution": {
            "baseline": {"count": obs.engine_usage.baseline, "pct": round(obs.engine_usage.baseline/total, 4)},
            "taylor": {"count": obs.engine_usage.taylor, "pct": round(obs.engine_usage.taylor/total, 4)},
            "moe": {"count": obs.engine_usage.moe, "pct": round(obs.engine_usage.moe/total, 4)},
            "other": {"count": obs.engine_usage.fallback, "pct": round(obs.engine_usage.fallback/total, 4)},
        },
        "per_sensor_sample": dict(list(obs.engine_usage.per_sensor.items())[:10]),
    }


@router.get("/ml/observability/semantic-impact")
async def semantic_impact(_: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Semantic enrichment statistics. Validates if enrichment is running."""
    obs = get_observability()
    sem = obs.semantic
    total = sem.executed + sem.skipped
    
    return {
        "enrichment_rate": round(sem.executed / total, 4) if total > 0 else 0,
        "executed": sem.executed,
        "skipped": sem.skipped,
        "errors": sem.errors,
        "entities_total": sem.entities_total,
        "critical_detected": sem.critical_detected,
        "avg_entities_per_run": round(sem.entities_total / sem.executed, 2) if sem.executed > 0 else 0,
    }


@router.get("/ml/observability/silent-failures")
async def silent_failures(_: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Silent failure detection. Lists locations where exceptions were caught but not escalated."""
    obs = get_observability()
    return {
        "total_locations": len(obs.silent_failures.errors_by_location),
        "failures_by_location": obs.silent_failures.errors_by_location,
        "total_silent_failures": sum(obs.silent_failures.errors_by_location.values()),
    }


@router.get("/ml/activation/status")
async def activation_status(_: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Feature activation status and safety recommendations."""
    from ..activation import get_activator
    return get_activator().get_activation_report()


@router.get("/ml/validation/semantic-impact")
async def validate_semantic_impact(_: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Validates whether SemanticEnrichment affects decisions or only explanations."""
    obs = get_observability()
    sem = obs.semantic
    
    # Analysis of enrichment effectiveness
    if sem.executed == 0:
        return {
            "conclusion": "NO_DATA",
            "message": "Semantic enrichment has not executed yet. Process text inputs first.",
        }
    
    # Calculate metrics
    enrichment_rate = sem.executed / (sem.executed + sem.skipped) if (sem.executed + sem.skipped) > 0 else 0
    critical_detection_rate = sem.critical_detected / sem.executed if sem.executed > 0 else 0
    error_rate = sem.errors / (sem.executed + sem.skipped) if (sem.executed + sem.skipped) > 0 else 0
    
    # Determine conclusion
    conclusion = "UNKNOWN"
    if error_rate > 0.1:
        conclusion = "DEGRADED"
    elif enrichment_rate > 0.8 and critical_detection_rate > 0:
        conclusion = "ACTIVE_WITH_CRITICAL_DETECTION"
    elif enrichment_rate > 0.8:
        conclusion = "ACTIVE_NO_CRITICAL"
    elif enrichment_rate > 0.5:
        conclusion = "PARTIAL"
    else:
        conclusion = "MOSTLY_SKIPPED"
    
    return {
        "conclusion": conclusion,
        "enrichment_rate": round(enrichment_rate, 4),
        "critical_detection_rate": round(critical_detection_rate, 4),
        "error_rate": round(error_rate, 4),
        "metrics": {
            "executed": sem.executed,
            "skipped": sem.skipped,
            "errors": sem.errors,
            "critical_detected": sem.critical_detected,
        },
        "interpretation": {
            "ACTIVE_WITH_CRITICAL_DETECTION": "Enrichment running and detecting critical entities. Affects severity classification.",
            "ACTIVE_NO_CRITICAL": "Enrichment running but no critical entities detected. May affect explanations only.",
            "PARTIAL": "Enrichment partially active. Check feature flags and text lengths.",
            "MOSTLY_SKIPPED": "Enrichment mostly skipped. Inputs may be too short or not text.",
            "DEGRADED": "High error rate in enrichment. Check logs for extraction failures.",
        }.get(conclusion, "Unknown"),
    }
