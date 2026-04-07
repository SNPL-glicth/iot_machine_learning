"""Servicio de feedback de plasticidad — aprendizaje online."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..ports.document_analysis import PlasticityPort

logger = logging.getLogger(__name__)


def update_plasticity_from_result(
    result: Any,
    plasticity_port: Optional[PlasticityPort] = None,
) -> None:
    """Actualiza plasticidad basado en resultado de análisis.
    
    Extrae dominio del resultado y delega al port para aprendizaje online.
    
    Args:
        result: Resultado de análisis (AnalysisResult o UniversalResult)
        plasticity_port: Puerto de plasticidad (opcional)
    """
    if plasticity_port is None:
        logger.debug("plasticity_feedback_skipped: no port provided")
        return
    
    try:
        # Extraer dominio del resultado
        domain = _extract_domain(result)
        
        if domain and domain != "unknown":
            plasticity_port.update(domain, result)
            logger.debug(f"plasticity_updated: domain={domain}")
        else:
            logger.debug("plasticity_feedback_skipped: unknown domain")
    
    except Exception as e:
        logger.warning(f"plasticity_feedback_failed: {e}")


def _extract_domain(result: Any) -> str:
    """Extrae dominio del resultado."""
    # Try attribute access
    if hasattr(result, 'domain'):
        return result.domain
    
    # Try signal.domain
    if hasattr(result, 'signal') and hasattr(result.signal, 'domain'):
        return result.signal.domain
    
    # Try dict access
    if isinstance(result, dict):
        return result.get('domain', 'unknown')
    
    return "unknown"
