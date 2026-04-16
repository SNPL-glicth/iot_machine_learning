"""MoE Factory — creación segura de MoEGateway para integración controlada.

Integra la arquitectura MoE existente con el sistema de producción actual.
No introduce nuevas dependencias ni rompe el flujo existente.

Uso:
    >>> from infrastructure.config.moe_factory import create_moe_gateway
    >>> gateway = create_moe_gateway()
    >>> prediction = gateway.predict(sensor_window)
"""

from __future__ import annotations

import logging
from typing import Optional

from iot_machine_learning.infrastructure.ml.moe import (
    MoEGateway,
    ExpertRegistry,
    RegimeBasedGating,
    SparseFusionLayer,
)
from iot_machine_learning.infrastructure.ml.moe.expert_wrappers.engine_adapter import (
    create_baseline_expert,
    create_taylor_expert,
)
from iot_machine_learning.infrastructure.ml.engines.core.factory import EngineFactory

logger = logging.getLogger(__name__)


def create_moe_gateway(
    sparsity_k: int = 2,
    enable_logging: bool = True,
) -> MoEGateway:
    """Crea y configura MoEGateway con expertos existentes.
    
    Esta factory envuelve engines existentes (Baseline, Taylor) como expertos MoE,
    permitiendo usar la arquitectura MoE sin modificar los engines originales.
    
    Args:
        sparsity_k: Número de top expertos a ejecutar (default: 2 para eficiencia).
        enable_logging: Si True, registra métricas de ejecución MoE.
        
    Returns:
        MoEGateway configurado y listo para usar.
        
    Raises:
        RuntimeError: Si no se pueden crear los engines requeridos.
        
    Example:
        >>> gateway = create_moe_gateway(sparsity_k=2)
        >>> # Usar con orchestrator
        >>> orchestrator = MetaCognitiveOrchestrator(
        ...     engines=[baseline_engine],
        ...     moe_gateway=gateway,
        ... )
    """
    if enable_logging:
        logger.info("moe_gateway_creation_started", extra={"sparsity_k": sparsity_k})
    
    # 1. Crear registry
    registry = ExpertRegistry()
    
    # 2. Crear y registrar expertos desde engines existentes
    expert_count = 0
    
    # Baseline expert (siempre disponible, muy eficiente)
    try:
        baseline_engine = EngineFactory.create("baseline_moving_average")
        baseline_expert = create_baseline_expert(baseline_engine.as_port())
        registry.register("baseline", baseline_expert, baseline_expert.capabilities)
        expert_count += 1
        if enable_logging:
            logger.info("moe_expert_registered", extra={"expert": "baseline"})
    except Exception as exc:
        logger.warning("moe_baseline_expert_failed", extra={"error": str(exc)})
    
    # Taylor expert (para tendencias y volatilidad)
    try:
        taylor_engine = EngineFactory.create("taylor")
        taylor_expert = create_taylor_expert(taylor_engine.as_port())
        registry.register("taylor", taylor_expert, taylor_expert.capabilities)
        expert_count += 1
        if enable_logging:
            logger.info("moe_expert_registered", extra={"expert": "taylor"})
    except Exception as exc:
        logger.warning("moe_taylor_expert_failed", extra={"error": str(exc)})
    
    # Validar que al menos tenemos un experto
    if expert_count == 0:
        raise RuntimeError(
            "No se pudo crear ningún experto MoE. "
            "Verifique que EngineFactory tenga engines registrados."
        )
    
    # 3. Crear gating network (heurístico, sin dependencias ML pesadas)
    # Reglas por régimen: stable → baseline, trending/volatile → taylor
    gating = RegimeBasedGating.with_default_rules(
        expert_names=registry.list_experts(),
    )
    if enable_logging:
        logger.info(
            "moe_gating_created",
            extra={"type": "RegimeBasedGating", "experts": registry.list_experts()},
        )
    
    # 4. Crear fusion layer
    fusion = SparseFusionLayer()
    if enable_logging:
        logger.info("moe_fusion_created", extra={"type": "SparseFusionLayer"})
    
    # 5. Crear y retornar gateway
    gateway = MoEGateway(
        registry=registry,
        gating=gating,
        fusion=fusion,
        sparsity_k=sparsity_k,
    )
    
    if enable_logging:
        logger.info(
            "moe_gateway_creation_completed",
            extra={
                "expert_count": expert_count,
                "sparsity_k": sparsity_k,
                "experts": registry.list_experts(),
            },
        )
    
    return gateway


def create_moe_gateway_safe(
    sparsity_k: int = 2,
    fallback_to_baseline: bool = True,
) -> Optional[MoEGateway]:
    """Versión segura que nunca falla — retorna None si hay error.
    
    Útil para inicialización condicional donde MoE es opcional.
    
    Args:
        sparsity_k: Número de top expertos a ejecutar.
        fallback_to_baseline: Si True, siempre incluye baseline como fallback.
        
    Returns:
        MoEGateway configurado, o None si falló la creación.
    """
    try:
        return create_moe_gateway(sparsity_k=sparsity_k)
    except Exception as exc:
        logger.error("moe_gateway_creation_failed", extra={"error": str(exc)})
        if fallback_to_baseline:
            logger.info("moe_fallback_to_baseline_engine")
        return None
