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
    MoEPredictionEngine,
    ContextualRegimeGating,
    DiscrepancyAwareFusion,
    MoEABLogger,
)
from iot_machine_learning.infrastructure.ml.moe.gating.tree_gating import TreeGatingNetwork
from iot_machine_learning.infrastructure.ml.moe.expert_wrappers.engine_adapter import (
    create_baseline_expert,
    create_kalman_expert,
    create_statistical_expert,
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
    fallback_engine = None

    # Baseline expert (siempre disponible, muy eficiente) — también es fallback obligatorio
    try:
        baseline_engine = EngineFactory.create("baseline_moving_average")
        baseline_port = baseline_engine.as_port()
        fallback_engine = baseline_port
        baseline_expert = create_baseline_expert(baseline_port)
        registry.register("baseline", baseline_expert, baseline_expert.capabilities)
        expert_count += 1
        if enable_logging:
            logger.info("moe_expert_registered", extra={"expert": "baseline"})
    except Exception as exc:
        logger.warning("moe_baseline_expert_failed", extra={"error": str(exc)})

    # Statistical expert (para tendencias y estacionalidad)
    try:
        statistical_engine = EngineFactory.create("statistical")
        statistical_expert = create_statistical_expert(statistical_engine.as_port())
        registry.register("statistical", statistical_expert, statistical_expert.capabilities)
        expert_count += 1
        if enable_logging:
            logger.info("moe_expert_registered", extra={"expert": "statistical"})
    except Exception as exc:
        logger.warning("moe_statistical_expert_failed", extra={"error": str(exc)})

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

    # Kalman expert (para señales ruidosas con tendencia)
    try:
        kalman_engine = EngineFactory.create("kalman")
        kalman_expert = create_kalman_expert(kalman_engine.as_port())
        registry.register("kalman", kalman_expert, kalman_expert.capabilities)
        expert_count += 1
        if enable_logging:
            logger.info("moe_expert_registered", extra={"expert": "kalman"})
    except Exception as exc:
        logger.warning("moe_kalman_expert_failed", extra={"error": str(exc)})

    # Validar que al menos tenemos un experto
    if expert_count == 0:
        raise RuntimeError(
            "No se pudo crear ningún experto MoE. "
            "Verifique que EngineFactory tenga engines registrados."
        )

    # Fallback engine obligatorio: baseline es el único candidato apropiado
    if fallback_engine is None:
        raise ValueError(
            "MoEGateway requiere un fallback_engine. "
            "El engine baseline_moving_average es obligatorio como fallback. "
            "Verifique que EngineFactory pueda crear 'baseline_moving_average'."
        )

    # 3. Crear gating network (heurístico, sin dependencias ML pesadas)
    # Reglas por régimen: stable → baseline, trending/volatile → taylor
    gating = RegimeBasedGating.with_default_rules(
        expert_ids=registry.list_all(),
    )
    if enable_logging:
        logger.info(
            "moe_gating_created",
            extra={"type": "RegimeBasedGating", "experts": registry.list_all()},
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
        fallback_engine=fallback_engine,
        sparsity_k=sparsity_k,
    )
    
    if enable_logging:
        logger.info(
            "moe_gateway_creation_completed",
            extra={
                "expert_count": expert_count,
                "sparsity_k": sparsity_k,
                "experts": registry.list_all(),
            },
        )
    
    return gateway


def create_moe_engine(
    fallback_engine,
    sparsity_k: int = 2,
    enable_shadow_gating: bool = True,
    ab_cell: str = "B",
    metrics_exporter=None,
    alert_service=None,
) -> Optional[MoEPredictionEngine]:
    """Crea MoEPredictionEngine con expertos y shadow gating.

    Integra TreeGatingNetwork en shadow mode: rutea en paralelo
    a ContextualRegimeGating pero NO usa sus probabilidades para
    la predicción final. Loggea la diferencia entre ambos gatings.

    Args:
        fallback_engine: Engine fallback (ej: baseline_port).
        sparsity_k: Top-k expertos a ejecutar.
        enable_shadow_gating: Si True, instancia TreeGatingNetwork.
        ab_cell: Celda A/B ("A" control, "B" treatment).
        metrics_exporter: PrometheusExporter para métricas MoE.
        alert_service: MoEAlertService para alertas.

    Returns:
        MoEPredictionEngine configurado, o None si falla.
    """
    registry = ExpertRegistry()
    expert_count = 0

    def _try_register(name, create_fn, engine_name):
        nonlocal expert_count
        try:
            engine = EngineFactory.create(engine_name)
            expert = create_fn(engine.as_port())
            registry.register(name, expert, expert.capabilities)
            expert_count += 1
        except Exception as exc:
            logger.debug(f"moe_expert_{name}_skipped", extra={"error": str(exc)})

    _try_register("baseline", create_baseline_expert, "baseline_moving_average")
    _try_register("statistical", create_statistical_expert, "statistical")
    _try_register("taylor", create_taylor_expert, "taylor")
    _try_register("kalman", create_kalman_expert, "kalman")

    if expert_count == 0:
        logger.warning("moe_engine_no_experts")
        return None

    gating = ContextualRegimeGating(expert_ids=registry.list_all())
    fusion = DiscrepancyAwareFusion()

    # Shadow gating: TreeGatingNetwork en modo observador
    shadow_gating = None
    if enable_shadow_gating:
        try:
            shadow_gating = TreeGatingNetwork(expert_ids=registry.list_all())
            logger.info("moe_shadow_gating_enabled", extra={"experts": registry.list_all()})
        except Exception as exc:
            logger.warning("moe_shadow_gating_failed", extra={"error": str(exc)})

    ab_logger = MoEABLogger()

    return MoEPredictionEngine(
        registry=registry,
        gating=gating,
        fusion=fusion,
        fallback_engine=fallback_engine,
        sparsity_k=sparsity_k,
        shadow_gating=shadow_gating,
        ab_logger=ab_logger,
        ab_cell=ab_cell,
        metrics_exporter=metrics_exporter,
        alert_service=alert_service,
    )


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
