"""Governance System Initializer - FASE-9.

Centralized initialization and lifecycle management for all governance components:
- ParameterRegistry (centralized parameter management)
- ParameterBoundsEnforcer (bounds validation)
- DynamicTuner (convergence + adaptive tuning)
- TemperatureScaler (confidence calibration)
- EngineCorrelationAnalyzer (ensemble correlation)
- EnsembleDecorrelator (weight decorrelation)

Single Responsibility: Orchestrate governance component initialization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.parameters.parameter_bounds import (
    BoundsConfig,
    BoundsResult,
    ParameterBoundsEnforcer,
)
from core.parameters.parameter_migration import register_all_parameters
from core.parameters.parameter_registry import ParameterRegistry
from core.tuning.dynamic_tuning import DynamicTuner
from core.tuning.temperature_scaling import TemperatureScaler
from core.ensemble.ensemble_correlation import EngineCorrelationAnalyzer
from core.ensemble.decorrelation import EnsembleDecorrelator
from core.ensemble.ensemble_watchdog import EnsembleWatchdog
from core.ensemble.forced_recovery import ForcedRecoveryManager
from core.ensemble.loop_bounds import LoopBoundsMonitor

logger = logging.getLogger(__name__)


@dataclass
class GovernanceComponents:
    """Container for all initialized governance components."""
    registry: ParameterRegistry
    bounds_enforcer: ParameterBoundsEnforcer
    dynamic_tuner: DynamicTuner
    temperature_scaler: TemperatureScaler
    correlation_analyzer: EngineCorrelationAnalyzer
    decorrelator: EnsembleDecorrelator
    watchdog: EnsembleWatchdog
    recovery_manager: ForcedRecoveryManager
    loop_monitor: LoopBoundsMonitor


class GovernanceInitializer:
    """Initializes the complete governance system in correct order.

    Usage in ml_service/main.py:
        initializer = GovernanceInitializer()
        components = initializer.initialize()
        # Inject components into BayesianWeightTracker, FusePhase, etc.

    Lifecycle:
        1. ParameterRegistry (base dependency)
        2. BoundsEnforcer (depends on registry)
        3. DynamicTuner (depends on registry + bounds)
        4. TemperatureScaler (independent)
        5. CorrelationAnalyzer + Decorrelator (independent)
        6. Watchdog + RecoveryManager (ensemble health)
        7. LoopBoundsMonitor (feedback loop bounds)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)
        self._components: Optional[GovernanceComponents] = None

    def initialize(self) -> GovernanceComponents:
        """Initialize all governance components in dependency order.

        Returns:
            GovernanceComponents with all initialized instances.

        Raises:
            RuntimeError: If initialization fails critically.
        """
        self._logger.info("[GOVERNANCE] Starting initialization...")

        # Step 1: Initialize ParameterRegistry (singleton base)
        self._logger.info("[GOVERNANCE] Step 1/6: ParameterRegistry")
        registry = ParameterRegistry()
        register_all_parameters(registry)
        self._logger.info(
            "[GOVERNANCE] Registry initialized",
            extra={"total_parameters": len(registry._parameters)},
        )

        # Step 2: Initialize BoundsEnforcer
        self._logger.info("[GOVERNANCE] Step 2/6: ParameterBoundsEnforcer")
        bounds_enforcer = ParameterBoundsEnforcer(use_defaults=True)
        self._logger.info(
            "[GOVERNANCE] BoundsEnforcer initialized",
            extra={"default_bounds": len(bounds_enforcer._bounds)},
        )

        # Step 3: Initialize DynamicTuner
        self._logger.info("[GOVERNANCE] Step 3/6: DynamicTuner")
        dynamic_tuner = DynamicTuner(
            bounds_enforcer=bounds_enforcer,
            convergence_window=20,
        )
        self._logger.info("[GOVERNANCE] DynamicTuner initialized")

        # Step 4: Initialize TemperatureScaler
        self._logger.info("[GOVERNANCE] Step 4/9: TemperatureScaler")
        temperature_scaler = TemperatureScaler()
        self._logger.info("[GOVERNANCE] TemperatureScaler initialized")

        # Step 5: Initialize CorrelationAnalyzer
        self._logger.info("[GOVERNANCE] Step 5/9: EngineCorrelationAnalyzer")
        correlation_analyzer = EngineCorrelationAnalyzer()
        self._logger.info("[GOVERNANCE] CorrelationAnalyzer initialized")

        # Step 6: Initialize Decorrelator
        self._logger.info("[GOVERNANCE] Step 6/9: EnsembleDecorrelator")
        decorrelator = EnsembleDecorrelator()
        self._logger.info("[GOVERNANCE] Decorrelator initialized")

        # Step 7: Initialize Watchdog
        self._logger.info("[GOVERNANCE] Step 7/9: EnsembleWatchdog")
        watchdog = EnsembleWatchdog()
        self._logger.info("[GOVERNANCE] Watchdog initialized")

        # Step 8: Initialize RecoveryManager
        self._logger.info("[GOVERNANCE] Step 8/9: ForcedRecoveryManager")
        recovery_manager = ForcedRecoveryManager()
        self._logger.info("[GOVERNANCE] RecoveryManager initialized")

        # Step 9: Initialize LoopBoundsMonitor
        self._logger.info("[GOVERNANCE] Step 9/9: LoopBoundsMonitor")
        loop_monitor = LoopBoundsMonitor()
        self._logger.info("[GOVERNANCE] LoopBoundsMonitor initialized")

        # Assemble components container
        self._components = GovernanceComponents(
            registry=registry,
            bounds_enforcer=bounds_enforcer,
            dynamic_tuner=dynamic_tuner,
            temperature_scaler=temperature_scaler,
            correlation_analyzer=correlation_analyzer,
            decorrelator=decorrelator,
            watchdog=watchdog,
            recovery_manager=recovery_manager,
            loop_monitor=loop_monitor,
        )

        self._logger.info(
            "[GOVERNANCE] Initialization complete",
            extra={"components_initialized": 9},
        )
        return self._components

    def get_status(self) -> dict:
        """Return governance status for /governance/status endpoint.

        Returns:
            Dict with registry summary, convergence report, bounds violations,
            ensemble state, temperature scaling state.
        """
        if not self._components:
            return {"status": "not_initialized"}

        registry = self._components.registry
        dynamic_tuner = self._components.dynamic_tuner
        temperature_scaler = self._components.temperature_scaler

        # Registry summary
        by_category = {}
        for name, meta in registry._parameters.items():
            cat = meta.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        # Convergence report
        convergence_report = {}
        if hasattr(dynamic_tuner, "_convergence_detector"):
            report = dynamic_tuner._convergence_detector.get_report("ML_BAYES_ALPHA")
            if report:
                convergence_report = {
                    "ML_BAYES_ALPHA": {
                        "status": report.status.value,
                        "recommendation": report.recommendation,
                    }
                }

        # Temperature scaling state
        temp_state = {
            "default_temperature": temperature_scaler._default_temperature,
            "floor": temperature_scaler._floor,
            "ceiling": temperature_scaler._ceiling,
        }

        return {
            "registry": {
                "total_parameters": len(registry._parameters),
                "by_category": by_category,
                "total_changes": len(registry._changes),
            },
            "convergence": convergence_report,
            "bounds_violations": [],  # TODO: track violations
            "ensemble": {
                "correlation_threshold": self._components.correlation_analyzer._low_threshold,
                "decorrelation_threshold": self._components.decorrelator._correlation_threshold,
            },
            "temperature_scaling": temp_state,
        }

    def shutdown(self) -> None:
        """Cleanup governance components on service shutdown."""
        if not self._components:
            self._logger.warning("[GOVERNANCE] Shutdown called but not initialized")
            return

        self._logger.info("[GOVERNANCE] Shutting down...")

        # Log final state
        status = self.get_status()
        self._logger.info(
            "[GOVERNANCE] Final state",
            extra={
                "registry_total": status["registry"]["total_parameters"],
                "registry_changes": status["registry"]["total_changes"],
            },
        )

        self._components = None
        self._logger.info("[GOVERNANCE] Shutdown complete")
