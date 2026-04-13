"""Checkpoint operations for Bayesian weight tracker.

Export/import state for gossip protocol and disaster recovery.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from iot_machine_learning.domain.value_objects.plasticity_scope import PlasticityScope
from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import GaussianPrior

logger = logging.getLogger(__name__)


class WeightTrackerCheckpoint:
    """Checkpoint operations for weight tracking state.
    
    Format designed for:
    1. Gossip protocol (Prompt C) - nodes exchange checkpoints
    2. Kubernetes pod migration - save before shutdown
    3. Debugging and auditing
    """
    
    CURRENT_VERSION = "1.0"
    
    @staticmethod
    def export(
        scope: Optional[PlasticityScope],
        accuracy: Dict[str, Dict[str, float]],
        priors: Dict[str, Dict[str, GaussianPrior]],
        regime_last_access: Dict[str, float],
        regime_last_update: Dict[str, float],
        alpha: float,
        min_weight: float,
    ) -> dict:
        """Export current state as serializable checkpoint.
        
        Returns:
            Dict with schema version 1.0
        """
        regimes_data: Dict[str, Any] = {}
        total_engines = 0
        
        for regime, engines in accuracy.items():
            regime_engines = {}
            for engine_name, acc in engines.items():
                prior = priors[regime].get(engine_name)
                if prior is None:
                    continue
                
                regime_engines[engine_name] = {
                    "accuracy": acc,
                    "prior_mu": getattr(prior, 'mu_0', acc),
                    "prior_sigma2": getattr(prior, 'sigma2_0', 1.0),
                    "last_access": regime_last_access.get(regime, 0.0),
                    "last_update": regime_last_update.get(regime, 0.0),
                }
                total_engines += 1
            
            if regime_engines:
                regimes_data[regime] = {
                    "engines": regime_engines,
                    "last_access": regime_last_access.get(regime, 0.0),
                    "last_update": regime_last_update.get(regime, 0.0),
                }
        
        return {
            "version": PlasticityCheckpoint.CURRENT_VERSION,
            "scope": {
                "domain": scope.domain if scope else None,
                "is_default": scope.is_default if scope else True,
            },
            "timestamp": time.time(),
            "regimes": regimes_data,
            "metadata": {
                "regime_count": len(regimes_data),
                "engine_count": total_engines,
                "alpha": alpha,
                "min_weight": min_weight,
            }
        }
    
    @staticmethod
    def restore(
        checkpoint_data: dict,
        scope: Optional[PlasticityScope],
        accuracy: Dict[str, Dict[str, float]],
        priors: Dict[str, Dict[str, GaussianPrior]],
        regime_last_access: Dict[str, float],
        regime_last_update: Dict[str, float],
    ) -> None:
        """Restore state from a checkpoint.
        
        Populates the provided dictionaries in-place.
        
        Raises:
            ValueError: If checkpoint version is incompatible.
        """
        version = checkpoint_data.get("version", "unknown")
        if version != PlasticityCheckpoint.CURRENT_VERSION:
            raise ValueError(f"Incompatible checkpoint version: {version}")
        
        # Validate scope matches
        scope_info = checkpoint_data.get("scope", {})
        checkpoint_domain = scope_info.get("domain")
        my_domain = scope.domain if scope else None
        
        if checkpoint_domain != my_domain:
            logger.warning(
                "checkpoint_scope_mismatch",
                extra={"checkpoint_domain": checkpoint_domain, "my_domain": my_domain},
            )
        
        regimes = checkpoint_data.get("regimes", {})
        restored_regimes = 0
        restored_engines = 0
        
        for regime, regime_data in regimes.items():
            engines = regime_data.get("engines", {})
            
            for engine_name, engine_data in engines.items():
                accuracy[regime][engine_name] = engine_data["accuracy"]
                priors[regime][engine_name] = GaussianPrior(
                    mu_0=engine_data["prior_mu"],
                    sigma2_0=engine_data["prior_sigma2"],
                )
                restored_engines += 1
            
            regime_last_access[regime] = regime_data.get("last_access", 0.0)
            regime_last_update[regime] = regime_data.get("last_update", 0.0)
            restored_regimes += 1
        
        logger.info(
            "plasticity_checkpoint_restored",
            extra={
                "regimes": restored_regimes,
                "engines": restored_engines,
                "timestamp": checkpoint_data.get("timestamp"),
            },
        )
