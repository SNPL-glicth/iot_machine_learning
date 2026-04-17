"""Controlled feature activation utilities.

Safe, incremental activation of features with rollback capability.
Restriction: < 180 lines.
"""
from __future__ import annotations
import logging
import os
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ActivationState:
    """Current activation state for progressive rollout."""
    moe_enabled_globally: bool = False
    moe_whitelist: Set[int] = None
    coherence_check_enabled: bool = False
    decision_arbiter_enabled: bool = False
    
    def __post_init__(self):
        if self.moe_whitelist is None:
            self.moe_whitelist = set()


class FeatureActivator:
    """Manages safe progressive feature activation.
    
    Uses environment variables for configuration (12-factor app).
    All changes require restart (immutable infra principle).
    """
    
    _instance: Optional[FeatureActivator] = None
    
    def __new__(cls) -> FeatureActivator:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._state = ActivationState()
            cls._instance._refresh_from_env()
        return cls._instance
    
    def _refresh_from_env(self) -> None:
        """Load configuration from environment (read-only after startup)."""
        # MoE configuration
        self._state.moe_enabled_globally = os.environ.get("ML_MOE_ENABLED", "").lower() == "true"
        whitelist_str = os.environ.get("ML_BATCH_ENTERPRISE_SENSORS", "")
        if whitelist_str:
            try:
                self._state.moe_whitelist = {int(s.strip()) for s in whitelist_str.split(",") if s.strip()}
            except ValueError:
                logger.error("Invalid ML_BATCH_ENTERPRISE_SENSORS format")
                self._state.moe_whitelist = set()
        
        # Progressive features (Phase 2)
        self._state.coherence_check_enabled = os.environ.get("ML_COHERENCE_CHECK_ENABLED", "").lower() == "true"
        self._state.decision_arbiter_enabled = os.environ.get("ML_DECISION_ARBITER_ENABLED", "").lower() == "true"
        
        logger.info("feature_activation_loaded", extra={
            "moe_global": self._state.moe_enabled_globally,
            "moe_whitelist_count": len(self._state.moe_whitelist),
            "coherence": self._state.coherence_check_enabled,
            "arbiter": self._state.decision_arbiter_enabled,
        })
    
    def should_use_moe_for_sensor(self, sensor_id: int) -> bool:
        """Determines if MoE should be used for a specific sensor.
        
        Priority:
        1. Whitelist (if specified) - ONLY these sensors get MoE
        2. Global flag (if no whitelist)
        
        Safe default: False (baseline only)
        """
        if not self._state.moe_enabled_globally:
            return False
        
        # If whitelist exists, ONLY use MoE for whitelisted sensors
        if self._state.moe_whitelist:
            is_whitelisted = sensor_id in self._state.moe_whitelist
            if is_whitelisted:
                logger.info("moe_whitelisted_sensor", extra={"sensor_id": sensor_id})
            return is_whitelisted
        
        # No whitelist - use global flag for all
        return True
    
    def is_coherence_enabled(self) -> bool:
        """Check if coherence check fix is enabled."""
        return self._state.coherence_check_enabled
    
    def get_activation_report(self) -> Dict[str, Any]:
        """Current activation status for monitoring."""
        return {
            "moe": {
                "enabled_globally": self._state.moe_enabled_globally,
                "whitelist": list(self._state.moe_whitelist) if self._state.moe_whitelist else None,
                "mode": "whitelist" if self._state.moe_whitelist else "global" if self._state.moe_enabled_globally else "disabled",
            },
            "progressive_fixes": {
                "coherence_check": self._state.coherence_check_enabled,
                "decision_arbiter": self._state.decision_arbiter_enabled,
            },
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> list[str]:
        """Generate safe activation recommendations."""
        recs = []
        
        if self._state.moe_enabled_globally and not self._state.moe_whitelist:
            recs.append("CRITICAL: MoE enabled globally without whitelist. Use ML_BATCH_ENTERPRISE_SENSORS for safe rollout.")
        
        if not self._state.coherence_check_enabled:
            recs.append("Enable ML_COHERENCE_CHECK_ENABLED=true for EJE-2 fix validation.")
        
        return recs


def get_activator() -> FeatureActivator:
    """Get singleton FeatureActivator instance."""
    return FeatureActivator()
