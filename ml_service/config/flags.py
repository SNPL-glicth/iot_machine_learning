"""Feature flags model for UTSAE control.

Refactored: Composed from sub-configs per domain (Paso 5).
Todos los flags tienen valores por defecto SEGUROS (desactivados).

Usage:
    flags = FeatureFlags()
    # Access via composed configs:
    flags.ML_TAYLOR_ORDER  # from TaylorConfig
    flags.ML_BATCH_MAX_WORKERS  # from BatchConfig
"""

from __future__ import annotations

from .taylor_config import TaylorConfig
from .batch_config import BatchConfig
from .cognitive_config import CognitiveConfig
from .decision_config import DecisionConfig
from .security_config import SecurityConfig


class FeatureFlags(TaylorConfig, BatchConfig, CognitiveConfig, DecisionConfig, SecurityConfig):
    """Composed feature flags from all sub-configs.
    
    Uses multiple inheritance to combine all configuration domains:
    - TaylorConfig: Taylor/Kalman predictors, engine selection
    - BatchConfig: Batch runners, streaming, ingest circuit breakers
    - CognitiveConfig: Weaviate memory, neural, embeddings
    - DecisionConfig: Decision engine, Bayesian weight tracking
    - SecurityConfig: Auth, CORS, enterprise features
    
    All fields maintain backward compatibility with original flags.py.
    """
    pass  # All fields inherited from parent configs


# Backward compatibility: keep old get_feature_flags if it existed
try:
    from .feature_flags import get_feature_flags as _legacy_get_flags
except ImportError:
    _legacy_get_flags = None


def get_feature_flags() -> FeatureFlags:
    """Factory function for FeatureFlags.
    
    Returns:
        FeatureFlags instance with all defaults.
    """
    return FeatureFlags()
