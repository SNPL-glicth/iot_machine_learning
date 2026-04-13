"""Decision engine and Bayesian weight tracker configuration.

Extracted from flags.py as part of refactoring Paso 5.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class DecisionConfig(BaseModel):
    """Configuration for decision engine and Bayesian weight tracking.
    
    Responsibilities:
    - Decision engine selection and thresholds
    - Bayesian weight tracker tuning
    - Contextual decision amplification
    """

    # --- Decision Engine ---
    ML_ENABLE_DECISION_ENGINE: bool = True
    ML_DECISION_ENGINE: str = "simple"  # simple | contextual | conservative | aggressive | cost_optimized
    ML_DECISION_ENGINE_STRATEGY: str = "simple"  # Deprecated

    # --- Bayesian Weight Tracker tuning ---
    ML_BAYES_ALPHA: float = 0.15
    ML_BAYES_MIN_WEIGHT: float = 0.05
    ML_BAYES_MAX_REGIMES: int = 10
    ML_BAYES_REGIME_TTL_SECONDS: float = 86400.0
    ML_BAYES_NOISE_THRESHOLD: float = 0.3
    ML_BAYES_PERSIST_EVERY_N: int = 10
    ML_BAYES_IMMEDIATE_PERSIST_THRESHOLD: float = 0.15
    ML_BAYES_REDIS_CACHE_TTL_SECONDS: float = 60.0

    # JSON strings for dicts
    ML_BAYES_REGIME_ALPHAS: str = '{"STABLE":0.10,"TRENDING":0.20,"VOLATILE":0.25,"NOISY":0.08,"TRANSITIONAL":0.18}'
    ML_BAYES_LR_FACTORS: str = '{"STABLE":1.0,"TRENDING":1.2,"VOLATILE":1.5,"NOISY":0.8,"UNKNOWN":1.0,"TRANSITIONAL":1.1}'

    # --- Decision engine tuning ---
    ML_DECISION_CONSERVATIVE_THRESHOLD: float = 0.8
    ML_DECISION_CONSERVATIVE_SAFETY_MARGIN: float = 1.2
    ML_DECISION_CONFIDENCE_FLOOR: float = 0.6
    ML_DECISION_CONFIDENCE_CEILING: float = 0.95
    ML_DECISION_ESCALATION_THRESHOLD: int = 5
    ML_DECISION_ATT_STABLE_DRIFT_THRESHOLD: float = 0.10
    ML_DECISION_CONFIDENCE_REDUCTION_SPARSE: float = 0.9

    # JSON for base_scores and amplifier thresholds
    ML_DECISION_BASE_SCORES: str = '{"critical":0.90,"high":0.70,"medium":0.45,"low":0.25,"info":0.05,"warning":0.45}'
    ML_DECISION_AMP_THRESHOLDS: str = '{"count_high":5,"count_medium":3,"ratio_high":0.60,"ratio_low":0.30,"drift_high":0.70,"drift_low":0.40}'

    # --- Contextual Decision Engine Config ---
    ML_DECISION_AMP_CONSECUTIVE_5: float = 1.35
    ML_DECISION_AMP_CONSECUTIVE_3: float = 1.20
    ML_DECISION_AMP_RATE_HIGH: float = 1.20
    ML_DECISION_AMP_RATE_MED: float = 1.10
    ML_DECISION_AMP_VOLATILE: float = 1.15
    ML_DECISION_AMP_NOISY: float = 1.10
    ML_DECISION_AMP_DRIFT_HIGH: float = 1.20
    ML_DECISION_AMP_DRIFT_MED: float = 1.10
    ML_DECISION_ATT_STABLE: float = 0.85
    ML_DECISION_ATT_LOW_CRITICALITY: float = 0.80
    ML_DECISION_ATT_NO_CONTEXT: float = 0.90
    ML_DECISION_SUPPRESSION_WINDOW_MINUTES: float = 5.0

    # --- Contextual Decision Thresholds ---
    ML_DECISION_THRESHOLD_ESCALATE: float = 0.85
    ML_DECISION_THRESHOLD_INVESTIGATE: float = 0.65
    ML_DECISION_THRESHOLD_MONITOR: float = 0.40

    @field_validator("ML_DECISION_ENGINE")
    @classmethod
    def _validate_decision_engine(cls, v: str) -> str:
        """Valida que el motor de decisión sea conocido."""
        allowed = {"simple", "contextual", "conservative", "aggressive", "cost_optimized"}
        if v not in allowed:
            return "simple"
        return v

    @field_validator("ML_DECISION_ENGINE_STRATEGY")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        """Valida que la estrategia sea conocida."""
        allowed = {"simple", "conservative", "aggressive", "cost_optimized"}
        if v not in allowed:
            return "simple"
        return v

    def get_decision_engine(self) -> str:
        """Retorna el motor de decisión activo."""
        return self.ML_DECISION_ENGINE or self.ML_DECISION_ENGINE_STRATEGY or "simple"
