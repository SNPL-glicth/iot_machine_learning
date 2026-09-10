"""Dominio de predicción ZENIN Market (FASE 3).

Ciclo: Prediction -> Outcome -> Evaluation -> Reward, puro e inmutable.
Solo ``EVALUATED -> REWARDED`` materializa reward que alimenta el
aprendizaje posterior (ver ``lifecycle``).
"""

from .distribution import (
    DISTRIBUTIONAL_HORIZONS,
    QUANTILE_LEVELS,
    ReturnDistribution,
)
from .distribution_metrics import (
    DistributionEvaluation,
    crps_gaussian,
    evaluate_distribution,
    pinball_loss,
)
from .evaluation import Evaluation, evaluate_prediction
from .lifecycle import (
    InvalidTransitionError,
    PredictionStatus,
    is_terminal,
    validate_transition,
)
from .outcome import Outcome
from .prediction import Prediction
from .reward import Reward, RewardConfig, compute_reward
from .types import InputContext, PredictionInterval, Regime

__all__ = [
    "PredictionStatus",
    "InvalidTransitionError",
    "is_terminal",
    "validate_transition",
    "Regime",
    "PredictionInterval",
    "InputContext",
    "ReturnDistribution",
    "DISTRIBUTIONAL_HORIZONS",
    "QUANTILE_LEVELS",
    "DistributionEvaluation",
    "pinball_loss",
    "crps_gaussian",
    "evaluate_distribution",
    "Prediction",
    "Outcome",
    "Evaluation",
    "evaluate_prediction",
    "Reward",
    "RewardConfig",
    "compute_reward",
]
