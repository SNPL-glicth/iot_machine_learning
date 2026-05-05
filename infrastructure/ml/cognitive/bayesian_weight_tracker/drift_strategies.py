"""Drift response strategies for GradualDriftResponse."""
from __future__ import annotations
import logging
from typing import Dict
from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import GaussianPrior

logger = logging.getLogger(__name__)


def _apply_gradual_decay(
    accuracies: Dict[str, float],
    priors: Dict[str, GaussianPrior],
    decay: float,
    expansion: float,
) -> tuple[Dict[str, float], Dict[str, GaussianPrior]]:
    """Default gradual decay (backward compatible)."""
    decayed_acc = {engine: acc * decay for engine, acc in accuracies.items()}
    expanded_priors: Dict[str, GaussianPrior] = {}
    for engine, prior in priors.items():
        mu_0 = prior.get_param("mu_0", 0.5) * decay
        sigma2_0 = prior.get_param("sigma2_0", 1.0) * expansion
        expanded_priors[engine] = GaussianPrior(mu_0=mu_0, sigma2_0=sigma2_0)
    logger.info(
        "drift_gradual_decay_applied",
        extra={
            "action": "gradual_decay",
            "n_engines": len(accuracies),
            "decay_factor": decay,
            "variance_expansion": expansion,
        },
    )
    return decayed_acc, expanded_priors


def _apply_mild_drift_response(
    accuracies: Dict[str, float],
    priors: Dict[str, GaussianPrior],
) -> tuple[Dict[str, float], Dict[str, GaussianPrior]]:
    """Mild drift: gentle decay to preserve learned structure."""
    mild_decay = 0.8
    mild_expansion = 1.5
    decayed_acc = {engine: acc * mild_decay for engine, acc in accuracies.items()}
    expanded_priors: Dict[str, GaussianPrior] = {}
    for engine, prior in priors.items():
        mu_0 = prior.get_param("mu_0", 0.5) * mild_decay
        sigma2_0 = prior.get_param("sigma2_0", 1.0) * mild_expansion
        expanded_priors[engine] = GaussianPrior(mu_0=mu_0, sigma2_0=sigma2_0)
    logger.info(
        "drift_mild_response_applied",
        extra={
            "action": "mild_decay",
            "n_engines": len(accuracies),
            "decay_factor": mild_decay,
            "variance_expansion": mild_expansion,
        },
    )
    return decayed_acc, expanded_priors


def _apply_severe_drift_response(
    accuracies: Dict[str, float],
    priors: Dict[str, GaussianPrior],
    decay: float,
    expansion: float,
) -> tuple[Dict[str, float], Dict[str, GaussianPrior]]:
    """Severe drift: partial reset blending old priors with defaults."""
    blend_alpha = 0.5
    severe_expansion = 3.0
    severe_decay = 0.5
    decayed_acc: Dict[str, float] = {}
    for engine, acc in accuracies.items():
        decayed_acc[engine] = (1.0 - blend_alpha) * acc * severe_decay + blend_alpha * 0.5
    expanded_priors: Dict[str, GaussianPrior] = {}
    default_prior = GaussianPrior(mu_0=0.5, sigma2_0=2.0)
    for engine, prior in priors.items():
        blended = _blend_prior(prior, default_prior, blend_alpha)
        mu_0 = blended.get_param("mu_0", 0.5) * severe_decay
        sigma2_0 = blended.get_param("sigma2_0", 2.0) * severe_expansion
        expanded_priors[engine] = GaussianPrior(mu_0=mu_0, sigma2_0=sigma2_0)
    logger.warning(
        "drift_severe_response_applied",
        extra={
            "action": "severe_partial_reset",
            "n_engines": len(accuracies),
            "blend_alpha": blend_alpha,
            "variance_expansion": severe_expansion,
        },
    )
    return decayed_acc, expanded_priors


def _blend_prior(
    old_prior: GaussianPrior,
    default_prior: GaussianPrior,
    alpha: float,
) -> GaussianPrior:
    """Blend old prior with default prior via convex combination."""
    old_mu = old_prior.get_param("mu_0", 0.5)
    old_sigma2 = old_prior.get_param("sigma2_0", 1.0)
    def_mu = default_prior.get_param("mu_0", 0.5)
    def_sigma2 = default_prior.get_param("sigma2_0", 2.0)
    new_mu = (1.0 - alpha) * old_mu + alpha * def_mu
    old_prec = 1.0 / old_sigma2 if old_sigma2 > 0 else 0.0
    def_prec = 1.0 / def_sigma2 if def_sigma2 > 0 else 0.0
    blended_prec = (1.0 - alpha) * old_prec + alpha * def_prec
    new_sigma2 = 1.0 / blended_prec if blended_prec > 0 else max(old_sigma2, def_sigma2)
    return GaussianPrior(mu_0=new_mu, sigma2_0=new_sigma2)
