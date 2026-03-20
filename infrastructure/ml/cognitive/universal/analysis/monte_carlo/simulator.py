"""Monte Carlo simulator for uncertainty quantification.

Main simulation engine with probabilistic perturbation and severity classification.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

from ..types import InputType
from iot_machine_learning.domain.services.severity_rules import (
    classify_severity_agnostic,
)

from .types import MonteCarloResult
from .noise_model import get_noise_sigma, perturb_scores
from .statistics import (
    compute_distribution,
    compute_confidence_interval,
    classify_uncertainty,
)
from .scenarios import simulate_future_scenarios

logger = logging.getLogger(__name__)

# Severity score mapping for numeric calculations
SEVERITY_TO_SCORE = {
    "info": 0.0,
    "minor": 0.2,
    "warning": 0.5,
    "high": 0.8,
    "critical": 1.0,
}


class MonteCarloSimulator:
    """Monte Carlo simulator for uncertainty quantification.
    
    Performs probabilistic analysis by perturbing scores and re-classifying
    severity to estimate confidence intervals and future scenarios.
    
    Thread-safe for concurrent use.
    """
    
    def __init__(self, n_simulations: int = 1000):
        """Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of Monte Carlo iterations (default: 1000)
        """
        self._n_simulations = max(100, min(10000, n_simulations))
        
    def simulate(
        self,
        analysis_scores: Dict[str, float],
        input_type: InputType,
        domain: str,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation for uncertainty quantification.
        
        Args:
            analysis_scores: Perception scores from analyzers
                Example: {"urgency": 0.8, "sentiment": -0.6, "confidence": 0.85}
            input_type: Detected input type (TEXT, NUMERIC, MIXED)
            domain: Classified domain (infrastructure, security, etc.)
            
        Returns:
            MonteCarloResult with distribution, confidence interval, scenarios
            
        Raises:
            ValueError: If analysis_scores empty or invalid
        """
        if not analysis_scores:
            raise ValueError("analysis_scores cannot be empty")
        
        t0 = time.monotonic()
        
        # Get noise sigma for this input type
        sigma = get_noise_sigma(input_type)
        
        # Run Monte Carlo iterations
        severity_samples = []
        severity_score_samples = []
        
        for _ in range(self._n_simulations):
            # Perturb scores with Gaussian noise
            perturbed = perturb_scores(analysis_scores, sigma)
            
            # Re-classify severity with perturbed scores
            severity, severity_score = self._classify_perturbed(perturbed)
            
            severity_samples.append(severity)
            severity_score_samples.append(severity_score)
        
        # Compute distribution
        distribution = compute_distribution(severity_samples)
        
        # Compute confidence interval (5th-95th percentile)
        conf_interval = compute_confidence_interval(severity_score_samples)
        
        # Determine expected severity and confidence
        expected_severity = max(distribution, key=distribution.get)
        confidence_score = distribution[expected_severity]
        
        # Classify uncertainty level
        uncertainty_level = classify_uncertainty(severity_score_samples)
        
        # Simulate future scenarios
        scenario_outcomes = simulate_future_scenarios(
            analysis_scores, severity_score_samples, input_type
        )
        
        elapsed_ms = (time.monotonic() - t0) * 1000
        
        logger.debug(
            f"monte_carlo_complete: {self._n_simulations} sims in {elapsed_ms:.1f}ms, "
            f"expected={expected_severity} conf={confidence_score:.2f}",
            extra={"domain": domain, "input_type": input_type.value},
        )
        
        return MonteCarloResult(
            n_simulations=self._n_simulations,
            severity_distribution=distribution,
            confidence_interval=conf_interval,
            expected_severity=expected_severity,
            confidence_score=confidence_score,
            scenario_outcomes=scenario_outcomes,
            uncertainty_level=uncertainty_level,
        )
    
    def _classify_perturbed(
        self,
        perturbed_scores: Dict[str, float],
    ) -> Tuple[str, float]:
        """Classify severity for perturbed scores.
        
        Args:
            perturbed_scores: Perturbed analysis scores
            
        Returns:
            Tuple of (severity_label, severity_score)
        """
        # Use primary score (urgency for text, mean for numeric)
        primary_score = perturbed_scores.get("urgency", 
                        perturbed_scores.get("mean",
                        perturbed_scores.get("confidence", 0.5)))
        
        # Classify using domain-agnostic severity rules
        result = classify_severity_agnostic(
            value=primary_score,
            anomaly=False,  # Monte Carlo focuses on score uncertainty, not anomaly
            threshold=None,
        )
        
        # Map severity to numeric score
        severity_score = SEVERITY_TO_SCORE.get(result.severity, 0.5)
        
        return result.severity, severity_score
