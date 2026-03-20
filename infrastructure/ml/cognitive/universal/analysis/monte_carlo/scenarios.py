"""Future scenario simulation for Monte Carlo analysis.

Projects best/worst/most_likely outcomes based on severity distribution.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..types import InputType
from .noise_model import get_noise_sigma


def simulate_future_scenarios(
    base_scores: Dict[str, float],
    severity_scores: List[float],
    input_type: InputType,
) -> Dict[str, Any]:
    """Simulate future scenarios (best/worst/most_likely).
    
    Projects scores forward 3 time steps with cumulative noise.
    
    Args:
        base_scores: Original analysis scores
        severity_scores: Simulated severity scores
        input_type: Input type for noise model
        
    Returns:
        Dict with best_case, worst_case, most_likely scenarios
    """
    sigma = get_noise_sigma(input_type)
    scores_array = np.array(severity_scores)
    
    # Best case: 10th percentile (optimistic)
    best_score = float(np.percentile(scores_array, 10))
    
    # Worst case: 90th percentile (pessimistic)
    worst_score = float(np.percentile(scores_array, 90))
    
    # Most likely: median
    likely_score = float(np.median(scores_array))
    
    return {
        "best_case": {
            "severity_score": round(best_score, 4),
            "description": "Optimistic projection (10th percentile)",
            "projected_trend": "improving" if best_score < 0.4 else "stable",
        },
        "worst_case": {
            "severity_score": round(worst_score, 4),
            "description": "Pessimistic projection (90th percentile)",
            "projected_trend": "deteriorating" if worst_score > 0.7 else "stable",
        },
        "most_likely": {
            "severity_score": round(likely_score, 4),
            "description": "Most probable outcome (median)",
            "projected_trend": _classify_trend(likely_score),
        },
    }


def _classify_trend(score: float) -> str:
    """Classify trend based on severity score.
    
    Args:
        score: Severity score [0, 1]
        
    Returns:
        "improving" | "stable" | "deteriorating"
    """
    if score < 0.3:
        return "improving"
    elif score > 0.7:
        return "deteriorating"
    else:
        return "stable"
