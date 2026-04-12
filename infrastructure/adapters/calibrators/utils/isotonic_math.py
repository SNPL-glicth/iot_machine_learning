"""Isotonic Regression mathematical utilities.
"""

from typing import List, Tuple

import numpy as np


def pava_isotonic_regression(
    sorted_scores: np.ndarray,
    sorted_outcomes: np.ndarray,
) -> List[Tuple[float, float]]:
    """PAVA: Pool Adjacent Violators Algorithm for isotonic regression.
    
    Args:
        sorted_scores: Scores sorted in ascending order.
        sorted_outcomes: Outcomes corresponding to sorted scores.
        
    Returns:
        List of (score, calibrated_value) tuples representing isotonic function.
    """
    n = len(sorted_scores)
    if n == 0:
        return []
    
    # Initialize blocks
    blocks: List[List[Tuple[float, float]]] = [
        [(float(sorted_scores[i]), float(sorted_outcomes[i]))] for i in range(n)
    ]
    
    # Merge violating blocks
    i = 0
    while i < len(blocks) - 1:
        block1 = blocks[i]
        block2 = blocks[i + 1]
        
        mean1 = sum(x[1] for x in block1) / len(block1)
        mean2 = sum(x[1] for x in block2) / len(block2)
        
        if mean1 > mean2:  # Monotonicity violation
            blocks[i] = block1 + block2
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    
    # Convert to isotonic points
    isotonic_points = []
    for block in blocks:
        if block:
            avg_score = sum(x[0] for x in block) / len(block)
            avg_outcome = sum(x[1] for x in block) / len(block)
            isotonic_points.append((float(avg_score), float(avg_outcome)))
    
    return isotonic_points


def interpolate_isotonic(
    score: float,
    isotonic_function: List[Tuple[float, float]],
) -> float:
    """Linear interpolation in isotonic function.
    
    Args:
        score: Score to interpolate.
        isotonic_function: List of (score, value) tuples.
        
    Returns:
        Interpolated calibrated value.
    """
    if not isotonic_function:
        return score
    
    # Bounds
    if score <= isotonic_function[0][0]:
        return isotonic_function[0][1]
    if score >= isotonic_function[-1][0]:
        return isotonic_function[-1][1]
    
    # Find interval and interpolate
    for i in range(len(isotonic_function) - 1):
        x1, y1 = isotonic_function[i]
        x2, y2 = isotonic_function[i + 1]
        
        if x1 <= score <= x2:
            if abs(x2 - x1) < 1e-10:
                return y1
            t = (score - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
    
    return isotonic_function[-1][1]
