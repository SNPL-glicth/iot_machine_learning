"""Asymmetric Error Penalty Service.

Applies business-impact-based penalties to prediction errors.
Penalizes critical errors more heavily than normal errors.

Uses the existing Threshold structure:
- warning_low, warning_high: Warning zone boundaries
- critical_low, critical_high: Critical zone boundaries
"""

from __future__ import annotations

from typing import Optional

from ..entities.series.series_context import SeriesContext, Threshold


class AsymmetricPenaltyService:
    """Calculates asymmetric penalties based on business impact.
    
    Penalty rules:
    - Underestimating critical high: 5.0x (dangerous, could miss alerts)
    - Overestimating normal as critical: 2.0x (false alarms)
    - Error in normal zone: 0.5x (low business impact)
    - Trending towards danger: 3.0x (preventive penalty)
    
    Attributes:
        critical_underestimate_penalty: Penalty for missing critical values (default: 5.0)
        false_alarm_penalty: Penalty for false critical alarms (default: 2.0)
        normal_zone_discount: Discount for errors in safe zone (default: 0.5)
        trending_danger_penalty: Penalty for trending towards danger (default: 3.0)
    
    Examples:
        >>> service = AsymmetricPenaltyService()
        >>> # Underestimating a critical value
        >>> penalty = service.compute_penalty(
        ...     predicted=80.0,
        ...     actual=95.0,
        ...     base_error=15.0,
        ...     context=series_context,
        ... )
        >>> penalty > 15.0 * 5.0  # Heavy penalty
        True
    """
    
    def __init__(
        self,
        critical_underestimate_penalty: float = 5.0,
        false_alarm_penalty: float = 2.0,
        normal_zone_discount: float = 0.5,
        trending_danger_penalty: float = 3.0,
    ) -> None:
        """Initialize asymmetric penalty service.
        
        Args:
            critical_underestimate_penalty: Multiplier for underestimating critical values
            false_alarm_penalty: Multiplier for false critical alarms
            normal_zone_discount: Multiplier for errors in normal zone (< 1.0 = discount)
            trending_danger_penalty: Multiplier for trending towards danger
        
        Raises:
            ValueError: If penalties are invalid
        """
        if critical_underestimate_penalty < 1.0:
            raise ValueError(f"critical_underestimate_penalty must be >= 1.0, got {critical_underestimate_penalty}")
        if false_alarm_penalty < 1.0:
            raise ValueError(f"false_alarm_penalty must be >= 1.0, got {false_alarm_penalty}")
        if not 0.0 < normal_zone_discount <= 1.0:
            raise ValueError(f"normal_zone_discount must be in (0, 1], got {normal_zone_discount}")
        if trending_danger_penalty < 1.0:
            raise ValueError(f"trending_danger_penalty must be >= 1.0, got {trending_danger_penalty}")
        
        self.critical_underestimate_penalty = critical_underestimate_penalty
        self.false_alarm_penalty = false_alarm_penalty
        self.normal_zone_discount = normal_zone_discount
        self.trending_danger_penalty = trending_danger_penalty
    
    def compute_penalty(
        self,
        predicted: float,
        actual: float,
        base_error: float,
        context: Optional[SeriesContext] = None,
    ) -> float:
        """Compute asymmetric penalty based on business impact.
        
        Uses Threshold.severity_for() to determine zones.
        
        Algorithm:
        1. If no context/threshold, return base_error (uniform penalty)
        2. Get severity zones for predicted and actual values
        3. Apply penalty based on severity mismatch:
           - Underestimated critical: 5x penalty
           - Overestimated normal as critical: 2x penalty
           - Both in normal zone: 0.5x discount
           - Trending towards danger: 3x penalty
        
        Args:
            predicted: Predicted value
            actual: Actual observed value
            base_error: Base error (|predicted - actual|)
            context: Series context with threshold (optional)
        
        Returns:
            Penalized error (base_error * penalty_multiplier)
        
        Examples:
            >>> service = AsymmetricPenaltyService()
            >>> # No context: uniform penalty
            >>> penalty = service.compute_penalty(10.0, 15.0, 5.0, None)
            >>> penalty == 5.0
            True
        """
        # No context or threshold: uniform penalty
        if context is None or context.threshold is None:
            return base_error
        
        threshold = context.threshold
        
        # Get severity zones
        actual_severity = threshold.severity_for(actual)
        predicted_severity = threshold.severity_for(predicted)
        
        # Case 1: Underestimated critical value (most dangerous)
        if actual_severity == "critical" and predicted_severity != "critical":
            return base_error * self.critical_underestimate_penalty
        
        # Case 2: Overestimated normal as critical (false alarm)
        if predicted_severity == "critical" and actual_severity == "normal":
            return base_error * self.false_alarm_penalty
        
        # Case 3: Both in normal zone (low impact)
        if actual_severity == "normal" and predicted_severity == "normal":
            return base_error * self.normal_zone_discount
        
        # Case 4: Trending towards danger
        if self._is_trending_towards_danger(predicted, actual, threshold):
            return base_error * self.trending_danger_penalty
        
        # Default: uniform penalty
        return base_error
    
    def _is_trending_towards_danger(
        self,
        predicted: float,
        actual: float,
        threshold: Threshold,
    ) -> bool:
        """Check if actual value is trending towards critical zone.
        
        Args:
            predicted: Predicted value
            actual: Actual value
            threshold: Threshold with critical boundaries
        
        Returns:
            True if actual is moving towards critical zone compared to predicted
        """
        # Trending up towards critical high
        if threshold.critical_high is not None:
            if actual > predicted and actual > threshold.critical_high * 0.8:
                return True
        
        # Trending down towards critical low
        if threshold.critical_low is not None:
            if actual < predicted and actual < threshold.critical_low * 1.2:
                return True
        
        return False
