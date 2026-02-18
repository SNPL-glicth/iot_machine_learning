"""Tests for AsymmetricPenaltyService.

Validates:
- Penalty multipliers for different error scenarios
- Critical underestimation (5x penalty)
- False alarms (2x penalty)
- Normal zone discount (0.5x)
- Trending towards danger (3x penalty)
"""

import pytest

from iot_machine_learning.domain.entities.series.series_context import (
    SeriesContext,
    Threshold,
)
from iot_machine_learning.domain.services.asymmetric_penalty_service import (
    AsymmetricPenaltyService,
)


class TestAsymmetricPenaltyServiceInitialization:
    """Test service initialization and validation."""
    
    def test_default_initialization(self) -> None:
        """Test default penalty multipliers."""
        service = AsymmetricPenaltyService()
        assert service.critical_underestimate_penalty == 5.0
        assert service.false_alarm_penalty == 2.0
        assert service.normal_zone_discount == 0.5
        assert service.trending_danger_penalty == 3.0
    
    def test_custom_initialization(self) -> None:
        """Test custom penalty multipliers."""
        service = AsymmetricPenaltyService(
            critical_underestimate_penalty=10.0,
            false_alarm_penalty=3.0,
            normal_zone_discount=0.3,
            trending_danger_penalty=4.0,
        )
        assert service.critical_underestimate_penalty == 10.0
        assert service.false_alarm_penalty == 3.0
        assert service.normal_zone_discount == 0.3
        assert service.trending_danger_penalty == 4.0
    
    def test_invalid_critical_penalty_raises(self) -> None:
        """Test that critical penalty < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="critical_underestimate_penalty"):
            AsymmetricPenaltyService(critical_underestimate_penalty=0.5)
    
    def test_invalid_normal_discount_raises(self) -> None:
        """Test that normal discount > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="normal_zone_discount"):
            AsymmetricPenaltyService(normal_zone_discount=1.5)


class TestCriticalUnderestimation:
    """Test heavy penalty for underestimating critical values."""
    
    def test_underestimate_critical_high(self) -> None:
        """Underestimating a critical high value should get 5x penalty."""
        service = AsymmetricPenaltyService()
        
        # Critical threshold: value > 90
        threshold = Threshold(
            warning_low=None,
            warning_high=70.0,
            critical_low=None,
            critical_high=90.0,
        )
        context = SeriesContext(
            domain_name="iot",
            entity_type="temperature_sensor",
            entity_id="sensor_1",
            threshold=threshold,
        )
        
        # Predicted 85, actual 95 (underestimated critical)
        penalty = service.compute_penalty(
            predicted=85.0,
            actual=95.0,
            base_error=10.0,
            context=context,
        )
        
        # Should apply 5x penalty
        assert penalty == 10.0 * 5.0
    
    def test_underestimate_critical_low(self) -> None:
        """Underestimating a critical low value should get 5x penalty."""
        service = AsymmetricPenaltyService()
        
        # Critical threshold: value < 10
        threshold = Threshold(
            warning_low=20.0,
            warning_high=None,
            critical_low=10.0,
            critical_high=None,
        )
        context = SeriesContext(
            domain_name="iot",
            entity_type="temperature_sensor",
            entity_id="sensor_1",
            threshold=threshold,
        )
        
        # Predicted 15, actual 5 (underestimated critical low)
        penalty = service.compute_penalty(
            predicted=15.0,
            actual=5.0,
            base_error=10.0,
            context=context,
        )
        
        # Should apply 5x penalty
        assert penalty == 10.0 * 5.0
    
    def test_underestimate_out_of_range(self) -> None:
        """Underestimating out-of-range critical should get 5x penalty."""
        service = AsymmetricPenaltyService()
        
        # Critical threshold: value outside [20, 80]
        threshold = Threshold(
            warning_low=30.0,
            warning_high=70.0,
            critical_low=20.0,
            critical_high=80.0,
        )
        context = SeriesContext(
            domain_name="iot",
            entity_type="temperature_sensor",
            entity_id="sensor_1",
            threshold=threshold,
        )
        
        # Predicted 50, actual 95 (underestimated critical high)
        penalty = service.compute_penalty(
            predicted=50.0,
            actual=95.0,
            base_error=45.0,
            context=context,
        )
        
        # Should apply 5x penalty
        assert penalty == 45.0 * 5.0


class TestFalseAlarmPenalty:
    """Test penalty for false critical alarms."""
    
    def test_overestimate_normal_as_critical(self) -> None:
        """Overestimating normal value as critical should get 2x penalty."""
        service = AsymmetricPenaltyService()
        
        # Critical: > 90, Warning: > 70
        threshold = Threshold(
            warning_low=None,
            warning_high=70.0,
            critical_low=None,
            critical_high=90.0,
        )
        context = SeriesContext(
            domain_name="iot",
            entity_type="temperature_sensor",
            entity_id="sensor_1",
            threshold=threshold,
        )
        
        # Predicted 95, actual 60 (false critical alarm)
        penalty = service.compute_penalty(
            predicted=95.0,
            actual=60.0,
            base_error=35.0,
            context=context,
        )
        
        # Should apply 2x penalty
        assert penalty == 35.0 * 2.0


class TestNormalZoneDiscount:
    """Test discount for errors in normal (safe) zone."""
    
    def test_both_in_normal_zone(self) -> None:
        """Errors in normal zone should get 0.5x discount."""
        service = AsymmetricPenaltyService()
        
        # Critical: > 90, Warning: > 70
        threshold = Threshold(
            warning_low=None,
            warning_high=70.0,
            critical_low=None,
            critical_high=90.0,
        )
        context = SeriesContext(
            domain_name="iot",
            entity_type="temperature_sensor",
            entity_id="sensor_1",
            threshold=threshold,
        )
        
        # Predicted 50, actual 55 (both in normal zone)
        penalty = service.compute_penalty(
            predicted=50.0,
            actual=55.0,
            base_error=5.0,
            context=context,
        )
        
        # Should apply 0.5x discount
        assert penalty == 5.0 * 0.5
    
    def test_normal_zone_without_warning_threshold(self) -> None:
        """Normal zone should work even without warning threshold."""
        service = AsymmetricPenaltyService()
        
        # Only critical threshold: > 90
        threshold = Threshold(
            warning_low=None,
            warning_high=None,
            critical_low=None,
            critical_high=90.0,
        )
        context = SeriesContext(
            domain_name="iot",
            entity_type="temperature_sensor",
            entity_id="sensor_1",
            threshold=threshold,
        )
        
        # Predicted 50, actual 60 (both below critical)
        penalty = service.compute_penalty(
            predicted=50.0,
            actual=60.0,
            base_error=10.0,
            context=context,
        )
        
        # Should apply 0.5x discount (normal zone)
        assert penalty == 10.0 * 0.5


class TestTrendingTowardsDanger:
    """Test penalty for trending towards critical zone."""
    
    def test_trending_up_towards_critical_high(self) -> None:
        """Trending up towards critical high should get 3x penalty."""
        service = AsymmetricPenaltyService()
        
        # Critical: > 100, Warning: > 80
        threshold = Threshold(
            warning_low=None,
            warning_high=80.0,
            critical_low=None,
            critical_high=100.0,
        )
        context = SeriesContext(
            domain_name="iot",
            entity_type="temperature_sensor",
            entity_id="sensor_1",
            threshold=threshold,
        )
        
        # Predicted 75, actual 85 (trending up, actual in warning zone and > 80% of critical)
        penalty = service.compute_penalty(
            predicted=75.0,
            actual=85.0,
            base_error=10.0,
            context=context,
        )
        
        # Should apply 3x penalty (trending towards danger)
        assert penalty == 10.0 * 3.0


class TestNoContextFallback:
    """Test fallback to uniform penalty when no context available."""
    
    def test_no_context_uniform_penalty(self) -> None:
        """No context should return base error (uniform penalty)."""
        service = AsymmetricPenaltyService()
        
        penalty = service.compute_penalty(
            predicted=50.0,
            actual=60.0,
            base_error=10.0,
            context=None,
        )
        
        # Should return base error unchanged
        assert penalty == 10.0
    
    def test_no_threshold_uniform_penalty(self) -> None:
        """Context without threshold should return base error."""
        service = AsymmetricPenaltyService()
        
        # Context without threshold
        context = SeriesContext(
            domain_name="iot",
            entity_type="temperature_sensor",
            entity_id="sensor_1",
            threshold=None,
        )
        
        penalty = service.compute_penalty(
            predicted=50.0,
            actual=60.0,
            base_error=10.0,
            context=context,
        )
        
        # Should return base error unchanged
        assert penalty == 10.0
