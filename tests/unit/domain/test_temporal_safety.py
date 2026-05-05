"""Tests for Temporal Safety Validators."""

import math
import time

import pytest

from iot_machine_learning.domain.validators.temporal_safety import (
    TemporalSafetyValidator,
    validate_temporal_safety,
)


class TestTemporalSafetyValidator:
    """Test temporal safety validation."""
    
    def test_valid_ordered_timestamps(self):
        """Valid ordered timestamps should pass."""
        validator = TemporalSafetyValidator()
        
        timestamps = [1000.0, 1001.0, 1002.0, 1003.0]
        is_valid, violations = validator.validate(timestamps)
        
        assert is_valid is True
        assert len(violations) == 0
    
    def test_nan_timestamp_fails(self):
        """NaN timestamp should fail with CRITICAL."""
        validator = TemporalSafetyValidator()
        
        timestamps = [1000.0, float('nan'), 1002.0]
        is_valid, violations = validator.validate(timestamps)
        
        assert is_valid is False
        assert len(violations) >= 1
        assert violations[0].violation_type == "INVALID_TIMESTAMP"
        assert violations[0].severity == "CRITICAL"
    
    def test_inf_timestamp_fails(self):
        """Inf timestamp should fail with CRITICAL."""
        validator = TemporalSafetyValidator()
        
        timestamps = [1000.0, float('inf'), 1002.0]
        is_valid, violations = validator.validate(timestamps)
        
        assert is_valid is False
        assert violations[0].violation_type == "INVALID_TIMESTAMP"
    
    def test_future_data_fails(self):
        """Future timestamps should fail (data leakage)."""
        validator = TemporalSafetyValidator(allow_future_seconds=10.0)
        
        now = time.time()
        future = now + 100.0  # 100s in future
        timestamps = [now - 10.0, now, future]
        
        is_valid, violations = validator.validate(timestamps)
        
        assert is_valid is False
        assert any(v.violation_type == "FUTURE_DATA" for v in violations)
        assert any(v.severity == "CRITICAL" for v in violations)
    
    def test_future_within_tolerance_passes(self):
        """Future within tolerance should pass (clock skew)."""
        validator = TemporalSafetyValidator(allow_future_seconds=60.0)
        
        now = time.time()
        near_future = now + 30.0  # Within 60s tolerance
        timestamps = [now - 10.0, now, near_future]
        
        is_valid, violations = validator.validate(timestamps)
        
        # Should pass or have only non-critical violations
        assert is_valid is True or not any(v.severity == "CRITICAL" for v in violations)
    
    def test_unordered_timestamps_fails(self):
        """Unordered timestamps should fail."""
        validator = TemporalSafetyValidator()
        
        timestamps = [1000.0, 1002.0, 1001.0]  # Out of order
        is_valid, violations = validator.validate(timestamps, strict=True)
        
        assert is_valid is False
        assert any(v.violation_type == "UNORDERED" for v in violations)
    
    def test_excessive_gap_detected(self):
        """Excessive gaps should be detected."""
        validator = TemporalSafetyValidator(max_gap_seconds=60.0)
        
        timestamps = [1000.0, 1010.0, 1200.0]  # 190s gap
        is_valid, violations = validator.validate(timestamps, strict=True)
        
        assert any(v.violation_type == "EXCESSIVE_GAP" for v in violations)
    
    def test_strict_mode_fails_on_any_violation(self):
        """Strict mode should fail on any violation."""
        validator = TemporalSafetyValidator(max_gap_seconds=10.0)
        
        timestamps = [1000.0, 1005.0, 1020.0]  # 15s gap
        is_valid_strict, _ = validator.validate(timestamps, strict=True)
        is_valid_lenient, _ = validator.validate(timestamps, strict=False)
        
        assert is_valid_strict is False
        assert is_valid_lenient is True  # Only MEDIUM violation
    
    def test_validate_and_raise_raises_on_invalid(self):
        """validate_and_raise should raise ValueError."""
        validator = TemporalSafetyValidator()
        
        timestamps = [1000.0, float('nan')]
        
        with pytest.raises(ValueError, match="CRITICAL"):
            validator.validate_and_raise(timestamps)
    
    def test_validate_and_raise_passes_on_valid(self):
        """validate_and_raise should not raise on valid."""
        validator = TemporalSafetyValidator()
        
        timestamps = [1000.0, 1001.0, 1002.0]
        
        # Should not raise
        validator.validate_and_raise(timestamps)
    
    def test_sanitize_removes_nan(self):
        """Sanitize should remove NaN timestamps."""
        validator = TemporalSafetyValidator()
        
        timestamps = [1000.0, float('nan'), 1002.0]
        values = [10.0, 20.0, 30.0]
        
        clean_ts, clean_vals = validator.sanitize(timestamps, values)
        
        assert len(clean_ts) == 2
        assert math.isnan(clean_ts[0]) is False
        assert math.isnan(clean_ts[1]) is False
    
    def test_sanitize_removes_future_data(self):
        """Sanitize should remove future timestamps."""
        validator = TemporalSafetyValidator(allow_future_seconds=10.0)
        
        now = time.time()
        timestamps = [now - 10.0, now + 100.0, now]
        values = [10.0, 20.0, 30.0]
        
        clean_ts, clean_vals = validator.sanitize(timestamps, values)
        
        assert len(clean_ts) == 2
        assert all(ts <= now + 10.0 for ts in clean_ts)
    
    def test_sanitize_sorts_timestamps(self):
        """Sanitize should sort timestamps."""
        validator = TemporalSafetyValidator()
        
        timestamps = [1002.0, 1000.0, 1001.0]
        values = [30.0, 10.0, 20.0]
        
        clean_ts, clean_vals = validator.sanitize(timestamps, values)
        
        assert clean_ts == [1000.0, 1001.0, 1002.0]
        assert clean_vals == [10.0, 20.0, 30.0]
    
    def test_sanitize_length_mismatch_raises(self):
        """Sanitize should raise on length mismatch."""
        validator = TemporalSafetyValidator()
        
        timestamps = [1000.0, 1001.0]
        values = [10.0]
        
        with pytest.raises(ValueError, match="Length mismatch"):
            validator.sanitize(timestamps, values)
    
    def test_empty_timestamps_valid(self):
        """Empty timestamps should be valid."""
        validator = TemporalSafetyValidator()
        
        is_valid, violations = validator.validate([])
        
        assert is_valid is True
        assert len(violations) == 0
    
    def test_convenience_function(self):
        """Test convenience function."""
        timestamps = [1000.0, 1001.0, 1002.0]
        
        is_valid, violations = validate_temporal_safety(timestamps)
        
        assert is_valid is True
        assert len(violations) == 0
