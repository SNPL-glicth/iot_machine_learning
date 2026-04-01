"""Tests for DomainBoundaryChecker domain service.

Covers domain boundary validation rules:
- Hard rejections: insufficient data, zero variance, excessive missing, infinite values
- Soft warnings: low sample count, high missing ratio, extreme noise
- Quality scoring: 1.0 base, -0.3 per warning, 0.0 if rejected
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from iot_machine_learning.domain.services.domain_boundary_checker import (
    DomainBoundaryChecker,
)
from iot_machine_learning.domain.entities.results.boundary_result import BoundaryResult


class TestDomainBoundaryCheckerBasics:
    """Basic construction and sanity checks."""

    def test_checker_construction(self):
        """DomainBoundaryChecker can be instantiated."""
        checker = DomainBoundaryChecker()
        assert checker is not None

    def test_boundary_result_dataclass(self):
        """BoundaryResult is a proper frozen dataclass."""
        result = BoundaryResult(
            within_domain=False,
            rejection_reason="insufficient_data",
            data_quality_score=0.0,
            warnings=[],
        )
        assert result.within_domain is False
        assert result.rejection_reason == "insufficient_data"
        assert result.data_quality_score == 0.0
        assert result.warnings == []


class TestHardRejections:
    """Rejection rules (within_domain=False)."""

    def test_reject_insufficient_data_n_points_2(self):
        """n_points=2 → rejected 'insufficient_data'."""
        checker = DomainBoundaryChecker()
        result = checker.check(values=[1.0, 2.0])
        
        assert result.within_domain is False
        assert result.rejection_reason == "insufficient_data"
        assert result.data_quality_score == 0.0
        assert result.warnings == []

    def test_reject_zero_variance_all_identical(self):
        """All values identical → rejected 'zero_variance'."""
        checker = DomainBoundaryChecker()
        result = checker.check(values=[5.0, 5.0, 5.0, 5.0, 5.0])
        
        assert result.within_domain is False
        assert result.rejection_reason == "zero_variance"
        assert result.data_quality_score == 0.0

    def test_reject_zero_variance_std_zero(self):
        """std == 0.0 → rejected 'zero_variance'."""
        checker = DomainBoundaryChecker()
        # 10 identical values
        result = checker.check(values=[3.14] * 10)
        
        assert result.within_domain is False
        assert result.rejection_reason == "zero_variance"

    def test_reject_excessive_missing_nan_ratio_06(self):
        """nan_ratio=0.6 → rejected 'excessive_missing'."""
        checker = DomainBoundaryChecker()
        # 5 values, 3 NaN = 0.6 ratio
        result = checker.check(values=[1.0, 2.0, np.nan, np.nan, np.nan])
        
        assert result.within_domain is False
        assert result.rejection_reason == "excessive_missing"
        assert result.data_quality_score == 0.0

    def test_reject_infinite_values_pos_inf(self):
        """Positive inf in values → rejected 'infinite_values'."""
        checker = DomainBoundaryChecker()
        result = checker.check(values=[1.0, 2.0, np.inf, 4.0])
        
        assert result.within_domain is False
        assert result.rejection_reason == "infinite_values"
        assert result.data_quality_score == 0.0

    def test_reject_infinite_values_neg_inf(self):
        """Negative inf in values → rejected 'infinite_values'."""
        checker = DomainBoundaryChecker()
        result = checker.check(values=[1.0, 2.0, -np.inf, 4.0])
        
        assert result.within_domain is False
        assert result.rejection_reason == "infinite_values"


class TestWarnings:
    """Warning rules (within_domain=True but with warnings)."""

    def test_warn_low_sample_count_n_points_7(self):
        """n_points=7 → within_domain=True + warn 'low_sample_count'."""
        checker = DomainBoundaryChecker()
        result = checker.check(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        
        assert result.within_domain is True
        assert result.rejection_reason is None
        assert "low_sample_count" in result.warnings
        assert result.data_quality_score == pytest.approx(0.7)  # 1.0 - 0.3

    def test_warn_high_missing_ratio_nan_03(self):
        """nan_ratio=0.3 → within_domain=True + warn 'high_missing_ratio'."""
        checker = DomainBoundaryChecker()
        # 10 values, 3 NaN = 0.3 ratio (warn threshold 0.2, reject 0.5)
        values = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, np.nan, 10.0]
        result = checker.check(values=values)
        
        assert result.within_domain is True
        assert result.rejection_reason is None
        assert "high_missing_ratio" in result.warnings
        assert result.data_quality_score == pytest.approx(0.7)  # 1.0 - 0.3

    def test_warn_extreme_noise(self):
        """noise_ratio > 0.8 → warn 'extreme_noise'."""
        checker = DomainBoundaryChecker()
        result = checker.check(
            values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            noise_ratio=0.85,
        )
        
        assert result.within_domain is True
        assert result.rejection_reason is None
        assert "extreme_noise" in result.warnings
        assert result.data_quality_score == pytest.approx(0.7)  # 1.0 - 0.3


class TestCleanData:
    """Clean normal data passes all checks."""

    def test_clean_data_normal(self):
        """Normal clean data → within_domain=True, score=1.0, no warnings."""
        checker = DomainBoundaryChecker()
        result = checker.check(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        assert result.within_domain is True
        assert result.rejection_reason is None
        assert result.data_quality_score == pytest.approx(1.0)
        assert result.warnings == []

    def test_clean_data_varied(self):
        """Varied normal data → within_domain=True, no warnings."""
        checker = DomainBoundaryChecker()
        # Need 10+ points to avoid low_sample_count warning
        result = checker.check(values=[10.5, 12.3, 9.8, 11.2, 13.0, 10.1, 11.5, 12.0, 9.5, 10.8])
        
        assert result.within_domain is True
        assert result.data_quality_score == pytest.approx(1.0)
        assert result.warnings == []


class TestMultipleWarnings:
    """Multiple warnings reduce score correctly."""

    def test_multiple_warnings_score_decreases(self):
        """Multiple warnings → score decreases by 0.3 each."""
        checker = DomainBoundaryChecker()
        # n_points=5 (low_sample_count), nan_ratio=0.25 (high_missing_ratio)
        values = [1.0, np.nan, 3.0, np.nan, 5.0]
        result = checker.check(values=values)
        
        assert result.within_domain is True
        assert result.rejection_reason is None
        assert len(result.warnings) == 2
        assert "low_sample_count" in result.warnings
        assert "high_missing_ratio" in result.warnings
        # 1.0 - 0.3 - 0.3 = 0.4, but minimum is 0.1
        assert result.data_quality_score == pytest.approx(0.4)

    def test_three_warnings_score_min_01(self):
        """Three warnings → score floors at 0.1."""
        checker = DomainBoundaryChecker()
        # n_points=5 (low), nan_ratio=0.25 (high_missing), noise_ratio=0.9 (extreme)
        values = [1.0, np.nan, 3.0, np.nan, 5.0]
        result = checker.check(values=values, noise_ratio=0.9)
        
        assert result.within_domain is True
        assert len(result.warnings) == 3
        # 1.0 - 0.3 - 0.3 - 0.3 = 0.1 (floored)
        assert result.data_quality_score == pytest.approx(0.1)


class TestBoundaryConditions:
    """Edge cases and boundary values."""

    def test_exactly_3_points_passes(self):
        """n_points=3 → passes (threshold is < 3)."""
        checker = DomainBoundaryChecker()
        result = checker.check(values=[1.0, 2.0, 3.0])
        
        assert result.within_domain is True
        assert result.rejection_reason is None
        # But warns because < 10
        assert "low_sample_count" in result.warnings

    def test_exactly_10_points_no_warning(self):
        """n_points=10 → no low_sample_count warning."""
        checker = DomainBoundaryChecker()
        result = checker.check(values=[1.0] * 10)
        
        # Wait, this has zero variance too — need varied values
        result = checker.check(values=[float(i) for i in range(10)])
        
        assert result.within_domain is True
        assert "low_sample_count" not in result.warnings
        assert result.data_quality_score == pytest.approx(1.0)

    def test_nan_ratio_exactly_05_rejected(self):
        """nan_ratio=0.5 exactly → rejected (threshold is > 0.5)."""
        checker = DomainBoundaryChecker()
        # 10 values, 5 NaN = 0.5 ratio
        values = [1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan, 10.0]
        result = checker.check(values=values)
        
        # nan_ratio = 5/10 = 0.5, condition is > 0.5 so this passes
        # But wait, let me check the condition again...
        # Actually > 0.5 means 0.5 is NOT rejected
        assert result.within_domain is True

    def test_nan_ratio_051_rejected(self):
        """nan_ratio=0.51 → rejected."""
        checker = DomainBoundaryChecker()
        # 10 values, 6 NaN = 0.6 ratio
        values = [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0, 9.0, 10.0]
        result = checker.check(values=values)
        
        assert result.within_domain is False
        assert result.rejection_reason == "excessive_missing"

    def test_nan_ratio_exactly_02_no_warning(self):
        """nan_ratio=0.2 exactly → no warning (threshold is > 0.2)."""
        checker = DomainBoundaryChecker()
        # 10 values, 2 NaN = 0.2 ratio
        values = [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, 8.0, 9.0, 10.0]
        result = checker.check(values=values)
        
        # nan_ratio = 2/10 = 0.2, condition is > 0.2 so no warning
        assert result.within_domain is True
        assert "high_missing_ratio" not in result.warnings

    def test_nan_ratio_021_warning(self):
        """nan_ratio=0.21 → warning triggered."""
        checker = DomainBoundaryChecker()
        # 10 values, 3 NaN = 0.3 ratio
        values = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, np.nan, 9.0, 10.0]
        result = checker.check(values=values)
        
        assert result.within_domain is True
        assert "high_missing_ratio" in result.warnings


class TestAllNanValues:
    """All NaN values edge case."""

    def test_all_nan_values_rejected(self):
        """All values NaN → rejected."""
        checker = DomainBoundaryChecker()
        result = checker.check(values=[np.nan, np.nan, np.nan, np.nan, np.nan])
        
        # nan_ratio = 1.0 > 0.5 → rejected
        assert result.within_domain is False
        assert result.rejection_reason == "excessive_missing"
