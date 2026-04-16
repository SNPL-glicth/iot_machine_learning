"""Tests for domain.model.context_vector.

Coverage: construction, serialization, immutability.
"""

import pytest
from dataclasses import FrozenInstanceError

from domain.model.context_vector import ContextVector


class TestContextVectorConstruction:
    """Tests for ContextVector construction and validation."""
    
    def test_basic_construction(self):
        """Create ContextVector with minimal required fields."""
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={"mean": 20.0, "std": 1.5}
        )
        
        assert ctx.regime == "stable"
        assert ctx.domain == "iot"
        assert ctx.n_points == 10
        assert ctx.signal_features == {"mean": 20.0, "std": 1.5}
    
    def test_construction_with_optional_fields(self):
        """Create ContextVector with all optional fields."""
        ctx = ContextVector(
            regime="volatile",
            domain="finance",
            n_points=20,
            signal_features={"mean": 100.0},
            temporal_features={"hour": 14, "day_of_week": 1},
            metadata={"source": "sensor_42"}
        )
        
        assert ctx.temporal_features == {"hour": 14, "day_of_week": 1}
        assert ctx.metadata == {"source": "sensor_42"}
    
    def test_negative_n_points_raises(self):
        """Negative n_points should raise ValueError."""
        with pytest.raises(ValueError, match="n_points must be non-negative"):
            ContextVector(
                regime="stable",
                domain="iot",
                n_points=-1,
                signal_features={}
            )
    
    def test_empty_regime_raises(self):
        """Empty regime should raise ValueError."""
        with pytest.raises(ValueError, match="regime cannot be empty"):
            ContextVector(
                regime="",
                domain="iot",
                n_points=5,
                signal_features={}
            )
    
    def test_empty_domain_raises(self):
        """Empty domain should raise ValueError."""
        with pytest.raises(ValueError, match="domain cannot be empty"):
            ContextVector(
                regime="stable",
                domain="",
                n_points=5,
                signal_features={}
            )
    
    def test_zero_n_points_valid(self):
        """Zero n_points is valid (edge case)."""
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=0,
            signal_features={}
        )
        
        assert ctx.n_points == 0


class TestContextVectorToArray:
    """Tests for to_array() serialization."""
    
    def test_to_array_length(self):
        """Array has deterministic length."""
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={"mean": 20.0, "std": 1.5}
        )
        
        arr = ctx.to_array()
        
        assert len(arr) == ctx.feature_count
        assert len(arr) == 11  # 7 base + 4 regime one-hot
    
    def test_to_array_values(self):
        """Array contains expected values in order."""
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={
                "mean": 20.0,
                "std": 1.5,
                "slope": 0.5,
                "curvature": 0.1,
                "noise_ratio": 0.05,
                "stability": 0.8,
            }
        )
        
        arr = ctx.to_array()
        
        assert arr[0] == 10.0  # n_points
        assert arr[1] == 20.0  # mean
        assert arr[2] == 1.5   # std
        assert arr[3] == 0.5   # slope
        assert arr[4] == 0.1   # curvature
        assert arr[5] == 0.05  # noise_ratio
        assert arr[6] == 0.8   # stability
        # Regime one-hot: stable=1, others=0
        assert arr[7] == 1.0   # stable
        assert arr[8] == 0.0   # trending
        assert arr[9] == 0.0   # volatile
        assert arr[10] == 0.0  # noisy
    
    def test_to_array_default_values(self):
        """Missing features default to 0.0."""
        ctx = ContextVector(
            regime="trending",
            domain="iot",
            n_points=5,
            signal_features={}  # Empty
        )
        
        arr = ctx.to_array()
        
        assert arr[0] == 5.0  # n_points
        assert arr[1] == 0.0  # mean (default)
        assert arr[2] == 0.0  # std (default)
        # trending one-hot
        assert arr[8] == 1.0
    
    def test_to_array_all_regimes(self):
        """Test one-hot encoding for all regime values."""
        regimes = ["stable", "trending", "volatile", "noisy"]
        
        for i, regime in enumerate(regimes):
            ctx = ContextVector(
                regime=regime,
                domain="iot",
                n_points=10,
                signal_features={}
            )
            arr = ctx.to_array()
            
            # Only one regime should be 1.0
            regime_start = 7
            for j in range(4):
                expected = 1.0 if j == i else 0.0
                assert arr[regime_start + j] == expected, f"Regime {regime} at index {j}"


class TestContextVectorSerialization:
    """Tests for dict serialization."""
    
    def test_to_dict(self):
        """Serialize to dictionary."""
        ctx = ContextVector(
            regime="volatile",
            domain="finance",
            n_points=15,
            signal_features={"mean": 100.0},
            temporal_features={"hour": 12},
            metadata={"version": "1.0"}
        )
        
        d = ctx.to_dict()
        
        assert d["regime"] == "volatile"
        assert d["domain"] == "finance"
        assert d["n_points"] == 15
        assert d["signal_features"] == {"mean": 100.0}
        assert d["temporal_features"] == {"hour": 12}
        assert d["metadata"] == {"version": "1.0"}
    
    def test_from_dict(self):
        """Deserialize from dictionary."""
        data = {
            "regime": "noisy",
            "domain": "healthcare",
            "n_points": 8,
            "signal_features": {"std": 2.5},
            "temporal_features": None,
            "metadata": None,
        }
        
        ctx = ContextVector.from_dict(data)
        
        assert ctx.regime == "noisy"
        assert ctx.domain == "healthcare"
        assert ctx.n_points == 8
        assert ctx.signal_features == {"std": 2.5}
    
    def test_from_dict_roundtrip(self):
        """Roundtrip serialization preserves data."""
        original = ContextVector(
            regime="trending",
            domain="iot",
            n_points=12,
            signal_features={"mean": 25.0, "slope": 1.2},
            temporal_features={"hour": 8},
            metadata={"id": "test"}
        )
        
        data = original.to_dict()
        restored = ContextVector.from_dict(data)
        
        assert restored == original


class TestContextVectorImmutability:
    """Tests for frozen dataclass behavior."""
    
    def test_cannot_modify_fields(self):
        """Frozen dataclass prevents field modification."""
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        with pytest.raises(FrozenInstanceError):
            ctx.regime = "volatile"
    
    def test_cannot_modify_signal_features(self):
        """Dict inside frozen dataclass is still mutable but discouraged."""
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={"mean": 20.0}
        )
        
        # This actually works because dict is mutable
        # But we should discourage it
        ctx.signal_features["mean"] = 30.0
        assert ctx.signal_features["mean"] == 30.0


class TestContextVectorProperties:
    """Tests for computed properties."""
    
    def test_is_stable(self):
        """is_stable property."""
        stable_ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=5,
            signal_features={}
        )
        volatile_ctx = ContextVector(
            regime="volatile",
            domain="iot",
            n_points=5,
            signal_features={}
        )
        
        assert stable_ctx.is_stable is True
        assert volatile_ctx.is_stable is False
    
    def test_has_minimum_points_true(self):
        """has_minimum_points returns True when sufficient."""
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={}
        )
        
        assert ctx.has_minimum_points(3) is True
        assert ctx.has_minimum_points(10) is True
    
    def test_has_minimum_points_false(self):
        """has_minimum_points returns False when insufficient."""
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=2,
            signal_features={}
        )
        
        assert ctx.has_minimum_points(3) is False
    
    def test_feature_count_constant(self):
        """feature_count is always 11."""
        ctx1 = ContextVector(
            regime="stable", domain="iot", n_points=5, signal_features={}
        )
        ctx2 = ContextVector(
            regime="volatile", domain="finance", n_points=20,
            signal_features={"extra": 1.0}
        )
        
        assert ctx1.feature_count == 11
        assert ctx2.feature_count == 11


class TestContextVectorWithMethods:
    """Tests for immutable update methods."""
    
    def test_with_regime(self):
        """Create new ContextVector with updated regime."""
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={"mean": 20.0}
        )
        
        new_ctx = ctx.with_regime("volatile")
        
        assert new_ctx.regime == "volatile"
        assert ctx.regime == "stable"  # Original unchanged
        assert new_ctx.n_points == ctx.n_points
        assert new_ctx.signal_features == ctx.signal_features
    
    def test_with_signal_features(self):
        """Create new ContextVector with merged signal features."""
        ctx = ContextVector(
            regime="stable",
            domain="iot",
            n_points=10,
            signal_features={"mean": 20.0, "std": 1.5}
        )
        
        new_ctx = ctx.with_signal_features(slope=0.5, curvature=0.1)
        
        assert new_ctx.signal_features["mean"] == 20.0  # Preserved
        assert new_ctx.signal_features["std"] == 1.5    # Preserved
        assert new_ctx.signal_features["slope"] == 0.5  # Added
        assert new_ctx.signal_features["curvature"] == 0.1  # Added


class TestContextVectorEquality:
    """Tests for dataclass equality."""
    
    def test_equal_contexts(self):
        """Same values are equal."""
        ctx1 = ContextVector(
            regime="stable", domain="iot", n_points=5,
            signal_features={"mean": 10.0}
        )
        ctx2 = ContextVector(
            regime="stable", domain="iot", n_points=5,
            signal_features={"mean": 10.0}
        )
        
        assert ctx1 == ctx2
    
    def test_unequal_contexts(self):
        """Different values are not equal."""
        ctx1 = ContextVector(
            regime="stable", domain="iot", n_points=5, signal_features={}
        )
        ctx2 = ContextVector(
            regime="volatile", domain="iot", n_points=5, signal_features={}
        )
        
        assert ctx1 != ctx2
