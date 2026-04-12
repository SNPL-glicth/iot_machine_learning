"""Tests for PlasticityScope value object."""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.value_objects.plasticity_scope import PlasticityScope


class TestPlasticityScope:
    """Test suite for PlasticityScope."""

    def test_default_scope_redis_key(self) -> None:
        """Default scope (no domain) produces backward-compatible key."""
        scope = PlasticityScope(regime="STABLE")
        
        assert scope.redis_key == "plasticity:STABLE"
        assert scope.sql_scope == ""
        assert scope.is_default is True

    def test_scoped_redis_key(self) -> None:
        """Scoped plasticity includes domain in Redis key."""
        scope = PlasticityScope(domain="iot", regime="STABLE")
        
        assert scope.redis_key == "plasticity:iot:STABLE"
        assert scope.sql_scope == "iot"
        assert scope.is_default is False

    def test_scoped_isolation(self) -> None:
        """Different domains produce different keys (no contamination)."""
        iot_scope = PlasticityScope(domain="iot", regime="STABLE")
        finance_scope = PlasticityScope(domain="finance", regime="STABLE")
        default_scope = PlasticityScope(regime="STABLE")
        
        # All have same regime but different keys
        assert iot_scope.redis_key != finance_scope.redis_key
        assert iot_scope.redis_key != default_scope.redis_key
        assert finance_scope.redis_key != default_scope.redis_key

    def test_with_regime(self) -> None:
        """Can create new scope with different regime but same domain."""
        stable_scope = PlasticityScope(domain="iot", regime="STABLE")
        trending_scope = stable_scope.with_regime("TRENDING")
        
        assert trending_scope.domain == "iot"
        assert trending_scope.regime == "TRENDING"
        assert trending_scope.redis_key == "plasticity:iot:TRENDING"

    def test_str_representation(self) -> None:
        """String representation is readable."""
        scoped = PlasticityScope(domain="iot", regime="STABLE")
        default = PlasticityScope(regime="STABLE")
        
        assert str(scoped) == "iot:STABLE"
        assert str(default) == "STABLE"

    def test_empty_regime_raises(self) -> None:
        """Empty regime is invalid."""
        with pytest.raises(ValueError, match="regime cannot be empty"):
            PlasticityScope(regime="")
        
        with pytest.raises(ValueError, match="regime cannot be empty"):
            PlasticityScope(regime="   ")

    def test_immutability(self) -> None:
        """Scope is frozen dataclass - cannot modify."""
        scope = PlasticityScope(domain="iot", regime="STABLE")
        
        with pytest.raises(AttributeError):
            scope.domain = "finance"  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            scope.regime = "VOLATILE"  # type: ignore[misc]

    def test_hashable(self) -> None:
        """Scope can be used as dict key."""
        scope1 = PlasticityScope(domain="iot", regime="STABLE")
        scope2 = PlasticityScope(domain="iot", regime="STABLE")
        scope3 = PlasticityScope(domain="finance", regime="STABLE")
        
        # Same values should be equal
        assert scope1 == scope2
        assert hash(scope1) == hash(scope2)
        
        # Different domain should not be equal
        assert scope1 != scope3
