"""Tests for plasticity warm start and persistence features."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, call

from iot_machine_learning.infrastructure.ml.cognitive.plasticity.base import PlasticityTracker
from iot_machine_learning.infrastructure.persistence.inmemory.plasticity_repository import (
    InMemoryPlasticityRepository,
)
from iot_machine_learning.domain.value_objects.plasticity_scope import PlasticityScope


class TestWarmStart:
    """Test suite for warm start functionality."""

    def test_repository_no_longer_optional(self) -> None:
        """Repository is now mandatory - uses InMemory if None."""
        repo = InMemoryPlasticityRepository(warn_on_init=False)
        tracker = PlasticityTracker(repository=repo)
        
        # Should have the persistence component with the repository
        assert tracker._persistence.repository is repo

    def test_warm_start_loads_state(self) -> None:
        """On initialization, state is loaded from repository."""
        # Create first tracker and add some state
        repo = InMemoryPlasticityRepository(warn_on_init=False)
        tracker1 = PlasticityTracker(repository=repo)
        tracker1.update("STABLE", "taylor", 0.1)
        tracker1.update("STABLE", "baseline", 0.2)
        tracker1._persistence.persist_regime_state(
            "STABLE",
            tracker1._accuracy,
            tracker1._priors,
            tracker1._regime_last_access,
            tracker1._regime_last_update,
        )  # Force persist
        
        # Create second tracker with same repository - should warm start
        tracker2 = PlasticityTracker(repository=repo)
        
        # Should have loaded the state
        assert tracker2.has_history("STABLE")
        weights = tracker2.get_weights("STABLE", ["taylor", "baseline"])
        assert "taylor" in weights
        assert "baseline" in weights

    def test_inmemory_logs_warning(self, caplog) -> None:
        """InMemory repository logs warning about state loss."""
        import logging
        
        with caplog.at_level(logging.WARNING):
            InMemoryPlasticityRepository(warn_on_init=True)
        
        assert "LOST" in caplog.text
        assert "restart" in caplog.text


class TestImmediatePersistence:
    """Test suite for immediate persistence on large changes."""

    def test_large_change_triggers_immediate_persist(self) -> None:
        """Accuracy change > threshold triggers immediate persist."""
        import logging
        
        repo = InMemoryPlasticityRepository(warn_on_init=False)
        tracker = PlasticityTracker(
            repository=repo,
            immediate_persist_threshold=0.15,
        )
        
        # Mock the repository to track calls
        repo.save_regime_state = Mock()
        
        # First update (no previous accuracy, so no persist_immediately)
        tracker.update("STABLE", "taylor", 0.5)
        
        # Second update with small change (< 0.15) - no immediate persist
        tracker.update("STABLE", "taylor", 0.52)  # accuracy ~0.49 -> ~0.49
        
        # Third update with large change (> 0.15) - triggers immediate persist
        tracker.update("STABLE", "taylor", 10.0)  # accuracy ~0.09, change > 0.15
        
        # Check that persist_immediately was called (at least once for the large change)
        assert repo.save_regime_state.call_count >= 1

    def test_persist_immediately_explicit_call(self) -> None:
        """Can explicitly persist a specific engine/regime."""
        repo = InMemoryPlasticityRepository(warn_on_init=False)
        tracker = PlasticityTracker(repository=repo)
        
        tracker.update("STABLE", "taylor", 0.1)
        tracker.update("STABLE", "baseline", 0.2)
        
        # Explicitly persist just taylor
        tracker.persist_immediately("taylor", "STABLE")
        
        # Create new tracker - should have taylor but maybe not baseline
        tracker2 = PlasticityTracker(repository=repo)
        assert tracker2.has_history("STABLE")


class TestCheckpoint:
    """Test suite for checkpoint export/import."""

    def test_export_checkpoint_structure(self) -> None:
        """Checkpoint has expected schema."""
        repo = InMemoryPlasticityRepository(warn_on_init=False)
        tracker = PlasticityTracker(repository=repo)
        tracker.update("STABLE", "taylor", 0.1)
        tracker.update("VOLATILE", "baseline", 0.2)
        
        checkpoint = tracker.export_checkpoint()
        
        assert checkpoint["version"] == "1.0"
        assert "scope" in checkpoint
        assert "timestamp" in checkpoint
        assert "regimes" in checkpoint
        assert "metadata" in checkpoint
        
        # Should have both regimes
        assert "STABLE" in checkpoint["regimes"]
        assert "VOLATILE" in checkpoint["regimes"]
        
        # Metadata
        assert checkpoint["metadata"]["regime_count"] == 2
        assert checkpoint["metadata"]["engine_count"] == 2

    def test_export_import_roundtrip(self) -> None:
        """Export then import restores state."""
        repo1 = InMemoryPlasticityRepository(warn_on_init=False)
        tracker1 = PlasticityTracker(repository=repo1)
        tracker1.update("STABLE", "taylor", 0.1)
        tracker1.update("STABLE", "baseline", 0.2)
        
        # Export
        checkpoint = tracker1.export_checkpoint()
        
        # Import to new tracker (doesn't need repo for checkpoint restore)
        tracker2 = PlasticityTracker()
        tracker2.restore_from_checkpoint(checkpoint)
        
        # Should have same state
        assert tracker2.has_history("STABLE")
        weights = tracker2.get_weights("STABLE", ["taylor", "baseline"])
        assert "taylor" in weights
        assert "baseline" in weights

    def test_checkpoint_scope_mismatch_warning(self, caplog) -> None:
        """Warns if checkpoint scope doesn't match tracker scope."""
        import logging
        
        repo1 = InMemoryPlasticityRepository(warn_on_init=False)
        tracker1 = PlasticityTracker(
            repository=repo1,
            scope=PlasticityScope(domain="iot", regime="STABLE"),
        )
        tracker1.update("STABLE", "taylor", 0.1)
        
        checkpoint = tracker1.export_checkpoint()
        
        # Restore to tracker with different scope
        repo2 = InMemoryPlasticityRepository(warn_on_init=False)
        tracker2 = PlasticityTracker(
            repository=repo2,
            scope=PlasticityScope(domain="finance", regime="STABLE"),
        )
        
        # Should warn about scope mismatch but still restore
        with caplog.at_level(logging.WARNING):
            tracker2.restore_from_checkpoint(checkpoint)
        
        assert "checkpoint_scope_mismatch" in caplog.text

    def test_incompatible_version_raises(self) -> None:
        """Incompatible checkpoint version raises ValueError."""
        repo = InMemoryPlasticityRepository(warn_on_init=False)
        tracker = PlasticityTracker(repository=repo)
        
        with pytest.raises(ValueError, match="Incompatible checkpoint version"):
            tracker.restore_from_checkpoint({"version": "2.0"})


class TestScopedPlasticity:
    """Test suite for scoped plasticity (no contamination)."""

    def test_different_scopes_different_redis_keys(self) -> None:
        """Different scopes produce different Redis keys."""
        repo1 = InMemoryPlasticityRepository(warn_on_init=False)
        repo2 = InMemoryPlasticityRepository(warn_on_init=False)
        
        iot_tracker = PlasticityTracker(
            repository=repo1,
            scope=PlasticityScope(domain="iot", regime="STABLE"),
        )
        finance_tracker = PlasticityTracker(
            repository=repo2,
            scope=PlasticityScope(domain="finance", regime="STABLE"),
        )
        
        assert iot_tracker._redis._get_redis_key("STABLE") == "plasticity:iot:STABLE"
        assert finance_tracker._redis._get_redis_key("STABLE") == "plasticity:finance:STABLE"
        assert iot_tracker._redis._get_redis_key("STABLE") != finance_tracker._redis._get_redis_key("STABLE")

    def test_default_scope_backward_compatible(self) -> None:
        """Default scope (no domain) uses backward-compatible key."""
        repo = InMemoryPlasticityRepository(warn_on_init=False)
        tracker = PlasticityTracker(repository=repo)  # scope=None
        
        assert tracker._redis._get_redis_key("STABLE") == "plasticity:STABLE"
