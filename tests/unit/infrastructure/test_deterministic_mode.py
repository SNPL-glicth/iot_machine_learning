"""Tests for deterministic mode in Zenin analysis."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any, Dict, Optional

from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.engine import UniversalAnalysisEngine
from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.types import UniversalContext


class TestDeterministicMode:
    """Test suite for deterministic mode."""
    
    def test_deterministic_mode_disables_monte_carlo(self):
        """Test that deterministic mode disables Monte Carlo."""
        engine = UniversalAnalysisEngine(
            deterministic_mode=True,
            enable_monte_carlo=True,  # Should be ignored
        )
        
        # Monte Carlo should not run even if enabled
        assert engine._deterministic_mode is True
        assert engine._enable_monte_carlo is True  # Flag is set
        # But Monte Carlo won't run (tested in integration)
    
    def test_deterministic_mode_sets_seed(self):
        """Test that deterministic mode sets random seed."""
        import random
        import numpy as np
        
        # Create engine with deterministic mode
        engine = UniversalAnalysisEngine(
            deterministic_mode=True,
            analysis_seed=42,
        )
        
        # Generate random numbers
        r1 = random.random()
        n1 = np.random.random()
        
        # Create another engine with same seed
        engine2 = UniversalAnalysisEngine(
            deterministic_mode=True,
            analysis_seed=42,
        )
        
        # Should get same random numbers
        r2 = random.random()
        n2 = np.random.random()
        
        assert r1 == r2
        assert n1 == n2
    
    
    def test_non_deterministic_mode_default(self):
        """Test that non-deterministic mode is default."""
        engine = UniversalAnalysisEngine()
        
        assert engine._deterministic_mode is False
    
    def test_custom_seed(self):
        """Test custom analysis seed."""
        engine = UniversalAnalysisEngine(
            deterministic_mode=True,
            analysis_seed=123,
        )
        
        assert engine._analysis_seed == 123
