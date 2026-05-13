"""Tests for BayesianWeightConfig and BayesianWeightTracker configuration injection.

Verifies that:
- Config validation rejects invalid parameter ranges
- Tracker uses injected config instead of hardcoded defaults
- No raw literals remain in tracker source code
- WeightTrackerConfig emits deprecation warning
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import warnings

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.base import (
    BayesianWeightTracker,
)
from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.bayesian_weight_config import (
    BayesianWeightConfig,
)
from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.constants import (
    WeightTrackerConfig,
)


class TestBayesianWeightConfigValidation:
    """Test suite for BayesianWeightConfig validation."""

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
    def test_validate_rejects_alpha_out_of_range(self, alpha: float):
        """Verify that config validation rejects alpha outside (0.0, 1.0).

        WHY: alpha must be strictly between 0 and 1 for valid learning rate.

        Tests:
        - alpha=0.0 (lower bound exclusive)
        - alpha=1.0 (upper bound exclusive)
        - alpha=1.5 (above upper bound)
        - alpha=-0.1 (below lower bound)
        """
        config = dataclasses.replace(BayesianWeightConfig(), alpha=alpha)
        with pytest.raises(ValueError, match="alpha"):
            config.validate()

    def test_validate_rejects_variance_window_smaller_than_min_samples(self):
        """Verify that variance_window must be >= variance_min_samples.

        WHY: Cannot estimate variance with more samples than window size.
        """
        config = dataclasses.replace(
            BayesianWeightConfig(),
            variance_window=3,
            variance_min_samples=5,
        )
        with pytest.raises(ValueError, match="variance_min_samples"):
            config.validate()

    def test_validate_rejects_convergence_window_larger_than_history(self):
        """Verify that convergence_window must be <= weight_history_maxlen.

        WHY: Cannot check convergence with more samples than history size.
        """
        config = dataclasses.replace(
            BayesianWeightConfig(),
            convergence_window=60,
            weight_history_maxlen=50,
        )
        with pytest.raises(ValueError, match="convergence_window"):
            config.validate()

    def test_validate_rejects_sigma2_min_larger_than_default(self):
        """Verify that sigma2_obs_min must be < sigma2_obs_default.

        WHY: Minimum variance must be less than default variance.
        """
        config = dataclasses.replace(
            BayesianWeightConfig(),
            sigma2_obs_min=2.0,
            sigma2_obs_default=1.0,
        )
        with pytest.raises(ValueError, match="sigma2_obs_min"):
            config.validate()

    def test_validate_rejects_drift_expansion_below_1(self):
        """Verify that drift_variance_expansion must be > 1.0.

        WHY: Drift must increase uncertainty, not decrease it.
        """
        config = dataclasses.replace(
            BayesianWeightConfig(),
            drift_variance_expansion=0.9,
        )
        with pytest.raises(ValueError, match="drift_variance_expansion"):
            config.validate()

    def test_validate_rejects_negative_floats(self):
        """Verify that all float parameters must be > 0.0.

        WHY: Negative values are invalid for variance, rates, thresholds.
        """
        config = dataclasses.replace(
            BayesianWeightConfig(),
            drift_decay_factor=-0.1,
        )
        with pytest.raises(ValueError, match="drift_decay_factor"):
            config.validate()

    def test_validate_rejects_zero_ints(self):
        """Verify that all int parameters must be >= 1.

        WHY: Zero or negative values are invalid for window sizes and counts.
        """
        config = dataclasses.replace(
            BayesianWeightConfig(),
            variance_window=0,
        )
        with pytest.raises(ValueError, match="variance_window"):
            config.validate()


class TestBayesianWeightConfigInjection:
    """Test suite for config injection into BayesianWeightTracker."""

    def test_config_injection_overrides_defaults(self):
        """Verify that tracker uses injected config, not hardcoded defaults.

        WHY: Ensures DIP is properly implemented — tracker depends on config
        abstraction, not on hardcoded values.
        """
        custom = dataclasses.replace(BayesianWeightConfig(), alpha=0.42)
        tracker = BayesianWeightTracker(config=custom)
        assert tracker._config.alpha == 0.42
        # Verify that it does NOT use the default 0.15
        assert tracker._config.alpha != 0.15

    def test_default_config_when_none_provided(self):
        """Verify that tracker uses default config when None is provided.

        WHY: Ensures backward compatibility for callers that don't inject config.
        """
        tracker = BayesianWeightTracker(config=None)
        assert tracker._config.alpha == 0.15  # Default value
        assert tracker._config.min_weight == 0.05
        assert tracker._config.drift_decay_factor == 0.5

    def test_validate_called_on_construction(self):
        """Verify that config.validate() is called during construction.

        WHY: Ensures fail-fast behavior — invalid config raises immediately.
        """
        invalid_config = dataclasses.replace(
            BayesianWeightConfig(),
            alpha=1.5,  # Invalid
        )
        with pytest.raises(ValueError, match="alpha"):
            BayesianWeightTracker(config=invalid_config)


class TestNoRawLiteralsInTrackerSource:
    """Test suite to verify no raw literals remain in tracker source code."""

    def test_no_raw_literals_in_tracker_source(self):
        """Verify that no magic number literals exist in BayesianWeightTracker source.

        WHY: Ensures all magic numbers were migrated to BayesianWeightConfig.
        Any remaining literal indicates incomplete refactoring.

        Tests that none of the 17 values from the inventory appear as
        standalone literals in the source code (excluding docstrings/comments).
        """
        source = inspect.getsource(BayesianWeightTracker)

        # Valores que NO deben aparecer como literales sueltos en base.py
        # Formatos específicos para evitar falsos positivos en comentarios
        forbidden_patterns = [
            r"alpha\s*=\s*0\.15",  # alpha = 0.15
            r"min_weight\s*=\s*0\.05",  # min_weight = 0.05
            r"drift_decay_factor\s*=\s*0\.5",  # drift_decay_factor = 0.5
            r"drift_variance_expansion\s*=\s*2\.0",  # drift_variance_expansion = 2.0
            r"regularization_strength\s*=\s*0\.01",  # regularization_strength = 0.01
            r"prior_variance_scale\s*=\s*1\.0",  # prior_variance_scale = 1.0
            r"maxlen\s*=\s*50",  # maxlen=50
            r"<\s*10\b",  # < 10 (convergence check)
            r"\[-10:\]",  # [-10:] (history slice)
            r"n\s*=\s*20",  # n=20 (variance window)
        ]

        # Filtrar patrones que podrían aparecer en comentarios o docstrings
        # Solo buscamos en líneas que no sean comentarios
        lines = source.split("\n")
        code_lines = []
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            # Detectar inicio/fin de docstrings
            if '"""' in stripped:
                if stripped.count('"""') == 2:
                    # Docstring de una sola línea
                    continue
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            # Saltar comentarios
            if stripped.startswith("#"):
                continue
            code_lines.append(line)
        code_source = "\n".join(code_lines)

        for pattern in forbidden_patterns:
            # Usar búsqueda estricta
            if re.search(pattern, code_source):
                pytest.fail(
                    f"Literal '{pattern}' encontrado en BayesianWeightTracker. "
                    f"Debe usar self._config.* en su lugar."
                )

    def test_config_references_present(self):
        """Verify that self._config is used throughout the tracker.

        WHY: Ensures the config is actually being used, not just stored.
        """
        source = inspect.getsource(BayesianWeightTracker)
        assert "self._config" in source, (
            "self._config no encontrado en BayesianWeightTracker. "
            "El tracker debe usar self._config para acceder a parámetros."
        )
        # Verificar usos específicos
        assert "self._config.alpha" in source
        assert "self._config.drift_decay_factor" in source
        assert "self._config.weight_history_maxlen" in source
        assert "self._config.convergence_window" in source


class TestWeightTrackerConfigDeprecation:
    """Test suite for WeightTrackerConfig deprecation warning."""

    def test_deprecation_warning_on_weight_tracker_config(self):
        """Verify that WeightTrackerConfig emits DeprecationWarning.

        WHY: Ensures legacy config class warns users to migrate to BayesianWeightConfig.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WeightTrackerConfig()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "BayesianWeightConfig" in str(w[0].message)

    def test_deprecation_message_is_clear(self):
        """Verify that deprecation message mentions BayesianWeightConfig.

        WHY: Users need to know what to migrate to.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WeightTrackerConfig()
            message = str(w[0].message)
            assert "BayesianWeightConfig" in message
            assert "deprecated" in message.lower()
