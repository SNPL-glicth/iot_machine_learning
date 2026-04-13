"""
Tests de robustez — DataSanitizer.
Generados por ROBUSTNESS_AUDIT.md 2026-04-11.
Cada test corresponde a un escenario de datos problemáticos (S1-S9).
"""
import math

import pytest

from domain.validators.data_sanitizer import DataSanitizer, SanitizedInput


@pytest.fixture
def sanitizer():
    return DataSanitizer()


class TestDataSanitizerRobustness:

    # S1 — Lista vacía
    def test_s1_empty_values_raises(self, sanitizer):
        with pytest.raises(ValueError, match="S1"):
            sanitizer.sanitize([], [1, 2, 3])

    def test_s1_empty_timestamps_raises(self, sanitizer):
        with pytest.raises(ValueError, match="S1"):
            sanitizer.sanitize([1.0, 2.0, 3.0], [])

    def test_s1_below_min_length_raises(self, sanitizer):
        with pytest.raises(ValueError, match="S4"):
            sanitizer.sanitize([42.0], [1])

    # S2 — None en valores
    def test_s2_none_value_raises(self, sanitizer):
        with pytest.raises(ValueError, match="S2"):
            sanitizer.sanitize([None, 1.0, 2.0], [1, 2, 3])

    # S3 — NaN e Infinitos
    def test_s3_nan_raises(self, sanitizer):
        with pytest.raises(ValueError, match="S3"):
            sanitizer.sanitize([float('nan'), 1.0, 2.0], [1, 2, 3])

    def test_s3_inf_raises(self, sanitizer):
        with pytest.raises(ValueError, match="S3"):
            sanitizer.sanitize([float('inf'), 1.0, 2.0], [1, 2, 3])

    def test_s3_neg_inf_raises(self, sanitizer):
        with pytest.raises(ValueError, match="S3"):
            sanitizer.sanitize([float('-inf'), 1.0, 2.0], [1, 2, 3])

    # S4 — Un solo punto
    def test_s4_single_point_raises(self, sanitizer):
        with pytest.raises(ValueError, match="S4"):
            sanitizer.sanitize([42.0], [1.0])

    # S5 — Varianza cero
    def test_s5_zero_variance_returns_warning(self, sanitizer):
        result = sanitizer.sanitize(
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        assert isinstance(result, SanitizedInput)
        assert any("S5" in w for w in result.warnings)
        assert result.values == [5.0, 5.0, 5.0, 5.0, 5.0]

    # S6 — Timestamps desordenados
    def test_s6_unordered_timestamps_reordered(self, sanitizer):
        result = sanitizer.sanitize(
            [30.0, 10.0, 20.0],
            [3.0, 1.0, 2.0]
        )
        assert result.timestamps == [1.0, 2.0, 3.0]
        assert result.values == [10.0, 20.0, 30.0]
        assert any("S6" in w for w in result.warnings)

    # S7 — Spike extremo
    def test_s7_extreme_spike_is_winsorized(self, sanitizer):
        result = sanitizer.sanitize(
            [1.0, 2.0, 1e15, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        assert max(result.values) < 1e10
        assert any("S7" in w for w in result.warnings)

    def test_s7_normal_values_not_clipped(self, sanitizer):
        values = [10.0, 12.0, 11.0, 13.0, 10.5]
        result = sanitizer.sanitize(
            values,
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        assert not any("S7" in w for w in result.warnings)

    # S8 — Timestamps duplicados
    def test_s8_duplicate_timestamps_fixed(self, sanitizer):
        result = sanitizer.sanitize(
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 2.0]
        )
        assert len(set(result.timestamps)) == len(result.timestamps)
        assert any("S8" in w for w in result.warnings)

    # S9 — Mezcla caótica
    def test_s9_chaos_input_raises_with_clear_message(self, sanitizer):
        with pytest.raises(ValueError) as exc:
            sanitizer.sanitize(
                [None, 1e15, float('nan'), 5.0],
                [2.0, 2.0, 1.0, None]
            )
        assert "S" in str(exc.value)  # identifica el escenario

    # Caso feliz — input limpio pasa sin warnings
    def test_clean_input_passes_without_warnings(self, sanitizer):
        result = sanitizer.sanitize(
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        assert result.values == [10.0, 11.0, 12.0, 13.0, 14.0]
        assert result.warnings == []
