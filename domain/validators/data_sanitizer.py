"""
DataSanitizer — boundary de entrada al pipeline cognitivo.

Intercepta los 9 escenarios de robustez identificados en
ROBUSTNESS_AUDIT.md antes de que lleguen al pipeline.
Un solo punto de validación. Sin lógica de negocio.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SanitizedInput:
    values: List[float]
    timestamps: List[float]
    warnings: List[str]  # non-fatal issues found and corrected


@dataclass
class SanitizationError:
    scenario: str
    reason: str
    field: str


class DataSanitizer:
    """
    Valida y normaliza input antes de entrar al pipeline.
    Distingue entre errores fatales (raise) y correcciones
    silenciosas (log warning, continúa).
    """

    MIN_POINTS: int = 3
    MAX_SPIKE_IQR_FACTOR: float = 10.0

    def sanitize(
        self,
        values: List[Optional[float]],
        timestamps: List[Optional[float]],
    ) -> SanitizedInput:
        """
        Punto de entrada único. Ejecuta todos los checks en orden.
        Raises ValueError con escenario identificado si es fatal.
        Returns SanitizedInput con warnings para issues corregidos.
        """
        warnings: List[str] = []

        # S1 — listas vacías
        self._check_not_empty(values, timestamps)

        # S2/S3 — None, NaN, Inf en values
        values = self._check_values_finite(values)

        # S8 — timestamps None o no numéricos
        timestamps = self._check_timestamps_valid(timestamps)

        # S1 ampliado — longitud mínima
        self._check_min_length(values, timestamps)

        # longitudes iguales
        self._check_equal_length(values, timestamps)

        # S6 — timestamps desordenados → corrige con warning
        values, timestamps, reordered = self._sort_by_timestamp(
            values, timestamps
        )
        if reordered:
            warnings.append(
                "S6: timestamps were out of order — "
                "reordered automatically"
            )

        # S8 — timestamps duplicados → corrige con epsilon + warning
        timestamps, had_dupes = self._fix_duplicate_timestamps(timestamps)
        if had_dupes:
            warnings.append(
                "S8: duplicate timestamps found — "
                "offset by epsilon to preserve order"
            )

        # S5 — varianza cero → warning, no error (pipeline lo maneja)
        if np.std(values) < 1e-10:
            warnings.append(
                "S5: zero-variance series detected — "
                "anomaly detection results may be undefined"
            )

        # S7 — spike extremo → winsorize con warning
        values, clipped = self._clip_spikes(values)
        if clipped:
            warnings.append(
                f"S7: extreme spike detected and winsorized "
                f"at {self.MAX_SPIKE_IQR_FACTOR}x IQR"
            )

        return SanitizedInput(
            values=values,
            timestamps=timestamps,
            warnings=warnings,
        )

    # ── checks fatales ──────────────────────────────────────────

    def _check_not_empty(self, values, timestamps) -> None:
        if not values:
            raise ValueError(
                "S1: values list is empty. "
                "Minimum 3 readings required."
            )
        if not timestamps:
            raise ValueError(
                "S1: timestamps list is empty. "
                "Minimum 3 timestamps required."
            )

    def _check_values_finite(
        self, values: List[Optional[float]]
    ) -> List[float]:
        result = []
        for i, v in enumerate(values):
            if v is None:
                raise ValueError(
                    f"S2: None value at index {i}. "
                    f"All values must be finite floats."
                )
            if not math.isfinite(v):
                raise ValueError(
                    f"S3: Non-finite value {v!r} at index {i}. "
                    f"NaN and Inf are not allowed."
                )
            result.append(float(v))
        return result

    def _check_timestamps_valid(
        self, timestamps: List[Optional[float]]
    ) -> List[float]:
        result = []
        for i, t in enumerate(timestamps):
            if t is None:
                raise ValueError(
                    f"S9: None timestamp at index {i}. "
                    f"All timestamps must be numeric."
                )
            if not math.isfinite(t):
                raise ValueError(
                    f"S9: Non-finite timestamp {t!r} at index {i}."
                )
            result.append(float(t))
        return result

    def _check_min_length(self, values, timestamps) -> None:
        if len(values) < self.MIN_POINTS:
            raise ValueError(
                f"S4: Only {len(values)} value(s) provided. "
                f"Minimum {self.MIN_POINTS} required for reliable prediction."
            )

    def _check_equal_length(self, values, timestamps) -> None:
        if len(values) != len(timestamps):
            raise ValueError(
                f"S9: values length ({len(values)}) != "
                f"timestamps length ({len(timestamps)}). "
                f"Both lists must have equal length."
            )

    # ── correcciones silenciosas ────────────────────────────────

    def _sort_by_timestamp(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> Tuple[List[float], List[float], bool]:
        if timestamps == sorted(timestamps):
            return values, timestamps, False
        pairs = sorted(zip(timestamps, values))
        ts_sorted = [p[0] for p in pairs]
        v_sorted = [p[1] for p in pairs]
        return v_sorted, ts_sorted, True

    def _fix_duplicate_timestamps(
        self, timestamps: List[float]
    ) -> Tuple[List[float], bool]:
        seen = {}
        result = list(timestamps)
        had_dupes = False
        epsilon = 1e-6
        for i, t in enumerate(timestamps):
            if t in seen:
                had_dupes = True
                result[i] = t + epsilon * (seen[t] + 1)
                seen[t] += 1
            else:
                seen[t] = 0
        return result, had_dupes

    def _clip_spikes(
        self, values: List[float]
    ) -> Tuple[List[float], bool]:
        arr = np.array(values)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-10:
            return values, False  # constant series, skip
        lower = q1 - self.MAX_SPIKE_IQR_FACTOR * iqr
        upper = q3 + self.MAX_SPIKE_IQR_FACTOR * iqr
        clipped = np.clip(arr, lower, upper)
        was_clipped = not np.allclose(arr, clipped)
        return clipped.tolist(), was_clipped
