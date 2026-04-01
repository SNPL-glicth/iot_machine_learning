"""Confidence calibrator — adjusts raw confidence to reflect real uncertainty.

Transforms optimistically-reported engine confidences into calibrated
estimates that account for:
- Data quality (sample size, noise)
- Engine diversity (disagreement between engines)
- System state (inhibition, coherence conflicts)

Pure domain logic — no I/O, no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class CalibratedConfidence:
    """Calibrated confidence with full audit trail.

    Attributes:
        calibrated: Final adjusted confidence [0.0, 1.0].
        raw: Original uncalibrated confidence.
        penalty_applied: Total penalty subtracted from raw.
        reasons: List of reasons for each penalty applied.
    """

    calibrated: float
    raw: float
    penalty_applied: float
    reasons: List[str] = field(default_factory=list)


class ConfidenceCalibrator:
    """Calibrates raw confidence scores to reflect true uncertainty.

    Stateless domain service — applies additive penalties based on
    signal quality indicators. Never goes below confidence floor.
    """

    # Penalty constants
    PENALTY_ONLY_BASELINE: float = 0.25
    PENALTY_LOW_SAMPLE_SIZE: float = 0.20
    PENALTY_HIGH_NOISE: float = 0.15
    PENALTY_ENGINE_DISAGREEMENT: float = 0.15
    PENALTY_COHERENCE_CONFLICT: float = 0.20

    # Bounds
    CONFIDENCE_FLOOR: float = 0.05
    CONFIDENCE_CEIL_NO_CONSENSUS: float = 0.85

    def calibrate(
        self,
        raw_confidence: float,
        n_points: int,
        noise_ratio: float = 0.0,
        engine_disagreement: float = 0.0,
        only_baseline_active: bool = False,
        coherence_conflict: bool = False,
        all_engines_inhibited: bool = False,
    ) -> CalibratedConfidence:
        """Calibrate raw confidence based on quality indicators.

        Penalties are additive and cumulative. Final confidence
        is clamped to [CONFIDENCE_FLOOR, 1.0].

        Args:
            raw_confidence: Original fused confidence from engines.
            n_points: Number of data points in the window.
            noise_ratio: Signal noise ratio (0.0–1.0) from analysis.
            engine_disagreement: Max difference between engine predictions.
            only_baseline_active: True if only baseline engine contributed.
            coherence_conflict: True if EJE 2 coherence check found conflict.
            all_engines_inhibited: True if all engines were suppressed.

        Returns:
            CalibratedConfidence with adjusted value and penalty audit.
        """
        penalties: List[tuple[float, str]] = []

        # Penalty 1: Only baseline or all inhibited
        if only_baseline_active or all_engines_inhibited:
            penalties.append(
                (self.PENALTY_ONLY_BASELINE, "only_baseline_active or all_engines_inhibited")
            )

        # Penalty 2: Low sample size
        if n_points < 10:
            penalties.append(
                (self.PENALTY_LOW_SAMPLE_SIZE, f"n_points={n_points} < 10")
            )

        # Penalty 3: High noise
        if noise_ratio > 0.6:
            penalties.append(
                (self.PENALTY_HIGH_NOISE, f"noise_ratio={noise_ratio:.3f} > 0.6")
            )

        # Penalty 4: Engine disagreement
        if engine_disagreement > 0.3:
            penalties.append(
                (self.PENALTY_ENGINE_DISAGREEMENT, f"engine_disagreement={engine_disagreement:.3f} > 0.3")
            )

        # Penalty 5: Coherence conflict
        if coherence_conflict:
            penalties.append(
                (self.PENALTY_COHERENCE_CONFLICT, "coherence_conflict detected")
            )

        # Calculate total penalty
        total_penalty = sum(p[0] for p in penalties)
        reasons = [p[1] for p in penalties]

        # Apply penalties
        calibrated = raw_confidence - total_penalty

        # Apply floor (never below minimum)
        calibrated = max(self.CONFIDENCE_FLOOR, calibrated)

        # Apply ceiling if no consensus (raw >= 0.95 without strong evidence)
        has_strong_evidence = (
            not only_baseline_active
            and not all_engines_inhibited
            and not coherence_conflict
            and engine_disagreement < 0.2
            and noise_ratio < 0.4
        )
        if raw_confidence >= 0.95 and not has_strong_evidence:
            # When raw is very high but we lack consensus, cap at ceiling
            # This overrides the penalized value to prevent false confidence
            calibrated = self.CONFIDENCE_CEIL_NO_CONSENSUS
            reasons.append(f"ceil_applied: raw={raw_confidence:.3f} >= 0.95 without consensus")

        # Final clamp to [0, 1]
        calibrated = max(0.0, min(1.0, calibrated))

        return CalibratedConfidence(
            calibrated=calibrated,
            raw=raw_confidence,
            penalty_applied=raw_confidence - calibrated,
            reasons=reasons,
        )

    def compute_engine_disagreement(
        self,
        perceptions: list[object],
    ) -> float:
        """Compute max normalized disagreement between engine predictions.

        Args:
            perceptions: List of EnginePerception objects with predicted_value.

        Returns:
            Max relative difference between any two engines [0.0, 1.0+].
        """
        if len(perceptions) < 2:
            return 0.0

        values = []
        for p in perceptions:
            if hasattr(p, 'predicted_value') and p.predicted_value is not None:
                values.append(p.predicted_value)

        if len(values) < 2:
            return 0.0

        # Normalize by mean to get relative disagreement
        mean_val = sum(values) / len(values)
        if abs(mean_val) < 1e-9:
            # If mean is near zero, use absolute differences
            max_diff = max(values) - min(values)
            return min(1.0, max_diff / 1.0)  # Normalize by 1.0 as reference

        diffs = [abs(v - mean_val) / abs(mean_val) for v in values]
        return max(diffs)
