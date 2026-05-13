"""Domain service for confidence calibration.

Applies additive penalties based on signal quality indicators.
Stateless service — no internal state, pure function behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from core.parameters.numerical_constants import PENALTY_THRESHOLDS


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
    """
    Penalty-Based Confidence Calibrator.

    FÓRMULA: calibrated = raw_confidence - sum(penalties)
             calibrated = max(CONFIDENCE_FLOOR, calibrated)

    CUÁNDO USAR ESTE MÉTODO:
    - Pre-decisión: ajustar confidence por calidad de datos ANTES de tomar decisiones
    - Contextual penalties: cuando hay información sobre sample size, noise, disagreement
    - Engine fusion: después de fusionar múltiples engines, antes de decisión final
    - NO usar para: anomaly scores (usar infrastructure/ml/calibration/confidence_calibrator.py)
    - NO usar para: post-fusión probabilística (usar core/tuning/temperature_scaling.py)

    DIFERENCIA con core/tuning/temperature_scaling.py:
    - Este módulo: penalidades aditivas basadas en calidad de datos
    - TemperatureScaler: sigmoid centrado para calibración probabilística
    - Son métodos COMPLEMENTARIOS usados en distintas fases del pipeline:
      * PenaltyCalibrator: ajuste pre-decisión por calidad contextual
      * TemperatureScaler: calibración post-fusión por régimen

    NOTA sobre CONFIDENCE_FLOOR=0.05:
      Intencionalmente menor que CONFIDENCE.MIN_CONFIDENCE=0.3.
      El floor global aplica a engine outputs, no a confidence
      post-penalización donde múltiples señales negativas pueden
      justificar valores bajos.
      
    CONFIDENCE FLOOR INCONSISTENCY (FASE-26 - Phase 9 audit):
      CONFIDENCE_FLOOR=0.05 (este calibrador) vs CONFIDENCE.MIN_CONFIDENCE=0.3
      RISK: Inconsistencia puede causar divergencia en confidence bounds.
      PENDING: Decidir valor unificado con datos históricos de accuracy.
      Ver: Phase 9 audit Section 2.2 "Confidence Divergence Risks".
    """

    # Penalty constants
    PENALTY_ONLY_BASELINE: float = 0.25
    PENALTY_LOW_SAMPLE_SIZE: float = 0.20
    PENALTY_HIGH_NOISE: float = 0.15
    PENALTY_ENGINE_DISAGREEMENT: float = 0.15
    PENALTY_COHERENCE_CONFLICT: float = 0.20

    # Penalty cap to prevent over-penalization
    MAX_PENALTY_RATIO: float = 0.5  # Cap total penalty at 50% of raw_confidence

    # Bounds
    # Floor intencional en 0.05 (no 0.3 del global MIN_CONFIDENCE).
    # Justificación: este calibrador aplica PENALIDADES ADITIVAS que pueden
    # reducir legitimamente la confianza por debajo de 0.3 cuando hay
    # múltiples señales negativas (solo baseline + alto ruido + desacuerdo).
    # 0.05 = mínimo absoluto para evitar confidence=0 que bloquearía decisiones.
    # CONFIDENCE.MIN_CONFIDENCE=0.3 aplica a OUTPUTS de engines, no post-penalización.
    CONFIDENCE_FLOOR: float = 0.05
    
    # Ceiling intencional en 0.85 (no 0.95 del global MAX_CONFIDENCE).
    # Justificación: 0.85 es 10pp por debajo del máximo global como margen
    # conservador cuando NO hay consenso entre engines (raw >= 0.95 sin evidencia fuerte).
    # Previene over-confidence cuando solo un engine reporta alta confianza.
    # CONFIDENCE.MAX_CONFIDENCE=0.95 aplica cuando HAY consenso entre múltiples engines.
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
        if n_points < PENALTY_THRESHOLDS.MIN_POINTS:
            penalties.append(
                (self.PENALTY_LOW_SAMPLE_SIZE, f"n_points={n_points} < {PENALTY_THRESHOLDS.MIN_POINTS}")
            )

        # Penalty 3: High noise
        if noise_ratio > PENALTY_THRESHOLDS.MAX_NOISE_RATIO:
            penalties.append(
                (self.PENALTY_HIGH_NOISE, f"noise_ratio={noise_ratio:.3f} > {PENALTY_THRESHOLDS.MAX_NOISE_RATIO}")
            )

        # Penalty 4: Engine disagreement
        if engine_disagreement > PENALTY_THRESHOLDS.MAX_ENGINE_DISAGREEMENT:
            penalties.append(
                (self.PENALTY_ENGINE_DISAGREEMENT, f"engine_disagreement={engine_disagreement:.3f} > {PENALTY_THRESHOLDS.MAX_ENGINE_DISAGREEMENT}")
            )

        # Penalty 5: Coherence conflict
        if coherence_conflict:
            penalties.append(
                (self.PENALTY_COHERENCE_CONFLICT, "coherence_conflict detected")
            )

        # Calculate total penalty
        total_penalty = sum(p[0] for p in penalties)
        reasons = [p[1] for p in penalties]

        # Apply penalty cap to prevent over-penalization
        max_penalty = raw_confidence * self.MAX_PENALTY_RATIO
        if total_penalty > max_penalty:
            total_penalty = max_penalty
            reasons.append(f"penalty_capped: {max_penalty:.3f} (50% of raw={raw_confidence:.3f})")

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
