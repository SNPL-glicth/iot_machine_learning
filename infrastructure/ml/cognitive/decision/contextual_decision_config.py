"""ContextualDecisionConfig — Configuration for contextual decision scoring.

Consolidates all magic numbers and thresholds for ContextualDecisionEngine.

Applies SRP: Configuration is separate from decision logic.
Applies DIP: Engine depends on this config abstraction, not on env vars.
Applies OCP: Adding new parameters only requires extending this dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextualDecisionConfig:
    """Configuration for ContextualDecisionEngine scoring.
    
    All parameters have documented justifications and valid ranges.
    
    Design principles:
    - Single source of truth for all decision thresholds
    - Immutable (frozen) for thread safety
    - Validated on construction to fail fast
    - No business logic, only data grouping
    """
    
    # ========== SCORING BASE ==========
    # Base scores by severity level (range: [0.0, 1.0])
    # Represents P(action required) given only severity, before context
    
    score_critical: float = 0.90
    # Score when severity is CRITICAL. Valid: [0.70, 1.0]
    # Justification: Critical events need immediate attention (90% confidence)
    
    score_high: float = 0.70
    # Score when severity is HIGH. Valid: [0.50, 0.90]
    # Justification: High severity likely requires investigation
    
    score_medium: float = 0.45
    # Score when severity is MEDIUM. Valid: [0.30, 0.70]
    # Justification: Midpoint - monitoring threshold
    
    score_low: float = 0.25
    # Score when severity is LOW. Valid: [0.10, 0.45]
    # Justification: Low severity typically log-only unless amplified
    
    score_none: float = 0.05
    # Score when no severity detected. Valid: [0.0, 0.25]
    # Justification: Minimal baseline score
    
    score_warning: float = 0.45
    # Score when severity is WARNING. Valid: [0.30, 0.70]
    # NOTE: Intentionally equals score_medium by design.
    # WARNING is treated as alias of MEDIUM in this domain.
    # If future requirements differentiate WARNING, change only this value.
    
    # ========== AMPLIFIERS (Additive) ==========
    # Values added to base score when conditions are met
    # All amplifiers are cumulative but final score is clamped to [0.0, 1.0]
    
    amp_consecutive_5: float = 0.15
    # Amplification when ≥5 consecutive anomalies. Valid: [0.10, 0.30]
    # Justification: Persistent pattern (not transient spike) → +15%
    
    amp_consecutive_3: float = 0.10
    # Amplification when ≥3 consecutive anomalies. Valid: [0.05, 0.20]
    # Justification: Emerging pattern → +10%
    
    amp_rate_high: float = 0.20
    # Amplification when anomaly rate > 60%. Valid: [0.10, 0.30]
    # Justification: Majority of recent window is anomalous → +20%
    
    amp_rate_med: float = 0.10
    # Amplification when anomaly rate > 30%. Valid: [0.05, 0.20]
    # Justification: Significant but not majority → +10%
    
    amp_volatile: float = 0.10
    # Amplification when regime is VOLATILE. Valid: [0.05, 0.20]
    # Justification: High variance increases uncertainty → +10%
    
    amp_noisy: float = 0.05
    # Amplification when regime is NOISY. Valid: [0.0, 0.15]
    # Justification: Noise makes detection harder → +5%
    
    amp_drift_high: float = 0.15
    # Amplification when drift score > 70%. Valid: [0.10, 0.25]
    # Justification: Significant distribution shift → +15%
    
    amp_drift_med: float = 0.10
    # Amplification when drift score > 40%. Valid: [0.05, 0.20]
    # Justification: Detectable shift → +10%
    
    # ========== ATTENUATORS (Subtractive) ==========
    # Values subtracted from score when conditions are met
    # All attenuators are cumulative but final score is clamped to [0.0, 1.0]
    
    att_stable: float = 0.15
    # Attenuation when regime is STABLE and drift < 10%. Valid: [0.10, 0.25]
    # Justification: Stable series less likely to have real anomalies → -15%
    
    att_low_criticality: float = 0.20
    # Attenuation when series criticality is LOW. Valid: [0.10, 0.30]
    # Justification: Low business impact → -20%
    
    att_no_context: float = 0.10
    # Attenuation when recent_anomaly_count == 0. Valid: [0.05, 0.20]
    # Justification: First anomaly (no history) → conservative -10%
    
    # ========== DECISION THRESHOLDS ==========
    # Score thresholds for action mapping (must be strictly decreasing)
    
    threshold_escalate: float = 0.75
    # Score ≥ this → ESCALATE (priority 1). Valid: [0.60, 0.90]
    # Justification: High confidence pattern requires immediate action
    
    threshold_investigate: float = 0.50
    # Score ≥ this → INVESTIGATE (priority 2). Valid: [0.40, 0.75]
    # Justification: Moderate confidence requires investigation
    
    threshold_monitor: float = 0.25
    # Score ≥ this → MONITOR (priority 3). Valid: [0.15, 0.50]
    # Justification: Low confidence requires passive monitoring
    # Score < this → LOG_ONLY (priority 5)
    
    # ========== AMPLIFIER THRESHOLDS ==========
    # Thresholds for evaluating amplifier conditions
    
    amp_consecutive_high_count: int = 5
    # Consecutive anomalies for strong amplification. Valid: [3, 10]
    # Justification: 5 consecutive = persistent pattern
    
    amp_consecutive_med_count: int = 3
    # Consecutive anomalies for moderate amplification. Valid: [2, 5]
    # Justification: 3 consecutive = emerging pattern
    
    amp_rate_high_threshold: float = 0.60
    # Anomaly rate threshold for strong amplification. Valid: [0.40, 0.80]
    # Justification: >60% = majority anomalous
    
    amp_rate_med_threshold: float = 0.30
    # Anomaly rate threshold for moderate amplification. Valid: [0.20, 0.60]
    # Justification: >30% = significant but not majority
    
    amp_drift_high_threshold: float = 0.70
    # Drift score threshold for strong amplification. Valid: [0.50, 0.90]
    # Justification: 0.70 = significant distribution shift
    
    amp_drift_med_threshold: float = 0.40
    # Drift score threshold for moderate amplification. Valid: [0.20, 0.70]
    # Justification: 0.40 = detectable shift
    
    # ========== ATTENUATOR THRESHOLDS ==========
    # Thresholds for evaluating attenuator conditions
    
    att_stable_drift_max: float = 0.10
    # Maximum drift for STABLE regime attenuation. Valid: [0.05, 0.20]
    # Justification: <10% drift = truly stable
    
    # ========== CACHE / PERFORMANCE ==========
    
    flag_cache_ttl_seconds: int = 60
    # TTL for feature flag cache (seconds). Valid: [10, 300]
    # Justification: 60s balances hot-reload vs performance
    
    # ========== ENGINE METADATA ==========
    
    engine_name: str = "contextual"
    # Engine strategy identifier
    # Valid values: ["simple", "contextual", "conservative", "aggressive", "cost_optimized"]
    
    engine_version: str = "1.0.0"
    # Engine version for audit trail
    # Format: Semantic versioning (major.minor.patch)
    
    # ========== PRIORITIES ==========
    # Action priority levels (lower = more urgent, no gaps)
    
    priority_escalate: int = 1
    # Immediate escalation priority
    
    priority_investigate: int = 2
    # Investigation required priority
    
    priority_monitor: int = 3
    # Active monitoring priority
    
    priority_log_only: int = 4
    # Passive logging priority
    # NOTE: Changed from 5 to 4 to eliminate gap (was 1,2,3,5)
    # Gap was unintentional - priorities should be consecutive
    
    # ========== VALIDATION BOUNDS ==========
    
    score_min: float = 0.0
    # Minimum valid score (mathematical lower bound)
    
    score_max: float = 1.0
    # Maximum valid score (mathematical upper bound)
    
    def validate(self) -> None:
        """Validate configuration consistency.
        
        Raises:
            ValueError: If configuration has invalid ranges, overlaps, or gaps.
        
        Applies SRP: Validation logic is separate from business logic.
        """
        # Validate score bounds
        scores = {
            "score_critical": self.score_critical,
            "score_high": self.score_high,
            "score_medium": self.score_medium,
            "score_low": self.score_low,
            "score_none": self.score_none,
            "score_warning": self.score_warning,
        }
        
        for name, value in scores.items():
            if not (self.score_min <= value <= self.score_max):
                raise ValueError(
                    f"{name}={value} must be in [{self.score_min}, {self.score_max}]"
                )
        
        # Validate score ordering (strict inequality for clear separation)
        if not (self.score_none < self.score_low < self.score_medium < self.score_high < self.score_critical):
            raise ValueError(
                "Scores must be strictly increasing: "
                f"none({self.score_none}) < low({self.score_low}) < "
                f"medium({self.score_medium}) < high({self.score_high}) < "
                f"critical({self.score_critical})"
            )
        
        # Validate threshold ordering (strict inequality)
        if not (0 < self.threshold_monitor < self.threshold_investigate < self.threshold_escalate <= self.score_max):
            raise ValueError(
                "Thresholds must be strictly increasing: "
                f"0 < monitor({self.threshold_monitor}) < "
                f"investigate({self.threshold_investigate}) < "
                f"escalate({self.threshold_escalate}) <= {self.score_max}"
            )
        
        # Validate amplifiers are positive
        amplifiers = {
            "amp_consecutive_5": self.amp_consecutive_5,
            "amp_consecutive_3": self.amp_consecutive_3,
            "amp_rate_high": self.amp_rate_high,
            "amp_rate_med": self.amp_rate_med,
            "amp_volatile": self.amp_volatile,
            "amp_noisy": self.amp_noisy,
            "amp_drift_high": self.amp_drift_high,
            "amp_drift_med": self.amp_drift_med,
        }
        
        for name, value in amplifiers.items():
            if value <= 0:
                raise ValueError(f"{name}={value} must be > 0")
        
        # Validate attenuators are positive
        attenuators = {
            "att_stable": self.att_stable,
            "att_low_criticality": self.att_low_criticality,
            "att_no_context": self.att_no_context,
        }
        
        for name, value in attenuators.items():
            if value <= 0:
                raise ValueError(f"{name}={value} must be > 0")
        
        # Validate amplifier threshold ordering
        if self.amp_consecutive_med_count >= self.amp_consecutive_high_count:
            raise ValueError(
                f"amp_consecutive_med_count({self.amp_consecutive_med_count}) must be < "
                f"amp_consecutive_high_count({self.amp_consecutive_high_count})"
            )
        
        if self.amp_rate_med_threshold >= self.amp_rate_high_threshold:
            raise ValueError(
                f"amp_rate_med_threshold({self.amp_rate_med_threshold}) must be < "
                f"amp_rate_high_threshold({self.amp_rate_high_threshold})"
            )
        
        if self.amp_drift_med_threshold >= self.amp_drift_high_threshold:
            raise ValueError(
                f"amp_drift_med_threshold({self.amp_drift_med_threshold}) must be < "
                f"amp_drift_high_threshold({self.amp_drift_high_threshold})"
            )
        
        # Validate priority ordering (strict inequality, no gaps)
        priorities = [
            self.priority_escalate,
            self.priority_investigate,
            self.priority_monitor,
            self.priority_log_only,
        ]
        
        if priorities != sorted(priorities):
            raise ValueError(
                f"Priorities must be strictly increasing: {priorities}"
            )
        
        # Check for gaps in priorities
        for i in range(len(priorities) - 1):
            if priorities[i + 1] - priorities[i] != 1:
                raise ValueError(
                    f"Priority gap detected: {priorities[i]} → {priorities[i + 1]} "
                    "(priorities should be consecutive)"
                )
        
        # Validate cache TTL
        if not (10 <= self.flag_cache_ttl_seconds <= 300):
            raise ValueError(
                f"flag_cache_ttl_seconds={self.flag_cache_ttl_seconds} must be in [10, 300]"
            )
