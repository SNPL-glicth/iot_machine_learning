"""Pattern plasticity — learns which patterns are most predictive per domain.

Second plasticity layer: tracks correlation between pattern detection and high severity.
Unlike PlasticityTracker (engine weight learning), this tracks pattern predictiveness.

Patterns tracked:
    - delta_spikes: How often spike detection correlates with high severity
    - change_points: How often drift detection precedes incidents
    - regime_changes: How often regime shifts signal problems

Pattern weights stored per domain: {domain: {pattern_name: weight}}

Design:
    - In-memory only (no persistence) — resets on restart
    - Exponential moving average for smoothness
    - Graceful-fail: if update fails, analysis continues
    - Thread-safe for concurrent use
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional
from threading import Lock

logger = logging.getLogger(__name__)

# Pattern names
PATTERN_DELTA_SPIKES = "delta_spikes"
PATTERN_CHANGE_POINTS = "change_points"
PATTERN_REGIME_CHANGES = "regime_changes"

ALL_PATTERNS = [
    PATTERN_DELTA_SPIKES,
    PATTERN_CHANGE_POINTS,
    PATTERN_REGIME_CHANGES,
]

# EMA smoothing factor for pattern weight updates
_ALPHA: float = 0.2

# Minimum weight floor to prevent complete suppression
_MIN_WEIGHT: float = 0.1

# Maximum domains to track before LRU eviction
_MAX_DOMAINS: int = 20

# Severity threshold for "high severity" classification
_HIGH_SEVERITY_THRESHOLD: float = 0.7


class PatternPlasticityTracker:
    """Tracks per-domain pattern predictiveness and computes adaptive weights.

    Learns which patterns (delta_spikes, change_points, regime_changes) are most
    predictive of high-severity incidents for each domain.

    Thread-safe for concurrent updates.

    Attributes:
        _weights: Dict[domain][pattern_name] → smoothed predictiveness score
        _alpha: EMA smoothing factor
        _min_weight: Minimum weight floor
        _max_domains: Maximum domains before LRU eviction
        _domain_last_access: Dict[domain] → timestamp for LRU
        _lock: Thread safety lock
    """

    def __init__(
        self,
        alpha: float = _ALPHA,
        min_weight: float = _MIN_WEIGHT,
        max_domains: int = _MAX_DOMAINS,
    ) -> None:
        """Initialize pattern plasticity tracker.

        Args:
            alpha: EMA smoothing factor (0-1)
            min_weight: Minimum weight floor
            max_domains: Maximum domains to track before LRU eviction
        """
        self._weights: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {p: 1.0 / len(ALL_PATTERNS) for p in ALL_PATTERNS}
        )
        self._alpha = max(0.0, min(1.0, alpha))
        self._min_weight = max(0.0, min(1.0, min_weight))
        self._max_domains = max(1, max_domains)
        self._domain_last_access: Dict[str, float] = {}
        self._lock = Lock()

    def record_pattern_outcome(
        self,
        domain: str,
        pattern_name: str,
        was_predictive: bool,
    ) -> None:
        """Record whether a pattern was predictive for this analysis.

        Args:
            domain: Domain (infrastructure, security, trading, etc.)
            pattern_name: Pattern identifier (delta_spikes, change_points, regime_changes)
            was_predictive: True if pattern detection correlated with high severity

        Gracefully handles errors without raising exceptions.
        """
        if not domain or not pattern_name:
            return

        if pattern_name not in ALL_PATTERNS:
            logger.warning(
                f"unknown_pattern_name: {pattern_name}",
                extra={"domain": domain, "pattern": pattern_name},
            )
            return

        try:
            with self._lock:
                self._evict_lru_if_needed(domain)
                self._update_weight(domain, pattern_name, was_predictive)
                self._domain_last_access[domain] = time.monotonic()
        except Exception as e:
            logger.error(
                f"pattern_plasticity_update_failed: {e}",
                extra={"domain": domain, "pattern": pattern_name},
                exc_info=True,
            )

    def get_pattern_weights(
        self,
        domain: str,
    ) -> Dict[str, float]:
        """Get current pattern weights for a domain.

        Args:
            domain: Domain to get weights for

        Returns:
            Dict[pattern_name → weight], normalized to sum to 1.0
            Falls back to uniform weights if no history for domain.
        """
        if not domain:
            return self._uniform_weights()

        try:
            with self._lock:
                domain_weights = self._weights.get(domain)
                
                if not domain_weights:
                    return self._uniform_weights()
                
                # Update access time
                self._domain_last_access[domain] = time.monotonic()
                
                # Apply minimum weight floor
                raw_weights = {
                    p: max(self._min_weight, domain_weights.get(p, self._min_weight))
                    for p in ALL_PATTERNS
                }
                
                # Normalize to sum to 1.0
                total = sum(raw_weights.values())
                if total < 1e-12:
                    return self._uniform_weights()
                
                return {p: w / total for p, w in raw_weights.items()}
        
        except Exception as e:
            logger.error(
                f"pattern_plasticity_get_weights_failed: {e}",
                extra={"domain": domain},
                exc_info=True,
            )
            return self._uniform_weights()

    def has_history(self, domain: str) -> bool:
        """Check if tracker has history for a domain.

        Args:
            domain: Domain to check

        Returns:
            True if any weight data exists for this domain
        """
        with self._lock:
            return domain in self._weights

    def reset(self, domain: Optional[str] = None) -> None:
        """Clear accumulated pattern weights.

        Args:
            domain: If provided, clear only that domain.
                If None, clear all domains.
        """
        with self._lock:
            if domain is not None:
                self._weights.pop(domain, None)
                self._domain_last_access.pop(domain, None)
            else:
                self._weights.clear()
                self._domain_last_access.clear()

    def _evict_lru_if_needed(self, domain: str) -> None:
        """Evict least recently used domain if at capacity.

        Args:
            domain: Current domain being accessed
        """
        if domain not in self._weights and len(self._weights) >= self._max_domains:
            if self._domain_last_access:
                lru_domain = min(
                    self._domain_last_access,
                    key=self._domain_last_access.get
                )
                self._weights.pop(lru_domain, None)
                self._domain_last_access.pop(lru_domain, None)

    def _update_weight(
        self,
        domain: str,
        pattern_name: str,
        was_predictive: bool,
    ) -> None:
        """Update pattern weight using EMA.

        Args:
            domain: Domain
            pattern_name: Pattern identifier
            was_predictive: Whether pattern was predictive (1.0) or not (0.0)
        """
        # Initialize domain if not exists
        if domain not in self._weights:
            self._weights[domain] = {p: 1.0 / len(ALL_PATTERNS) for p in ALL_PATTERNS}
        
        # Compute new observation value
        obs_value = 1.0 if was_predictive else 0.0
        
        # Get previous weight
        prev_weight = self._weights[domain].get(pattern_name, 1.0 / len(ALL_PATTERNS))
        
        # EMA update
        new_weight = (1.0 - self._alpha) * prev_weight + self._alpha * obs_value
        
        # Apply minimum floor
        new_weight = max(self._min_weight, new_weight)
        
        # Store updated weight
        self._weights[domain][pattern_name] = new_weight

    def _uniform_weights(self) -> Dict[str, float]:
        """Return uniform weights for all patterns.

        Returns:
            Dict with equal weight for each pattern
        """
        uniform = 1.0 / len(ALL_PATTERNS)
        return {p: uniform for p in ALL_PATTERNS}

    def get_stats(self, domain: Optional[str] = None) -> Dict[str, any]:
        """Get tracker statistics for debugging/monitoring.

        Args:
            domain: If provided, get stats for specific domain.
                If None, get global stats.

        Returns:
            Dict with tracker statistics
        """
        with self._lock:
            if domain:
                if domain not in self._weights:
                    return {"domain": domain, "has_history": False}
                
                return {
                    "domain": domain,
                    "has_history": True,
                    "weights": self._weights[domain].copy(),
                    "last_access": self._domain_last_access.get(domain, 0),
                }
            else:
                return {
                    "total_domains": len(self._weights),
                    "max_domains": self._max_domains,
                    "alpha": self._alpha,
                    "min_weight": self._min_weight,
                    "domains": list(self._weights.keys()),
                }
