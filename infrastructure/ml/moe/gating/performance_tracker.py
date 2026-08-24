"""Expert performance tracking for adaptive routing."""

from __future__ import annotations

from typing import Dict


class ExpertPerformanceTracker:
    """Track performance history per expert for routing adjustments."""
    
    def __init__(self, decay: float = 0.3, window: int = 20):
        self._decay = decay
        self._window = window
        self._scores: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
    
    def record(self, expert_id: str, error: float) -> None:
        prev = self._scores.get(expert_id, 0.0)
        count = self._counts.get(expert_id, 0)
        if count >= self._window:
            self._scores[expert_id] = prev * (1 - self._decay) + error * self._decay
        else:
            self._scores[expert_id] = (prev * count + error) / (count + 1)
        self._counts[expert_id] = min(count + 1, self._window)
    
    def get_reliability(self, expert_id: str) -> float:
        """Return reliability [0, 1] based on average historical error."""
        score = self._scores.get(expert_id)
        if score is None:
            return 0.5
        return max(0.1, 1.0 - abs(score))