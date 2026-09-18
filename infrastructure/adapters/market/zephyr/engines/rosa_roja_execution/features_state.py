"""Feature state storage for market observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class FeatureState:
    """Running state for incremental feature computation."""
    mid_prices: Deque[float]
    returns: Deque[float]
    volumes: Deque[float]
    spreads: Deque[float]
    buy_volumes: Deque[float]
    sell_volumes: Deque[float]
    imbalances: Deque[float]
    last_timestamp: Optional[float] = None

    def __init__(self, window: int = 100):
        self.window = window
        self.mid_prices = deque(maxlen=window)
        self.returns = deque(maxlen=window)
        self.volumes = deque(maxlen=window)
        self.spreads = deque(maxlen=window)
        self.buy_volumes = deque(maxlen=window)
        self.sell_volumes = deque(maxlen=window)
        self.imbalances = deque(maxlen=window)
