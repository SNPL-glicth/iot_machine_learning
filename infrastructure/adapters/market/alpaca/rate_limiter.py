"""Token bucket rate limiter for Alpaca API."""

from __future__ import annotations

import asyncio
import time
from collections import deque


class RateLimiter:
    """Rate limiter token bucket para Alpaca API."""

    def __init__(
        self,
        max_requests_per_minute: int = 200,
        max_orders_per_second: int = 10,
    ) -> None:
        self.max_rpm = max_requests_per_minute
        self.max_ops = max_orders_per_second

        self._request_times: deque = deque(maxlen=max_requests_per_minute)
        self._order_times: deque = deque(maxlen=max_orders_per_second)
        self._lock = asyncio.Lock()

    async def acquire(self, weight: int = 1) -> None:
        """Espera hasta que haya capacidad."""
        async with self._lock:
            now = time.time()

            # Rate limit per minute
            cutoff_min = now - 60
            while self._request_times and self._request_times[0] < cutoff_min:
                self._request_times.popleft()

            if len(self._request_times) >= self.max_rpm:
                wait = 60 - (now - self._request_times[0])
                await asyncio.sleep(max(0, wait) + 0.1)

            # Rate limit per second (orders)
            cutoff_sec = now - 1
            while self._order_times and self._order_times[0] < cutoff_sec:
                self._order_times.popleft()

            if len(self._order_times) >= self.max_ops:
                wait = 1 - (now - self._order_times[0])
                await asyncio.sleep(max(0, wait) + 0.01)

            self._request_times.append(now)
            self._order_times.append(now)
