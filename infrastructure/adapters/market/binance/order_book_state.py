"""OrderBookL2 -- Estado de libro L2 con sincronizacion snapshot + deltas.

Implementa el protocolo oficial de Binance para sincronizacion:
1. Fetch snapshot REST (GET /api/v3/depth) -> lastUpdateId inicial
2. Buffer eventos @depth@100ms mientras llega snapshot
3. Aplicar deltas donde u >= lastUpdateId + 1
4. Descarte eventos u < lastUpdateId + 1
4. Verificacion periodica con snapshot REST cada 30-60s
"""

from __future__ import annotations

import bisect
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PriceLevel:
    """Nivel de precio en el libro."""
    price: float
    quantity: float

    def __lt__(self, other: "PriceLevel") -> bool:
        return self.price < other.price


@dataclass
class OrderBookMetrics:
    """Metricas del libro para features."""
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    volume_imbalance: float = 0.0
    weighted_mid: float = 0.0
    microprice: float = 0.0
    timestamp: float = 0.0
    update_id: int = 0

    def to_dict(self) -> dict:
        return {
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "bid_volume": self.bid_volume,
            "ask_volume": self.ask_volume,
            "volume_imbalance": self.volume_imbalance,
            "weighted_mid": self.weighted_mid,
            "microprice": self.microprice,
            "timestamp": self.timestamp,
            "update_id": self.update_id,
        }


class OrderBookL2:
    """
    Libro de ordenes L2 (precio x cantidad) con sincronizacion Binance.

    Mantiene:
    - Bids ordenados descendente (mayor precio primero)
    - Asks ordenados ascendente (menor precio primero)
    - lastUpdateId para sincronizacion
    - Metricas computadas incrementalmente
    """

    def __init__(self, symbol: str, max_levels: int = 100):
        self.symbol = symbol.upper()
        self.max_levels = max_levels

        self._bids: Dict[float, float] = {}
        self._bid_prices: List[float] = []
        self._asks: Dict[float, float] = {}
        self._ask_prices: List[float] = []

        self._last_update_id: int = 0
        self._initialized: bool = False
        self._buffer: List[Dict] = []
        self._snapshot_in_progress: bool = False

        self._metrics: Optional[OrderBookMetrics] = None
        self._metrics_dirty: bool = True

        self.snapshot_interval_sec: float = 30.0
        self._last_snapshot_time: float = 0.0

        self.on_update: Optional[callable] = None
        self.on_metrics_update: Optional[callable] = None

        logger.info(
            "OrderBookL2 initialized",
            extra={"symbol": self.symbol, "max_levels": max_levels},
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def last_update_id(self) -> int:
        return self._last_update_id

    @property
    def best_bid(self) -> Optional[float]:
        return self._bid_prices[0] if self._bid_prices else None

    @property
    def best_ask(self) -> Optional[float]:
        return self._ask_prices[0] if self._ask_prices else None

    @property
    def mid_price(self) -> Optional[float]:
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None:
            return ask - bid
        return None

    @property
    def metrics(self) -> Optional[OrderBookMetrics]:
        if self._metrics_dirty or self._metrics is None:
            self._recompute_metrics()
        return self._metrics

    def start_snapshot_sync(self) -> None:
        self._snapshot_in_progress = True
        self._buffer.clear()
        logger.debug("Snapshot sync started", extra={"symbol": self.symbol})

    def apply_snapshot(self, snapshot: Dict) -> bool:
        try:
            last_update_id = int(snapshot["lastUpdateId"])
            bids = snapshot.get("bids", [])
            asks = snapshot.get("asks", [])

            self._bids.clear()
            self._bid_prices.clear()
            self._asks.clear()
            self._ask_prices.clear()

            for price_str, qty_str in bids:
                price = float(price_str)
                qty = float(qty_str)
                if qty > 0:
                    self._bids[price] = qty

            for price_str, qty_str in asks:
                price = float(price_str)
                qty = float(qty_str)
                if qty > 0:
                    self._asks[price] = qty

            self._bid_prices = sorted(self._bids.keys(), reverse=True)
            self._ask_prices = sorted(self._asks.keys())

            if len(self._bid_prices) > self.max_levels:
                for price in self._bid_prices[self.max_levels:]:
                    del self._bids[price]
                self._bid_prices = self._bid_prices[:self.max_levels]

            if len(self._ask_prices) > self.max_levels:
                for price in self._ask_prices[self.max_levels:]:
                    del self._asks[price]
                self._ask_prices = self._ask_prices[:self.max_levels]

            self._last_update_id = last_update_id
            self._initialized = True
            self._snapshot_in_progress = False
            self._metrics_dirty = True

            min_u = last_update_id + 1
            applied = 0
            for delta in self._buffer:
                u = delta.get("u", 0)
                if u >= min_u:
                    self._apply_delta(delta)
                    applied += 1

            self._buffer.clear()
            self._last_snapshot_time = time.time()
            self._metrics_dirty = True

            logger.info(
                "Snapshot applied",
                extra={
                    "symbol": self.symbol,
                    "last_update_id": last_update_id,
                    "buffer_processed": applied,
                    "bid_levels": len(self._bid_prices),
                    "ask_levels": len(self._ask_prices),
                },
            )

            if self.on_update:
                try:
                    self.on_update(self)
                except Exception as e:
                    logger.error("Error in on_update callback", extra={"error": str(e)})

            return True

        except Exception as e:
            logger.error("Failed to apply snapshot", extra={"symbol": self.symbol, "error": str(e)})
            self._snapshot_in_progress = False
            return False

    def apply_delta(self, delta: Dict) -> bool:
        if not self._initialized:
            self._buffer.append(delta)
            if len(self._buffer) > 1000:
                self._buffer = self._buffer[-500:]
            return False

        u = delta.get("u", 0)
        U = delta.get("U", 0)

        if U > self._last_update_id + 1:
            logger.warning(
                "Order book gap detected, scheduling resync",
                extra={
                    "symbol": self.symbol,
                    "expected_U": self._last_update_id + 1,
                    "received_U": U,
                    "last_update_id": self._last_update_id,
                },
            )
            return False

        if u < self._last_update_id + 1:
            return True

        try:
            self._apply_delta(delta)
            self._last_update_id = u
            self._metrics_dirty = True

            if self.on_update:
                try:
                    self.on_update(self)
                except Exception as e:
                    logger.error("Error in on_update callback", extra={"error": str(e)})

            return True

        except Exception as e:
            logger.error("Failed to apply delta", extra={"symbol": self.symbol, "error": str(e)})
            return False

    def _apply_delta(self, delta: Dict) -> None:
        for price_str, qty_str in delta.get("b", []):
            price = float(price_str)
            qty = float(qty_str)

            if qty == 0:
                if price in self._bids:
                    del self._bids[price]
                    idx = bisect.bisect_left(self._bid_prices, -price)
                    if idx < len(self._bid_prices) and self._bid_prices[idx] == price:
                        self._bid_prices.pop(idx)
            else:
                if price not in self._bids:
                    idx = bisect.bisect_left(self._bid_prices, -price)
                    self._bid_prices.insert(idx, price)
                self._bids[price] = qty

        for price_str, qty_str in delta.get("a", []):
            price = float(price_str)
            qty = float(qty_str)

            if qty == 0:
                if price in self._asks:
                    del self._asks[price]
                    idx = bisect.bisect_left(self._ask_prices, price)
                    if idx < len(self._ask_prices) and self._ask_prices[idx] == price:
                        self._ask_prices.pop(idx)
            else:
                if price not in self._asks:
                    idx = bisect.bisect_left(self._ask_prices, price)
                    self._ask_prices.insert(idx, price)
                self._asks[price] = qty

        if len(self._bid_prices) > self.max_levels:
            for price in self._bid_prices[self.max_levels:]:
                del self._bids[price]
            self._bid_prices = self._bid_prices[:self.max_levels]

        if len(self._ask_prices) > self.max_levels:
            for price in self._ask_prices[self.max_levels:]:
                del self._asks[price]
            self._ask_prices = self._ask_prices[:self.max_levels]

    def _recompute_metrics(self) -> None:
        if not self._initialized or not self._bid_prices or not self._ask_prices:
            self._metrics = OrderBookMetrics(timestamp=time.time(), update_id=self._last_update_id)
            self._metrics_dirty = False
            return

        bid = self._bid_prices[0]
        ask = self._ask_prices[0]
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_bps = (spread / mid) * 10000 if mid > 0 else 0.0

        top_n = min(10, len(self._bid_prices), len(self._ask_prices))

        bid_vol = sum(self._bids[p] for p in self._bid_prices[:10])
        ask_vol = sum(self._asks[p] for p in self._ask_prices[:10])

        total_vol = bid_vol + ask_vol
        vol_imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

        bid_vwap = sum(p * self._bids[p] for p in self._bid_prices[:10]) / bid_vol if bid_vol > 0 else bid
        ask_vwap = sum(p * self._asks[p] for p in self._ask_prices[:10]) / ask_vol if ask_vol > 0 else ask
        weighted_mid = (bid_vwap * bid_vol + ask_vwap * ask_vol) / (bid_vol + ask_vol) if total_vol > 0 else mid

        bid_qty = self._bids.get(bid, 0)
        ask_qty = self._asks.get(ask, 0)
        total_qty = bid_qty + ask_qty
        microprice = (bid * ask_qty + ask * bid_qty) / total_qty if total_qty > 0 else mid

        self._metrics = OrderBookMetrics(
            mid_price=mid,
            spread=spread,
            spread_bps=(spread / mid) * 10000 if mid > 0 else 0.0,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            volume_imbalance=vol_imbalance,
            weighted_mid=weighted_mid,
            microprice=microprice,
            timestamp=time.time(),
            update_id=self._last_update_id,
        )
        self._metrics_dirty = False

        if self.on_metrics_update:
            try:
                self.on_metrics_update(self._metrics)
            except Exception as e:
                logger.error("Error in on_metrics_update callback", extra={"error": str(e)})

    def get_top_levels(self, n: int = 10) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        bids = [(p, self._bids[p]) for p in self._bid_prices[:n]]
        asks = [(p, self._asks[p]) for p in self._ask_prices[:n]]
        return bids, asks

    def get_depth_at_price(self, price: float, side: str) -> float:
        if side == "bid":
            return self._bids.get(price, 0.0)
        elif side == "ask":
            return self._asks.get(price, 0.0)
        return 0.0

    def volume_at_price(self, price: float, side: str, levels: int = 1, max_distance: float = 1.0) -> float:
        prices = self._bid_prices if side == "bid" else self._ask_prices
        book = self._bids if side == "bid" else self._asks

        if not prices:
            return 0.0

        if side == "bid":
            neg_prices = [-p for p in prices]
            idx = bisect.bisect_left(neg_prices, -price)
        else:
            idx = bisect.bisect_left(prices, price)

        if idx >= len(prices):
            idx = len(prices) - 1
        elif idx > 0 and abs(prices[idx] - price) > abs(prices[idx - 1] - price):
            idx -= 1

        closest_price = prices[idx]
        if abs(closest_price - price) > max_distance:
            return 0.0

        start = max(0, idx - levels)
        end = min(len(prices), idx + levels + 1)
        return sum(book[p] for p in prices[start:end])

    def needs_resync(self) -> bool:
        if not self._initialized or self._last_snapshot_time == 0:
            return False
        return time.time() - self._last_snapshot_time > self.snapshot_interval_sec

    def get_state_snapshot(self) -> Dict:
        return {
            "symbol": self.symbol,
            "last_update_id": self._last_update_id,
            "initialized": self._initialized,
            "bids": [[str(p), str(self._bids[p])] for p in self._bid_prices],
            "asks": [[str(p), str(self._asks[p])] for p in self._ask_prices],
            "timestamp": time.time(),
        }

    def load_state_snapshot(self, state: Dict) -> bool:
        try:
            self._last_update_id = state["last_update_id"]
            self._initialized = state["initialized"]

            self._bids.clear()
            self._bid_prices.clear()
            self._asks.clear()
            self._ask_prices.clear()

            for price_str, qty_str in state.get("bids", []):
                price = float(price_str)
                qty = float(qty_str)
                self._bids[price] = qty
                self._bid_prices.append(price)

            for price_str, qty_str in state.get("asks", []):
                price = float(price_str)
                qty = float(qty_str)
                self._asks[price] = qty
                self._ask_prices.append(price)

            self._bid_prices.sort(reverse=True)
            self._ask_prices.sort()

            self._metrics_dirty = True
            return True

        except Exception as e:
            logger.error("Failed to load order book state", extra={"error": str(e)})
            return False


__all__ = [
    "OrderBookL2",
    "OrderBookMetrics",
    "PriceLevel",
]
