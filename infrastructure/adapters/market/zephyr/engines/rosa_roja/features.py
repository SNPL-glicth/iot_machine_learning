"""Market Feature Extractor — converts market observations to Rosa Roja delta_state vector."""

from __future__ import annotations

from collections import deque
import numpy as np

from iot_machine_learning.domain.entities.market.observations import (
    Candle, MarketObservation, OrderBookSnapshot, Quote, Trade,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja.features_state import FeatureState


class MarketFeatureExtractor:
    """Extracts feature vector (delta_state) from market observations for Rosa Roja with zero GC churn."""

    def __init__(self, window: int = 100):
        self.state = FeatureState(window=window)
        self._last_mid: float | None = None
        self._trade_count = 0
        self._last_trade_time: float | None = None
        self._state_buffer = np.zeros(10, dtype=np.float32)

    def process(self, observation: MarketObservation) -> tuple[np.ndarray, float]:
        """Processes observation reusing pre-allocated numpy array to prevent hot-path GC pressure."""
        current_ts = observation.timestamp
        delta_time = current_ts - self.state.last_timestamp if self.state.last_timestamp is not None else 0.0
        self.state.last_timestamp = current_ts

        self._state_buffer.fill(0.0)
        buf = self._state_buffer

        if isinstance(observation, Candle): self._process_candle(observation, buf)
        elif isinstance(observation, Quote): self._process_quote(observation, buf)
        elif isinstance(observation, Trade): self._process_trade(observation, buf)
        elif isinstance(observation, OrderBookSnapshot): self._process_orderbook(observation, buf)
        return buf, delta_time

    def _process_candle(self, c: Candle, buf: np.ndarray) -> None:
        mid, close, open_, high, low, volume = (c.high + c.low) / 2.0, c.close, c.open, c.high, c.low, c.volume
        if self._last_mid is not None and self._last_mid > 0:
            log_ret = float(np.log(close / self._last_mid))
            buf[0] = np.clip(log_ret, -0.1, 0.1)
            self.state.returns.append(log_ret)
        self._last_mid = close
        self.state.mid_prices.append(close)
        if len(self.state.returns) >= 10: buf[1] = float(np.std(self.state.returns))
        if open_ > 0: buf[7] = (close - open_) / open_
        if close > 0: buf[8] = (high - low) / close
        if len(self.state.volumes) >= 20:
            v_mean, v_std = np.mean(self.state.volumes), np.std(self.state.volumes) + 1e-8
            buf[3] = np.clip((volume - v_mean) / v_std, -5, 5)
        self.state.volumes.append(volume)

    def _process_quote(self, q: Quote, buf: np.ndarray) -> None:
        mid, spread, bid_vol, ask_vol = q.midpoint, q.spread, q.bid_size, q.ask_size
        if self._last_mid is not None and self._last_mid > 0:
            log_ret = float(np.log(mid / self._last_mid))
            buf[0] = np.clip(log_ret, -0.1, 0.1)
            self.state.returns.append(log_ret)
        self._last_mid = mid
        self.state.mid_prices.append(mid)
        if len(self.state.returns) >= 10: buf[1] = float(np.std(self.state.returns))
        if mid > 0:
            buf[2] = (spread / mid) * 10000
            self.state.spreads.append(spread / mid)
        total = bid_vol + ask_vol
        if total > 0:
            imb = (bid_vol - ask_vol) / total
            buf[5] = np.clip(imb, -1.0, 1.0)
            self.state.imbalances.append(imb)

    def _process_trade(self, t: Trade, buf: np.ndarray) -> None:
        price, size, side = t.price, t.size, t.taker_side
        if self._last_mid is not None and self._last_mid > 0:
            log_ret = float(np.log(price / self._last_mid))
            buf[0] = np.clip(log_ret, -0.1, 0.1)
            self.state.returns.append(log_ret)
        self._last_mid = price
        self.state.mid_prices.append(price)
        if len(self.state.returns) >= 10: buf[1] = float(np.std(self.state.returns))
        self.state.volumes.append(size)
        if len(self.state.volumes) >= 20:
            v_mean, v_std = np.mean(self.state.volumes), np.std(self.state.volumes) + 1e-8
            buf[3] = np.clip((size - v_mean) / v_std, -5, 5)
        if side == "buy":
            self.state.buy_volumes.append(size)
            self.state.sell_volumes.append(0.0)
        elif side == "sell":
            self.state.buy_volumes.append(0.0)
            self.state.sell_volumes.append(size)
        else:
            self.state.buy_volumes.append(size / 2)
            self.state.sell_volumes.append(size / 2)
        if len(self.state.buy_volumes) >= 50:
            bv, sv = sum(self.state.buy_volumes), sum(self.state.sell_volumes)
            if bv + sv > 0: buf[6] = abs(bv - sv) / (bv + sv)
        self._trade_count += 1
        if self._last_trade_time is not None: buf[9] = min(self._trade_count / 100.0, 1.0)
        self._last_trade_time = t.timestamp if hasattr(t, "timestamp") else None

    def _process_orderbook(self, ob: OrderBookSnapshot, buf: np.ndarray) -> None:
        bid_vol, ask_vol = sum(lvl[1] for lvl in ob.bids), sum(lvl[1] for lvl in ob.asks)
        total = bid_vol + ask_vol
        if total > 0:
            imb = (bid_vol - ask_vol) / total
            buf[5] = np.clip(imb, -1.0, 1.0)
            self.state.imbalances.append(imb)
        if ob.best_bid and ob.best_ask and ob.best_bid > 0:
            spread, mid = ob.best_ask - ob.best_bid, (ob.best_bid + ob.best_ask) / 2
            buf[2] = (spread / mid) * 10000
            self.state.spreads.append(spread / mid)

    def get_state_summary(self) -> dict:
        return {
            "mid_prices": len(self.state.mid_prices), "returns": len(self.state.returns),
            "volatility": float(np.std(self.state.returns)) if len(self.state.returns) >= 10 else 0.0,
            "volumes": len(self.state.volumes), "spreads": len(self.state.spreads),
            "buy_volumes": len(self.state.buy_volumes), "sell_volumes": len(self.state.sell_volumes),
            "imbalances": len(self.state.imbalances), "last_timestamp": self.state.last_timestamp,
        }