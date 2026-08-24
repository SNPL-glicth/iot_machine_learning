"""Market Feature Extractor — converts market observations to Rosa Roja delta_state vector."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional, Deque
import numpy as np

from iot_machine_learning.domain.entities.market.observations import (
    MarketObservation, Candle, Quote, Trade, OrderBookSnapshot
)


@dataclass
class FeatureState:
    """Running state for incremental feature computation."""
    # Price history for volatility
    mid_prices: Deque[float]
    returns: Deque[float]
    # Volume history
    volumes: Deque[float]
    # Spread history
    spreads: Deque[float]
    # VPIN computation
    buy_volumes: Deque[float]
    sell_volumes: Deque[float]
    # Order book imbalance
    imbalances: Deque[float]
    # Timestamp tracking
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


class MarketFeatureExtractor:
    """
    Extracts feature vector (delta_state) from market observations for Rosa Roja.
    
    Features extracted (in order):
    0. log_return (price change)
    1. volatility (rolling std of returns)
    2. spread_bps (spread in basis points)
    3. volume (normalized)
    4. volume_imbalance (buy - sell / total)
    5. order_book_imbalance (bid_vol - ask_vol / total)
    6. vpin (volume-synchronized probability of informed trading)
    7. candle_body (close - open / open)
    8. candle_range (high - low / close)
    9. trade_intensity (trades per second)
    
    Output: delta_state vector of shape (10,) + delta_time
    """
    
    def __init__(self, window: int = 100):
        self.state = FeatureState(window=window)
        self._last_mid = None
        self._trade_count = 0
        self._last_trade_time = None
    
    def process(self, observation: MarketObservation) -> tuple[np.ndarray, float]:
        """
        Process a market observation and return (delta_state, delta_time).
        
        Args:
            observation: Candle, Quote, Trade, or OrderBookSnapshot
            
        Returns:
            (delta_state_vector, delta_time_seconds)
        """
        current_ts = observation.timestamp
        delta_time = 0.0
        if self.state.last_timestamp is not None:
            delta_time = current_ts - self.state.last_timestamp
        self.state.last_timestamp = current_ts
        
        delta_state = np.zeros(10, dtype=np.float32)
        
        if isinstance(observation, Candle):
            delta_state = self._process_candle(observation, delta_state)
        elif isinstance(observation, Quote):
            delta_state = self._process_quote(observation, delta_state)
        elif isinstance(observation, Trade):
            delta_state = self._process_trade(observation, delta_state)
        elif isinstance(observation, OrderBookSnapshot):
            delta_state = self._process_orderbook(observation, delta_state)
        
        return delta_state, delta_time
    
    def _process_candle(self, candle: Candle, delta_state: np.ndarray) -> np.ndarray:
        """Extract features from OHLCV candle."""
        mid = (candle.high + candle.low) / 2.0
        close = candle.close
        open_ = candle.open
        high = candle.high
        low = candle.low
        volume = candle.volume
        
        # 0. log_return
        if self._last_mid is not None and self._last_mid > 0:
            log_ret = np.log(close / self._last_mid)
            delta_state[0] = np.clip(log_ret, -0.1, 0.1)
            self.state.returns.append(log_ret)
        self._last_mid = close
        self.state.mid_prices.append(close)
        
        # 1. volatility (rolling std of returns)
        if len(self.state.returns) >= 10:
            delta_state[1] = float(np.std(self.state.returns))
        
        # 7. candle_body
        if open_ > 0:
            delta_state[7] = (close - open_) / open_
        
        # 8. candle_range
        if close > 0:
            delta_state[8] = (high - low) / close
        
        # 3. volume (normalized)
        if len(self.state.volumes) >= 20:
            vol_mean = np.mean(self.state.volumes)
            vol_std = np.std(self.state.volumes) + 1e-8
            delta_state[3] = np.clip((volume - vol_mean) / vol_std, -5, 5)
        self.state.volumes.append(volume)
        
        return delta_state
    
    def _process_quote(self, quote: Quote, delta_state: np.ndarray) -> np.ndarray:
        """Extract features from quote (bid/ask)."""
        mid = quote.midpoint
        spread = quote.spread
        bid_vol = quote.bid_size
        ask_vol = quote.ask_size
        
        # 0. log_return
        if self._last_mid is not None and self._last_mid > 0:
            log_ret = np.log(mid / self._last_mid)
            delta_state[0] = np.clip(log_ret, -0.1, 0.1)
            self.state.returns.append(log_ret)
        self._last_mid = mid
        self.state.mid_prices.append(mid)
        
        # 1. volatility
        if len(self.state.returns) >= 10:
            delta_state[1] = float(np.std(self.state.returns))
        
        # 2. spread_bps
        if mid > 0:
            delta_state[2] = (spread / mid) * 10000  # basis points
            self.state.spreads.append(spread / mid)
        
        # 5. order_book_imbalance (top of book)
        total = bid_vol + ask_vol
        if total > 0:
            imb = (bid_vol - ask_vol) / total
            delta_state[5] = np.clip(imb, -1.0, 1.0)
            self.state.imbalances.append(imb)
        
        return delta_state
    
    def _process_trade(self, trade: Trade, delta_state: np.ndarray) -> np.ndarray:
        """Extract features from trade."""
        price = trade.price
        size = trade.size
        side = trade.taker_side
        
        # 0. log_return
        if self._last_mid is not None and self._last_mid > 0:
            log_ret = np.log(price / self._last_mid)
            delta_state[0] = np.clip(log_ret, -0.1, 0.1)
            self.state.returns.append(log_ret)
        self._last_mid = price
        self.state.mid_prices.append(price)
        
        # 1. volatility
        if len(self.state.returns) >= 10:
            delta_state[1] = float(np.std(self.state.returns))
        
        # 3. volume
        self.state.volumes.append(size)
        if len(self.state.volumes) >= 20:
            vol_mean = np.mean(self.state.volumes)
            vol_std = np.std(self.state.volumes) + 1e-8
            delta_state[3] = np.clip((size - vol_mean) / vol_std, -5, 5)
        
        # 4. volume_imbalance (VPIN component)
        if side == "buy":
            self.state.buy_volumes.append(size)
            self.state.sell_volumes.append(0.0)
        elif side == "sell":
            self.state.buy_volumes.append(0.0)
            self.state.sell_volumes.append(size)
        else:
            # Unknown side, split
            self.state.buy_volumes.append(size / 2)
            self.state.sell_volumes.append(size / 2)
        
        # 6. VPIN approximation
        if len(self.state.buy_volumes) >= 50:
            buy_vol = sum(self.state.buy_volumes)
            sell_vol = sum(self.state.sell_volumes)
            total_vol = buy_vol + sell_vol
            if total_vol > 0:
                delta_state[6] = abs(buy_vol - sell_vol) / total_vol
        
        # 9. trade_intensity
        current_time = None  # Would need timestamp from trade
        self._trade_count += 1
        if self._last_trade_time is not None:
            # Approximate: trades per observation
            delta_state[9] = min(self._trade_count / 100.0, 1.0)
        self._last_trade_time = current_time
        
        return delta_state
    
    def _process_orderbook(self, ob: OrderBookSnapshot, delta_state: np.ndarray) -> np.ndarray:
        """Extract features from order book snapshot."""
        bid_vol = sum(level[1] for level in ob.bids)
        ask_vol = sum(level[1] for level in ob.asks)
        
        # 5. order_book_imbalance
        total = bid_vol + ask_vol
        if total > 0:
            imb = (bid_vol - ask_vol) / total
            delta_state[5] = np.clip(imb, -1.0, 1.0)
            self.state.imbalances.append(imb)
        
        # 2. spread from best bid/ask
        if ob.best_bid and ob.best_ask and ob.best_bid > 0:
            spread = ob.best_ask - ob.best_bid
            mid = (ob.best_bid + ob.best_ask) / 2
            delta_state[2] = (spread / mid) * 10000
            self.state.spreads.append(spread / mid)
        
        return delta_state
    
    def get_state_summary(self) -> dict:
        """Get current feature state for debugging."""
        return {
            "mid_prices": len(self.state.mid_prices),
            "returns": len(self.state.returns),
            "volatility": float(np.std(self.state.returns)) if len(self.state.returns) >= 10 else 0.0,
            "volumes": len(self.state.volumes),
            "spreads": len(self.state.spreads),
            "buy_volumes": len(self.state.buy_volumes),
            "sell_volumes": len(self.state.sell_volumes),
            "imbalances": len(self.state.imbalances),
            "last_timestamp": self.state.last_timestamp,
        }


def extract_features(observations: list[MarketObservation]) -> np.ndarray:
    """Convenience function to extract features from a list of observations."""
    extractor = MarketFeatureExtractor()
    features = []
    for obs in observations:
        vec, dt = extractor.process(obs)
        features.append(vec)
    return np.array(features)