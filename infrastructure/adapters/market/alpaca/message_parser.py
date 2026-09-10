"""Message parser and entity mapper for Alpaca WebSocket messages."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from iot_machine_learning.domain.entities.market import DataStatus
from iot_machine_learning.domain.entities.market.observations import Candle, MarketObservation, Quote, Trade
from iot_machine_learning.infrastructure.adapters.market.alpaca.feed_models import FeedStats
from iot_machine_learning.infrastructure.adapters.market.alpaca.helpers import as_float, iso_to_epoch, require

logger = logging.getLogger(__name__)

INTERVAL_MAP = {
    "1Min": 60, "5Min": 300, "15Min": 900, "30Min": 1800,
    "1Hour": 3600, "1Day": 86400,
}


def create_trade(data: Dict, default_symbol: str, receive_time: float) -> Trade:
    """Crea Trade desde mensaje Alpaca."""
    raw_conditions = data.get("c", ())
    conditions = tuple(str(c).strip() for c in raw_conditions if str(c).strip())
    return Trade(
        symbol=str(data.get("S", default_symbol)),
        timestamp=iso_to_epoch(str(require(data, "t"))),
        data_status=DataStatus.REALTIME,
        source_provider="alpaca",
        price=as_float(require(data, "p"), "p"),
        size=as_float(require(data, "s"), "s"),
        trade_id=str(data.get("i", "")),
        taker_side=None,
        conditions=conditions,
        tape=str(data.get("z", "A")),
        corrected=bool(data.get("u", False)),
    )


def create_quote(data: Dict, default_symbol: str, receive_time: float) -> Quote:
    """Crea Quote desde mensaje Alpaca."""
    raw_conditions = data.get("c", ())
    conditions = tuple(str(c).strip() for c in raw_conditions if str(c).strip())
    bx = str(data.get("bx", "V"))
    ax = str(data.get("ax", "V"))
    return Quote(
        symbol=str(data.get("S", default_symbol)),
        timestamp=iso_to_epoch(str(require(data, "t"))),
        data_status=DataStatus.REALTIME,
        source_provider="alpaca",
        venue=bx,
        bid=as_float(require(data, "bp"), "bp"),
        bid_size=as_float(require(data, "bs"), "bs"),
        ask=as_float(require(data, "ap"), "ap"),
        ask_size=as_float(require(data, "as"), "as"),
        bid_exchange=bx,
        ask_exchange=ax,
        conditions=conditions,
        tape=str(data.get("z", "A")),
    )


def create_candle(data: Dict, default_symbol: str, bar_interval: str, receive_time: float) -> Candle:
    """Crea Candle desde mensaje Alpaca."""
    interval_seconds = INTERVAL_MAP.get(bar_interval, 60)
    return Candle(
        symbol=str(data.get("S", default_symbol)),
        timestamp=iso_to_epoch(str(require(data, "t"))),
        data_status=DataStatus.REALTIME,
        source_provider="alpaca",
        venue=None,
        open=as_float(require(data, "o"), "o"),
        high=as_float(require(data, "h"), "h"),
        low=as_float(require(data, "l"), "l"),
        close=as_float(require(data, "c"), "c"),
        volume=as_float(require(data, "v"), "v"),
        interval_seconds=interval_seconds,
        vwap=as_float(data.get("vw", 0), "vw") if data.get("vw") else None,
        trade_count=int(require(data, "n", label="trade_count")),
        adjusted=False,
    )


def process_single_message(
    msg: Dict, default_symbol: str, bar_interval: str, receive_time: float, stats: FeedStats
) -> List[MarketObservation]:
    """Procesa un mensaje individual del WebSocket."""
    obs_list: List[MarketObservation] = []
    msg_type = msg.get("T", "")
    try:
        if msg_type == "t":
            stats.trades_received += 1
            obs_list.append(create_trade(msg, default_symbol, receive_time))
        elif msg_type == "q":
            stats.quotes_received += 1
            try:
                obs = create_quote(msg, default_symbol, receive_time)
            except ValueError as e:
                stats.quotes_stale_dropped += 1
                logger.debug("Stale quote dropped (validación)", extra={"error": str(e)})
                obs = None
            if obs and (obs.bid <= 0 or obs.ask <= 0 or obs.ask < obs.bid):
                stats.quotes_stale_dropped += 1
                logger.debug("Stale quote dropped", extra={"symbol": obs.symbol, "bid": obs.bid, "ask": obs.ask})
            elif obs:
                obs_list.append(obs)
        elif msg_type == "b":
            stats.bars_received += 1
            obs_list.append(create_candle(msg, default_symbol, bar_interval, receive_time))
        elif msg_type == "error":
            logger.error("Alpaca error message", extra={"msg": msg.get("msg"), "code": msg.get("code")})
    except Exception as e:
        logger.error("Error processing single message", extra={"error": str(e), "type": msg_type})
    return obs_list


def parse_raw_message(
    raw: str, default_symbol: str, bar_interval: str, receive_time: float, stats: FeedStats
) -> List[MarketObservation]:
    """Convierte mensaje raw WS a lista de MarketObservation."""
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            result = []
            for item in data:
                result.extend(process_single_message(item, default_symbol, bar_interval, receive_time, stats))
            return result
        return process_single_message(data, default_symbol, bar_interval, receive_time, stats)
    except Exception as e:
        logger.error("Error parsing message", extra={"error": str(e)})
        return []
