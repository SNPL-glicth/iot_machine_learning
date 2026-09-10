"""Constants and enumeration values for Alpaca Order Client."""

from __future__ import annotations


class OrderClientConstants:
    """Constantes de endpoints, tipos de órdenes y estados de Alpaca."""

    BASE_URL = "https://paper-api.alpaca.markets/v2"
    LIVE_BASE_URL = "https://api.alpaca.markets/v2"

    # Order types
    ORDER_TYPE_MARKET = "market"
    ORDER_TYPE_LIMIT = "limit"
    ORDER_TYPE_STOP = "stop"
    ORDER_TYPE_STOP_LIMIT = "stop_limit"
    ORDER_TYPE_TRAILING_STOP = "trailing_stop"

    # Time in force
    TIME_IN_FORCE_DAY = "day"
    TIME_IN_FORCE_GTC = "gtc"
    TIME_IN_FORCE_OPG = "opg"
    TIME_IN_FORCE_CLS = "cls"
    TIME_IN_FORCE_IOC = "ioc"
    TIME_IN_FORCE_FOK = "fok"

    # Order statuses
    STATUS_NEW = "new"
    STATUS_PARTIALLY_FILLED = "partially_filled"
    STATUS_FILLED = "filled"
    STATUS_DONE_FOR_DAY = "done_for_day"
    STATUS_CANCELED = "canceled"
    STATUS_EXPIRED = "expired"
    STATUS_REPLACED = "replaced"
    STATUS_PENDING_CANCEL = "pending_cancel"
    STATUS_PENDING_REPLACE = "pending_replace"
    STATUS_ACCEPTED = "accepted"
    STATUS_PENDING_NEW = "pending_new"
    STATUS_ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    STATUS_STOPPED = "stopped"
    STATUS_SUSPENDED = "suspended"
    STATUS_CALCULATED = "calculated"
