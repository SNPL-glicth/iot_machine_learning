"""EngineReliabilityTracker \u2014 Beta-Bernoulli per-engine reliability model.

Per ``(series_id, engine_name)`` maintains a Beta(\u03b1, \u03b2) posterior over
``P(broken)``. After each outcome the threshold is the ``percentile``-th
percentile from :class:`EngineErrorStore`; ``beta`` increments when the
observed error exceeds it, else ``alpha``. Recovery is automatic.

Key:  ``engine_reliability:{series_id}:{engine_name}`` \u2014 Redis Hash
(fields ``alpha``, ``beta``), TTL reset on every write. In-memory
thread-safe fallback when Redis is absent or raises. Public methods
never raise. SRP: owns posterior state only.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from ..error_store import EngineErrorStore

logger = logging.getLogger(__name__)


_DEFAULT_PREFIX: str = "engine_reliability"
_DEFAULT_TTL_SECONDS: int = 7 * 24 * 3600  # 7 days
_SCHEMA_VERSION: str = "1"
_DEFAULT_PERCENTILE: float = 75.0
_DEFAULT_UNRELIABLE_THRESHOLD: float = 0.95
_DEFAULT_ALPHA_PRIOR: float = 1.0
_DEFAULT_BETA_PRIOR: float = 1.0
_DEFAULT_MAX_PAIRS: int = 10_000


class EngineReliabilityTracker:
    """Beta-Bernoulli reliability estimator with Redis + in-memory fallback."""

    def __init__(
        self,
        error_store: EngineErrorStore,
        redis_client: Optional[Any] = None,
        *,
        key_prefix: str = _DEFAULT_PREFIX,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        percentile: float = _DEFAULT_PERCENTILE,
        unreliable_threshold: float = _DEFAULT_UNRELIABLE_THRESHOLD,
        alpha_prior: float = _DEFAULT_ALPHA_PRIOR,
        beta_prior: float = _DEFAULT_BETA_PRIOR,
        max_pairs: int = _DEFAULT_MAX_PAIRS,
    ) -> None:
        if not 0.0 < percentile < 100.0:
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        if not 0.0 < unreliable_threshold < 1.0:
            raise ValueError(
                f"unreliable_threshold must be in (0, 1), got {unreliable_threshold}"
            )
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("alpha_prior and beta_prior must be > 0")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")

        self._store = error_store
        self._redis = redis_client
        self._prefix = key_prefix
        self._ttl = int(ttl_seconds)
        self._percentile = float(percentile)
        self._threshold = float(unreliable_threshold)
        self._alpha0 = float(alpha_prior)
        self._beta0 = float(beta_prior)
        self._max_pairs = int(max_pairs)
        self._schema_version = _SCHEMA_VERSION

        # In-memory stores (alpha, beta, schema_version)
        self._memory: "OrderedDict[Tuple[str, str], Tuple[float, float, str]]" = OrderedDict()
        self._lock = threading.RLock()

    # -- public API ---------------------------------------------------

    def record_outcome(self, series_id: str, engine_name: str, error: float) -> None:
        """Update the Beta posterior with one observation. Never raises.

        Silently skips the update when there is not enough history in
        :class:`EngineErrorStore` to compute a meaningful percentile
        threshold (returned as ``0.0``).
        """
        try:
            err = float(error)
        except (TypeError, ValueError):
            return
        if not math.isfinite(err) or err < 0.0:
            return

        threshold = self._store.get_percentile(series_id, engine_name, self._percentile)
        if threshold <= 0.0:
            return

        alpha, beta = self._load(series_id, engine_name)
        if err > threshold:
            beta += 1.0
        else:
            alpha += 1.0
        self._save(series_id, engine_name, alpha, beta)

    def is_reliable(self, series_id: str, engine_name: str) -> bool:
        """True iff ``P(broken) <= unreliable_threshold``.

        With the uninformative prior a fresh ``(series, engine)`` pair
        starts at ``P(broken) = 0.5`` which is below the default 0.95
        gate \u2014 new engines are reliable by default until evidence
        accumulates.
        """
        return self.p_broken(series_id, engine_name) <= self._threshold

    def p_broken(self, series_id: str, engine_name: str) -> float:
        """Posterior mean ``\u03b2 / (\u03b1 + \u03b2)`` \u2014 exposed for observability."""
        alpha, beta = self._load(series_id, engine_name)
        total = alpha + beta
        return (beta / total) if total > 0.0 else 0.0

    def reset(self, series_id: str, engine_name: str) -> None:
        """Drop the posterior back to the prior. Never raises."""
        self._save(series_id, engine_name, self._alpha0, self._beta0)

    # -- helpers ------------------------------------------------------

    def _key(self, series_id: str, engine_name: str) -> str:
        return f"{self._prefix}:{series_id}:{engine_name}"

    def _load(self, series_id: str, engine_name: str) -> Tuple[float, float]:
        if self._redis is not None:
            try:
                raw = self._redis.hgetall(self._key(series_id, engine_name))
                if raw:
                    stored_version = raw.get(
                        b"schema_version", raw.get("schema_version")
                    )
                    if stored_version is None:
                        stored_version = ""
                    stored_version = (
                        stored_version.decode()
                        if isinstance(stored_version, bytes)
                        else str(stored_version)
                    )
                    if stored_version != self._schema_version:
                        logger.warning(
                            "reliability_schema_mismatch",
                            extra={
                                "series_id": series_id,
                                "engine": engine_name,
                                "stored_version": stored_version,
                                "expected_version": self._schema_version,
                            },
                        )
                        return self._alpha0, self._beta0
                    alpha = float(raw.get(b"alpha", raw.get("alpha", self._alpha0)))
                    beta = float(raw.get(b"beta", raw.get("beta", self._beta0)))
                    return alpha, beta
            except Exception as exc:  # pragma: no cover \u2014 defensive
                logger.warning(
                    "reliability_redis_read_failed",
                    extra={"series_id": series_id, "engine": engine_name, "error": str(exc)},
                )
        with self._lock:
            entry = self._memory.get((series_id, engine_name))
            if entry is None:
                return self._alpha0, self._beta0
            alpha, beta, version = entry
            if version != self._schema_version:
                logger.warning(
                    "reliability_schema_mismatch",
                    extra={
                        "series_id": series_id,
                        "engine": engine_name,
                        "stored_version": version,
                        "expected_version": self._schema_version,
                    },
                )
                return self._alpha0, self._beta0
            return alpha, beta

    def _save(
        self, series_id: str, engine_name: str, alpha: float, beta: float
    ) -> None:
        if self._redis is not None:
            try:
                key = self._key(series_id, engine_name)
                pipe = self._redis.pipeline()
                pipe.hset(
                    key,
                    mapping={
                        "alpha": alpha,
                        "beta": beta,
                        "schema_version": self._schema_version,
                    },
                )
                pipe.expire(key, self._ttl)
                pipe.execute()
                return
            except Exception as exc:  # pragma: no cover \u2014 defensive
                logger.warning(
                    "reliability_redis_write_failed",
                    extra={"series_id": series_id, "engine": engine_name, "error": str(exc)},
                )
        with self._lock:
            pair = (series_id, engine_name)
            if pair not in self._memory and len(self._memory) >= self._max_pairs:
                self._memory.popitem(last=False)
            self._memory[pair] = (alpha, beta, self._schema_version)
            self._memory.move_to_end(pair)
