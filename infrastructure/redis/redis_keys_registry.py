"""Redis Keys Registry — specific key patterns and TTL definitions.

Contains all key generation methods and TTL configurations.
"""

from __future__ import annotations
from typing import Optional

from .redis_keys_base import RedisKeysBase


class RedisKeysRegistry(RedisKeysBase):
    """Registry of all Redis key patterns with TTL definitions.
    
    Extends RedisKeysBase with specific key generation methods.
    """
    
    # --- Plasticity keys ---
    def plasticity(self, regime: str) -> str:
        """Key for plasticity weights per regime.

        Format: {env}:{app}:{tenant}:plasticity:{regime}
        Type: Redis Hash (engine_name -> accuracy)
        TTL: None (persistent)
        """
        return self._build_key("plasticity", regime)
    
    def plasticity_ttl(self) -> Optional[int]:
        """TTL for plasticity keys (None = persistent)."""
        return None

    # --- Error history keys ---
    def error_history(self, series_id: str, engine_name: str) -> str:
        """Key for error history per series and engine.

        Format: {env}:{app}:{tenant}:error_history:{series_id}:{engine_name}
        Type: Redis List (maxlen configured in flags)
        TTL: 7 days
        """
        return self._build_key("error_history", series_id, engine_name)
    
    def error_history_ttl(self) -> int:
        """TTL for error history keys (7 days)."""
        return 7 * 86400

    # --- Anomaly tracking keys ---
    def anomaly_track(self, series_id: str) -> str:
        """Key for anomaly tracking (sorted set by timestamp).

        Format: {env}:{app}:{tenant}:anomaly_track:{series_id}
        Type: Redis SortedSet (timestamp -> anomaly_data)
        TTL: 30 days
        """
        return self._build_key("anomaly_track", series_id)
    
    def anomaly_track_ttl(self) -> int:
        """TTL for anomaly tracking keys (30 days)."""
        return 30 * 86400

    def anomaly_consecutive(self, series_id: str) -> str:
        """Key for consecutive anomaly counter.

        Format: {env}:{app}:{tenant}:anomaly_consecutive:{series_id}
        Type: Redis String (INCR/DEL operations)
        TTL: 1 hour
        """
        return self._build_key("anomaly_consecutive", series_id)
    
    def anomaly_consecutive_ttl(self) -> int:
        """TTL for consecutive anomaly counter (1 hour)."""
        return 3600

    # --- Alert suppression keys ---
    def last_alert(self, series_id: str) -> str:
        """Key for last emitted alert per series.

        Format: {env}:{app}:{tenant}:last_alert:{series_id}
        Type: Redis String with TTL (JSON blob)
        TTL: 1 hour
        """
        return self._build_key("last_alert", series_id)
    
    def last_alert_ttl(self) -> int:
        """TTL for last alert keys (1 hour)."""
        return 3600

    def suppressed(self, series_id: str) -> str:
        """Key for suppressed alert counter per series.

        Format: {env}:{app}:{tenant}:suppressed:{series_id}
        Type: Redis String (counter with TTL)
        TTL: 1 hour
        """
        return self._build_key("suppressed", series_id)
    
    def suppressed_ttl(self) -> int:
        """TTL for suppressed alert keys (1 hour)."""
        return 3600

    # --- Rate limiting keys ---
    def rate_limit_tenant(self) -> str:
        """Key for tenant-level rate limiting.
        
        Format: {env}:{app}:{tenant}:rate_limit:tenant
        Type: Redis String (counter)
        TTL: 60 seconds (sliding window)
        """
        return self._build_key("rate_limit", "tenant")
    
    def rate_limit_series(self, series_id: str) -> str:
        """Key for series-level rate limiting.
        
        Format: {env}:{app}:{tenant}:rate_limit:series:{series_id}
        Type: Redis String (counter)
        TTL: 60 seconds (sliding window)
        """
        return self._build_key("rate_limit", "series", series_id)
    
    def rate_limit_ttl(self) -> int:
        """TTL for rate limit keys (60 seconds)."""
        return 60
    
    # --- Pattern registry for wildcards ---
    def pattern_all_error_history(self, series_id: str) -> str:
        """Pattern for all error history keys of a series.

        For use with SCAN operations (not for production queries).
        """
        return f"{self.env}:{self.app_name}:{self.tenant_id}:error_history:{self._sanitize(series_id, 'series_id')}:*"

    def pattern_all_plasticity(self) -> str:
        """Pattern for all plasticity regime keys.

        For use with SCAN operations (not for production queries).
        """
        return f"{self.env}:{self.app_name}:{self.tenant_id}:plasticity:*"
    
    def pattern_tenant_all(self) -> str:
        """Pattern for all keys belonging to this tenant.
        
        For use with SCAN operations (admin/cleanup only).
        """
        return f"{self.env}:{self.app_name}:{self.tenant_id}:*"
