"""
RegimeMetadataRegistry for managing regime configuration per sensor type.

Provides configuration with fallback chain: sensor_id -> sensor_type -> default.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .models.regime_config import RegimeConfig


class RegimeMetadataRegistry:
    """Registry for regime detection configuration per sensor type/equipment."""
    
    def __init__(self, enable_cache: bool = True):
        """
        Initialize regime metadata registry.
        
        Args:
            enable_cache: Whether to enable in-memory caching
        """
        self._enable_cache = enable_cache
        self._cache: Dict[Tuple[int, str], RegimeConfig] = {}
        
        # Sensor-specific configurations (sensor_id -> RegimeConfig)
        self._sensor_configs: Dict[int, RegimeConfig] = {}
        
        # Sensor-type configurations (sensor_type -> RegimeConfig)
        self._type_configs: Dict[str, RegimeConfig] = {}
        
        # Default configuration
        self._default_config = RegimeConfig.default()
    
    def register_sensor_config(self, sensor_id: int, config: RegimeConfig) -> None:
        """
        Register configuration for a specific sensor.
        
        Args:
            sensor_id: Sensor identifier
            config: Regime configuration
        """
        self._sensor_configs[sensor_id] = config
        self._invalidate_cache(sensor_id)
    
    def register_type_config(self, sensor_type: str, config: RegimeConfig) -> None:
        """
        Register configuration for a sensor type.
        
        Args:
            sensor_type: Sensor type (e.g., "TEMPERATURE", "PRESSURE")
            config: Regime configuration
        """
        self._type_configs[sensor_type] = config
        # Invalidate all cache entries for this type
        self._cache = {k: v for k, v in self._cache.items() if k[1] != sensor_type}
    
    def set_default_config(self, config: RegimeConfig) -> None:
        """
        Set default configuration.
        
        Args:
            config: Default regime configuration
        """
        self._default_config = config
        self._invalidate_cache()
    
    def get_config(self, sensor_id: int, sensor_type: str) -> RegimeConfig:
        """
        Get configuration for a sensor with fallback chain.
        
        Fallback order:
        1. Sensor-specific configuration
        2. Sensor-type configuration
        3. Default configuration
        
        Args:
            sensor_id: Sensor identifier
            sensor_type: Sensor type
        
        Returns:
            Regime configuration
        """
        cache_key = (sensor_id, sensor_type)
        
        # Check cache
        if self._enable_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try sensor-specific config
        config = self._sensor_configs.get(sensor_id)
        if config:
            if self._enable_cache:
                self._cache[cache_key] = config
            return config
        
        # Try sensor-type config
        config = self._type_configs.get(sensor_type)
        if config:
            if self._enable_cache:
                self._cache[cache_key] = config
            return config
        
        # Fallback to default
        if self._enable_cache:
            self._cache[cache_key] = self._default_config
        return self._default_config
    
    def _invalidate_cache(self, sensor_id: Optional[int] = None) -> None:
        """
        Invalidate cache (partial or total).
        
        Args:
            sensor_id: Sensor ID to invalidate (None = invalidate all)
        """
        if sensor_id is None:
            self._cache.clear()
        else:
            keys_to_remove = [k for k in self._cache.keys() if k[0] == sensor_id]
            for key in keys_to_remove:
                del self._cache[key]
    
    def get_registered_sensors(self) -> Dict[int, RegimeConfig]:
        """Get all sensor-specific configurations."""
        return self._sensor_configs.copy()
    
    def get_registered_types(self) -> Dict[str, RegimeConfig]:
        """Get all sensor-type configurations."""
        return self._type_configs.copy()
    
    def remove_sensor_config(self, sensor_id: int) -> None:
        """
        Remove configuration for a specific sensor.
        
        Args:
            sensor_id: Sensor identifier
        """
        self._sensor_configs.pop(sensor_id, None)
        self._invalidate_cache(sensor_id)
    
    def remove_type_config(self, sensor_type: str) -> None:
        """
        Remove configuration for a sensor type.
        
        Args:
            sensor_type: Sensor type
        """
        self._type_configs.pop(sensor_type, None)
        self._cache = {k: v for k, v in self._cache.items() if k[1] != sensor_type}
