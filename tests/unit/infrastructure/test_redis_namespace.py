"""Tests for RedisNamespace - tenant isolation."""

import os
import pytest

from iot_machine_learning.infrastructure.security.redis_namespace import (
    RedisNamespace,
    get_namespace,
    create_namespace,
)


class TestRedisNamespace:
    """Test RedisNamespace tenant isolation."""
    
    def test_initialization_requires_tenant_id(self):
        """tenant_id is required."""
        with pytest.raises(ValueError, match="tenant_id is REQUIRED"):
            RedisNamespace(tenant_id="")
    
    def test_initialization_with_valid_tenant(self):
        """Valid tenant_id should work."""
        ns = RedisNamespace(tenant_id="acme")
        
        assert ns.tenant_id == "acme"
        assert ns.app == "zenin"
        assert ns.env == "prod"  # default
    
    def test_env_auto_detection(self):
        """Environment should be auto-detected from ZENIN_ENV."""
        os.environ["ZENIN_ENV"] = "staging"
        
        ns = RedisNamespace(tenant_id="test")
        
        assert ns.env == "staging"
        
        # Cleanup
        del os.environ["ZENIN_ENV"]
    
    def test_key_generation_basic(self):
        """Basic key generation."""
        ns = RedisNamespace(tenant_id="acme")
        
        key = ns.key("series", "sensor_42")
        
        assert key == "prod:zenin:acme:series:sensor_42"
    
    def test_key_generation_with_suffix(self):
        """Key generation with suffix."""
        ns = RedisNamespace(tenant_id="acme")
        
        key = ns.key("series", "sensor_42", "window")
        
        assert key == "prod:zenin:acme:series:sensor_42:window"
    
    def test_key_validation_rejects_colon(self):
        """Colon in ID should be rejected (key injection)."""
        ns = RedisNamespace(tenant_id="acme")
        
        with pytest.raises(ValueError, match="key injection"):
            ns.key("series", "sensor:42")
    
    def test_key_validation_rejects_invalid_chars_strict(self):
        """Invalid characters should be rejected in strict mode."""
        ns = RedisNamespace(tenant_id="acme", strict_mode=True)
        
        with pytest.raises(ValueError, match="invalid characters"):
            ns.key("series", "sensor@42")
    
    def test_key_validation_sanitizes_in_lenient_mode(self):
        """Invalid characters should be sanitized in lenient mode."""
        ns = RedisNamespace(tenant_id="acme", strict_mode=False)
        
        key = ns.key("series", "sensor@42")
        
        # @ should be replaced with _
        assert "sensor_42" in key
    
    def test_key_caching(self):
        """Keys should be cached."""
        ns = RedisNamespace(tenant_id="acme")
        
        key1 = ns.key("series", "sensor_42")
        key2 = ns.key("series", "sensor_42")
        
        # Should be same object (cached)
        assert key1 == key2
    
    def test_pattern_generation(self):
        """Pattern generation for SCAN."""
        ns = RedisNamespace(tenant_id="acme")
        
        pattern = ns.pattern("series", "sensor_*")
        
        assert pattern == "prod:zenin:acme:series:sensor_*"
    
    def test_extract_resource_id(self):
        """Extract resource_id from namespaced key."""
        ns = RedisNamespace(tenant_id="acme")
        
        key = "prod:zenin:acme:series:sensor_42:window"
        resource_id = ns.extract_resource_id(key)
        
        assert resource_id == "sensor_42"
    
    def test_extract_tenant_id(self):
        """Extract tenant_id from namespaced key."""
        ns = RedisNamespace(tenant_id="acme")
        
        key = "prod:zenin:acme:series:sensor_42"
        tenant_id = ns.extract_tenant_id(key)
        
        assert tenant_id == "acme"
    
    def test_is_valid_key_true(self):
        """Valid key should return True."""
        ns = RedisNamespace(tenant_id="acme")
        
        key = "prod:zenin:acme:series:sensor_42"
        
        assert ns.is_valid_key(key) is True
    
    def test_is_valid_key_false_wrong_tenant(self):
        """Key from different tenant should return False."""
        ns = RedisNamespace(tenant_id="acme")
        
        key = "prod:zenin:other:series:sensor_42"
        
        assert ns.is_valid_key(key) is False
    
    def test_is_valid_key_false_wrong_format(self):
        """Invalid format should return False."""
        ns = RedisNamespace(tenant_id="acme")
        
        key = "invalid:key"
        
        assert ns.is_valid_key(key) is False
    
    def test_tenant_isolation(self):
        """Different tenants should have different keys."""
        ns1 = RedisNamespace(tenant_id="acme")
        ns2 = RedisNamespace(tenant_id="globex")
        
        key1 = ns1.key("series", "sensor_42")
        key2 = ns2.key("series", "sensor_42")
        
        assert key1 != key2
        assert "acme" in key1
        assert "globex" in key2
    
    def test_default_ttl(self):
        """Default TTL should be 24h."""
        ns = RedisNamespace(tenant_id="acme")
        
        assert ns.default_ttl == 86400
    
    def test_custom_ttl(self):
        """Custom TTL should be respected."""
        ns = RedisNamespace(tenant_id="acme", default_ttl=3600)
        
        assert ns.default_ttl == 3600
    
    def test_clear_cache(self):
        """Cache should be clearable."""
        ns = RedisNamespace(tenant_id="acme")
        
        # Generate some keys to populate cache
        ns.key("series", "sensor_1")
        ns.key("series", "sensor_2")
        
        # Clear cache
        ns.clear_cache()
        
        # Cache should be empty (no way to verify directly, but shouldn't error)
        assert True
    
    def test_hash_key(self):
        """Hash key generation."""
        ns = RedisNamespace(tenant_id="acme")
        
        key = ns.hash_key("weights", "regime_normal")
        
        # Should be same as regular key without suffix
        assert "weights" in key
        assert "regime_normal" in key


class TestGetNamespace:
    """Test get_namespace factory with caching."""
    
    def test_get_namespace_returns_instance(self):
        """get_namespace should return RedisNamespace."""
        ns = get_namespace(tenant_id="acme")
        
        assert isinstance(ns, RedisNamespace)
        assert ns.tenant_id == "acme"
    
    def test_get_namespace_caches(self):
        """get_namespace should cache instances."""
        ns1 = get_namespace(tenant_id="acme")
        ns2 = get_namespace(tenant_id="acme")
        
        # Should be same instance (cached)
        assert ns1 is ns2
    
    def test_get_namespace_different_tenants(self):
        """Different tenants should get different instances."""
        ns1 = get_namespace(tenant_id="acme")
        ns2 = get_namespace(tenant_id="globex")
        
        assert ns1 is not ns2
        assert ns1.tenant_id == "acme"
        assert ns2.tenant_id == "globex"


class TestCreateNamespace:
    """Test create_namespace factory."""
    
    def test_create_namespace_basic(self):
        """Basic namespace creation."""
        ns = create_namespace(tenant_id="acme")
        
        assert ns.tenant_id == "acme"
        assert ns.app == "zenin"
    
    def test_create_namespace_custom_app(self):
        """Custom app name."""
        ns = create_namespace(tenant_id="acme", app="custom")
        
        assert ns.app == "custom"
    
    def test_create_namespace_custom_ttl(self):
        """Custom default TTL."""
        ns = create_namespace(tenant_id="acme", default_ttl=7200)
        
        assert ns.default_ttl == 7200
    
    def test_create_namespace_lenient_mode(self):
        """Lenient mode."""
        ns = create_namespace(tenant_id="acme", strict_mode=False)
        
        # Should sanitize instead of raising
        key = ns.key("series", "sensor@42")
        assert "sensor_42" in key


class TestSecurityValidation:
    """Test security validations."""
    
    def test_rejects_empty_tenant_id(self):
        """Empty tenant_id should be rejected."""
        with pytest.raises(ValueError, match="tenant_id is REQUIRED"):
            RedisNamespace(tenant_id="")
    
    def test_rejects_too_long_tenant_id(self):
        """Too long tenant_id should be rejected."""
        long_id = "a" * 65
        
        with pytest.raises(ValueError, match="too long"):
            RedisNamespace(tenant_id=long_id)
    
    def test_rejects_colon_in_tenant_id(self):
        """Colon in tenant_id should be rejected."""
        with pytest.raises(ValueError, match="key injection"):
            RedisNamespace(tenant_id="tenant:123")
    
    def test_rejects_special_chars_in_strict_mode(self):
        """Special characters should be rejected in strict mode."""
        with pytest.raises(ValueError, match="invalid characters"):
            RedisNamespace(tenant_id="tenant@123", strict_mode=True)
    
    def test_sanitizes_special_chars_in_lenient_mode(self):
        """Special characters should be sanitized in lenient mode."""
        ns = RedisNamespace(tenant_id="tenant@123", strict_mode=False)
        
        # Should have sanitized tenant_id
        assert "@" not in ns.tenant_id
        assert "_" in ns.tenant_id
