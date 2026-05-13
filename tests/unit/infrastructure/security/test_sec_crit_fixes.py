"""Tests for SEC-CRIT-1, SEC-CRIT-2, SEC-CRIT-3, SEC-CRIT-4 fixes.

Validates security critical issue resolutions from audit report.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pydantic import BaseModel, Field, ValidationError

from iot_machine_learning.domain.value_objects.series_id import (
    SeriesId,
    TenantId,
    DocumentId,
)
from iot_machine_learning.infrastructure.security.auth_provider import (
    ApiKeyAuthProvider,
    AuthResult,
)
from iot_machine_learning.infrastructure.security.audit_logger import FileAuditLogger
from iot_machine_learning.infrastructure.security.secret_redactor import SecretRedactor
from iot_machine_learning.ml_service.config.security_config import SecurityConfig


class TestSecCrit1AuthFailClosed:
    """SEC-CRIT-1: Authentication must fail-closed."""
    
    def test_security_config_validate_fails_without_key_in_production(self):
        """Validate fails when auth enabled but no key in production."""
        config = SecurityConfig(
            ML_API_KEY="",
            ML_AUTH_ENABLED=True,
            ML_DEV_MODE=False,
        )
        
        with pytest.raises(ValueError, match="ML_API_KEY must be set"):
            config.validate()
    
    def test_security_config_validate_passes_with_key(self):
        """Validate passes when key is set."""
        config = SecurityConfig(
            ML_API_KEY="a" * 32,  # Min 32 chars
            ML_AUTH_ENABLED=True,
            ML_DEV_MODE=False,
        )
        
        config.validate()  # Should not raise
    
    def test_security_config_validate_passes_in_dev_mode(self):
        """Validate passes in dev mode even without key."""
        config = SecurityConfig(
            ML_API_KEY="",
            ML_AUTH_ENABLED=True,
            ML_DEV_MODE=True,
        )
        
        config.validate()  # Should not raise
    
    def test_security_config_rejects_short_api_key(self):
        """API key must be at least 32 characters."""
        with pytest.raises(ValueError, match="at least 32 characters"):
            SecurityConfig(ML_API_KEY="short")
    
    def test_auth_provider_abstraction_dip(self):
        """AuthProvider follows DIP principle."""
        # Middleware depends on abstraction, not concrete SecurityConfig
        provider = ApiKeyAuthProvider(
            primary_key="a" * 32,
            readonly_key="b" * 32,
            enabled=True,
        )
        
        # Primary key
        result = provider.authenticate("a" * 32)
        assert result.authenticated
        assert not result.is_readonly
        
        # Readonly key
        result = provider.authenticate("b" * 32)
        assert result.authenticated
        assert result.is_readonly
        
        # Invalid key
        result = provider.authenticate("invalid")
        assert not result.authenticated
        assert result.reason == "invalid_credentials"
    
    def test_auth_provider_disabled_mode(self):
        """Auth provider allows all when disabled."""
        provider = ApiKeyAuthProvider(
            primary_key="a" * 32,
            enabled=False,
        )
        
        result = provider.authenticate("")
        assert result.authenticated
        assert result.reason == "auth_disabled"


class TestSecCrit2AuditHashFull:
    """SEC-CRIT-2: Audit log hash must be full SHA-256."""
    
    def test_audit_logger_uses_full_sha256_hash(self):
        """Integrity hash must be 64 characters (256 bits)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.jsonl"
            logger = FileAuditLogger(log_file, include_hash=True)
            
            logger.log_event(
                event_type="test",
                action="test_action",
                resource="test_resource",
                details={"key": "value"},
            )
            
            # Read log entry
            with open(log_file) as f:
                entry = json.loads(f.readline())
            
            # Verify hash length
            assert "integrity_hash" in entry
            assert len(entry["integrity_hash"]) == 64  # Full SHA-256
            
            # Verify hash is hex
            int(entry["integrity_hash"], 16)  # Should not raise
    
    def test_audit_logger_hash_prevents_tampering(self):
        """Hash verification detects tampering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.jsonl"
            logger = FileAuditLogger(log_file, include_hash=True)
            
            logger.log_event(
                event_type="test",
                action="test_action",
                resource="test_resource",
                details={"key": "value"},
            )
            
            # Read and verify original
            with open(log_file) as f:
                entry = json.loads(f.readline())
            
            original_hash = entry.pop("integrity_hash")
            
            # Recompute hash
            import hashlib
            content_str = json.dumps(entry, sort_keys=True, default=str)
            computed_hash = hashlib.sha256(content_str.encode("utf-8")).hexdigest()
            
            assert computed_hash == original_hash
            
            # Tamper with entry
            entry["details"]["key"] = "tampered"
            tampered_str = json.dumps(entry, sort_keys=True, default=str)
            tampered_hash = hashlib.sha256(tampered_str.encode("utf-8")).hexdigest()
            
            assert tampered_hash != original_hash  # Tampering detected


class TestSecCrit3SeriesIdValidation:
    """SEC-CRIT-3: Series ID must be validated to prevent injection."""
    
    def test_series_id_validates_format(self):
        """SeriesId validates format to prevent injection."""
        # Create test model using SeriesId
        class TestModel(BaseModel):
            series_id: SeriesId
        
        # Valid ID
        model = TestModel(series_id="valid-series_123")
        assert model.series_id == "valid-series_123"
        
        # Invalid ID with slash (path traversal)
        with pytest.raises(ValidationError, match="Invalid series_id format"):
            TestModel(series_id="invalid/path")
    
    def test_tenant_id_validates_format(self):
        """TenantId validates format to prevent SQL injection."""
        class TestModel(BaseModel):
            tenant_id: TenantId
        
        # Valid tenant ID
        model = TestModel(tenant_id="tenant-123")
        assert model.tenant_id == "tenant-123"
        
        # Invalid tenant ID (SQL injection attempt)
        with pytest.raises(ValidationError, match="String should match pattern"):
            TestModel(tenant_id="tenant'; DROP TABLE--")
    
    def test_document_id_validates_format(self):
        """DocumentId validates format to prevent path traversal."""
        class TestModel(BaseModel):
            document_id: DocumentId
        
        # Valid
        model = TestModel(document_id="doc_abc")
        assert model.document_id == "doc_abc"
        
        # Invalid (path traversal attempt)
        with pytest.raises(ValidationError, match="String should match pattern"):
            TestModel(document_id="../../../etc/passwd")
    
    def test_series_id_length_limits(self):
        """Series ID must respect length limits."""
        class TestModel(BaseModel):
            series_id: SeriesId
        
        # Too long
        with pytest.raises(ValidationError):
            TestModel(series_id="a" * 101)  # Max 100
        
        # Empty (too short)
        with pytest.raises(ValidationError):
            TestModel(series_id="")
    
    def test_series_id_whitespace_stripped(self):
        """Series ID whitespace is stripped automatically."""
        class TestModel(BaseModel):
            series_id: SeriesId
        
        model = TestModel(series_id="  doc-123  ")
        assert model.series_id == "doc-123"
    
    def test_series_id_rejects_special_characters(self):
        """Series ID rejects special characters that could be dangerous."""
        class TestModel(BaseModel):
            series_id: SeriesId
        
        dangerous_chars = [
            "series;DROP",  # Semicolon
            "series'OR'1",  # Single quote
            "series\"test",  # Double quote
            "series<script>",  # HTML tags
            "series&param",  # Ampersand
            "series|cmd",  # Pipe
            "series`cmd`",  # Backtick
        ]
        
        for dangerous in dangerous_chars:
            with pytest.raises(ValidationError):
                TestModel(series_id=dangerous)


class TestSecCrit4SecretRedaction:
    """SEC-CRIT-4: Secrets must be redacted from logs."""
    
    def test_secret_redactor_redacts_api_keys(self):
        """SecretRedactor redacts API_KEY fields."""
        data = {
            "ML_API_KEY": "secret123",
            "ML_API_KEY_READONLY": "readonly456",
            "safe_field": "visible",
        }
        
        redacted = SecretRedactor.redact(data)
        
        assert redacted["ML_API_KEY"] == SecretRedactor.REDACTED
        assert redacted["ML_API_KEY_READONLY"] == SecretRedactor.REDACTED
        assert redacted["safe_field"] == "visible"
    
    def test_secret_redactor_redacts_passwords(self):
        """SecretRedactor redacts PASSWORD fields."""
        data = {
            "DB_PASSWORD": "pass123",
            "user_password": "pass456",
            "username": "john",
        }
        
        redacted = SecretRedactor.redact(data)
        
        assert redacted["DB_PASSWORD"] == SecretRedactor.REDACTED
        assert redacted["user_password"] == SecretRedactor.REDACTED
        assert redacted["username"] == "john"
    
    def test_secret_redactor_handles_nested_dicts(self):
        """SecretRedactor recursively redacts nested dictionaries."""
        data = {
            "config": {
                "API_KEY": "secret",
                "timeout": 30,
            },
            "database": {
                "password": "pass",
                "username": "user",
            },
        }
        
        redacted = SecretRedactor.redact(data)
        
        # Verify nested redaction
        assert isinstance(redacted["config"], dict)
        assert redacted["config"]["API_KEY"] == SecretRedactor.REDACTED
        assert redacted["config"]["timeout"] == 30
        
        assert isinstance(redacted["database"], dict)
        assert redacted["database"]["password"] == SecretRedactor.REDACTED
        assert redacted["database"]["username"] == "user"
    
    def test_secret_redactor_handles_lists(self):
        """SecretRedactor redacts dicts inside lists."""
        data = {
            "servers": [
                {"host": "server1", "API_KEY": "key1"},
                {"host": "server2", "PASSWORD": "pass2"},
            ],
        }
        
        redacted = SecretRedactor.redact(data)
        
        assert redacted["servers"][0]["host"] == "server1"
        assert redacted["servers"][0]["API_KEY"] == SecretRedactor.REDACTED
        assert redacted["servers"][1]["PASSWORD"] == SecretRedactor.REDACTED
    
    def test_secret_redactor_case_insensitive(self):
        """SecretRedactor is case-insensitive."""
        data = {
            "api_key": "secret1",
            "Api_Key": "secret2",
            "API_KEY": "secret3",
        }
        
        redacted = SecretRedactor.redact(data)
        
        assert redacted["api_key"] == SecretRedactor.REDACTED
        assert redacted["Api_Key"] == SecretRedactor.REDACTED
        assert redacted["API_KEY"] == SecretRedactor.REDACTED
    
    def test_secret_redactor_all_secret_patterns(self):
        """SecretRedactor handles all defined secret patterns."""
        data = {
            "API_KEY": "key",
            "PASSWORD": "pass",
            "SECRET": "secret",
            "TOKEN": "token",
            "CONNECTION_STRING": "conn",
            "PRIVATE_KEY": "priv",
            "AUTH_HEADER": "auth",
            "CREDENTIAL": "cred",
            "safe": "visible",
        }
        
        redacted = SecretRedactor.redact(data)
        
        # All secrets redacted
        assert redacted["API_KEY"] == SecretRedactor.REDACTED
        assert redacted["PASSWORD"] == SecretRedactor.REDACTED
        assert redacted["SECRET"] == SecretRedactor.REDACTED
        assert redacted["TOKEN"] == SecretRedactor.REDACTED
        assert redacted["CONNECTION_STRING"] == SecretRedactor.REDACTED
        assert redacted["PRIVATE_KEY"] == SecretRedactor.REDACTED
        assert redacted["AUTH_HEADER"] == SecretRedactor.REDACTED
        assert redacted["CREDENTIAL"] == SecretRedactor.REDACTED
        
        # Safe field visible
        assert redacted["safe"] == "visible"
    
    def test_secret_redactor_does_not_modify_original(self):
        """SecretRedactor returns new dict, does not modify original."""
        original = {
            "API_KEY": "secret",
            "safe": "visible",
        }
        
        redacted = SecretRedactor.redact(original)
        
        # Original unchanged
        assert original["API_KEY"] == "secret"
        assert original["safe"] == "visible"
        
        # Redacted is different
        assert redacted["API_KEY"] == SecretRedactor.REDACTED
        assert redacted["safe"] == "visible"
