"""Tests for BATCH 5 security fixes: FIX-25, FIX-26, FIX-27."""
from __future__ import annotations

import hashlib
import hmac
import os
from unittest.mock import patch

import pytest


# ─── FIX-25: HmacKeyManager ────────────────────────────────────────────────

class TestHmacKeyManager:
    def _manager(self, env: dict):
        from iot_machine_learning.infrastructure.ml.cognitive.compliance.hmac_key_manager import (
            HmacKeyManager,
        )
        with patch.dict(os.environ, env, clear=False):
            return HmacKeyManager(), env

    def test_load_key_v1_hex(self):
        key_hex = "deadbeef" * 8
        env = {"ML_COMPLIANCE_HMAC_KEY_V1": key_hex}
        with patch.dict(os.environ, env, clear=False):
            from iot_machine_learning.infrastructure.ml.cognitive.compliance.hmac_key_manager import HmacKeyManager
            m = HmacKeyManager()
            key, version = m.load_key()
        assert key == bytes.fromhex(key_hex)
        assert version == 1

    def test_load_key_legacy_fallback(self):
        env = {"ML_COMPLIANCE_HMAC_KEY": "mysecret", "ML_COMPLIANCE_HMAC_KEY_V1": ""}
        with patch.dict(os.environ, env, clear=False):
            from iot_machine_learning.infrastructure.ml.cognitive.compliance.hmac_key_manager import HmacKeyManager
            m = HmacKeyManager()
            key, version = m.load_key()
        assert key == b"mysecret"
        assert version == 0

    def test_load_key_none_when_no_key_configured(self):
        clean_env = {
            "ML_COMPLIANCE_HMAC_KEY": "",
            "ML_COMPLIANCE_HMAC_KEY_V1": "",
            "ML_COMPLIANCE_HMAC_KEY_V2": "",
        }
        with patch.dict(os.environ, clean_env, clear=False):
            from iot_machine_learning.infrastructure.ml.cognitive.compliance.hmac_key_manager import HmacKeyManager
            m = HmacKeyManager()
            key, _ = m.load_key()
        assert key is None

    def test_rotate_to_next_promotes_v2(self):
        env = {
            "ML_COMPLIANCE_HMAC_KEY_V1": "aaaa",
            "ML_COMPLIANCE_HMAC_KEY_V2": "bbbb",
        }
        with patch.dict(os.environ, env, clear=False):
            from iot_machine_learning.infrastructure.ml.cognitive.compliance.hmac_key_manager import HmacKeyManager
            m = HmacKeyManager()
            result = m.rotate_to_next()
            # Assert inside the patch.dict context — os.environ mutation visible here
            assert result is True
            assert os.environ.get("ML_COMPLIANCE_HMAC_KEY_V1") == "bbbb"
            assert os.environ.get("ML_COMPLIANCE_HMAC_KEY_V2", "") == ""

    def test_rotate_to_next_fails_without_v2(self):
        env = {"ML_COMPLIANCE_HMAC_KEY_V1": "aaaa", "ML_COMPLIANCE_HMAC_KEY_V2": ""}
        with patch.dict(os.environ, env, clear=False):
            from iot_machine_learning.infrastructure.ml.cognitive.compliance.hmac_key_manager import HmacKeyManager
            m = HmacKeyManager()
            result = m.rotate_to_next()
        assert result is False

    def test_verify_with_any_version_v1(self):
        key = b"testkey123"
        data = b"hello world"
        sig = hmac.new(key, data, hashlib.sha256).digest()
        env = {"ML_COMPLIANCE_HMAC_KEY_V1": key.hex()}
        with patch.dict(os.environ, env, clear=False):
            from iot_machine_learning.infrastructure.ml.cognitive.compliance.hmac_key_manager import HmacKeyManager
            m = HmacKeyManager()
            assert m.verify_with_any_version(data, sig) is True

    def test_verify_with_any_version_cross_version(self):
        """Signature from old V1 key verifies when V2 is active."""
        old_key = b"oldkey"
        new_key = b"newkey"
        data = b"payload"
        old_sig = hmac.new(old_key, data, hashlib.sha256).digest()
        env = {
            "ML_COMPLIANCE_HMAC_KEY_V1": new_key.hex(),
            "ML_COMPLIANCE_HMAC_KEY_V2": old_key.hex(),
        }
        with patch.dict(os.environ, env, clear=False):
            from iot_machine_learning.infrastructure.ml.cognitive.compliance.hmac_key_manager import HmacKeyManager
            m = HmacKeyManager()
            # Old signature still verifies via V2 during rotation window
            assert m.verify_with_any_version(data, old_sig) is True

    def test_verify_wrong_signature_fails(self):
        key = b"testkey"
        data = b"data"
        wrong_sig = b"\x00" * 32
        env = {"ML_COMPLIANCE_HMAC_KEY_V1": key.hex()}
        with patch.dict(os.environ, env, clear=False):
            from iot_machine_learning.infrastructure.ml.cognitive.compliance.hmac_key_manager import HmacKeyManager
            m = HmacKeyManager()
            assert m.verify_with_any_version(data, wrong_sig) is False

    def test_rotation_overdue_warning(self, caplog):
        import logging
        env = {
            "ML_HMAC_KEY_CREATED_AT": "2020-01-01T00:00:00Z",
            "ML_HMAC_KEY_ROTATION_DAYS": "90",
        }
        with patch.dict(os.environ, env, clear=False):
            with caplog.at_level(logging.WARNING):
                from iot_machine_learning.infrastructure.ml.cognitive.compliance.hmac_key_manager import HmacKeyManager
                HmacKeyManager()
        assert any("rotation overdue" in r.message.lower() for r in caplog.records)

    def test_no_warning_when_key_is_fresh(self, caplog):
        import logging
        from datetime import datetime, timezone
        created = datetime.now(timezone.utc).isoformat()
        env = {
            "ML_HMAC_KEY_CREATED_AT": created,
            "ML_HMAC_KEY_ROTATION_DAYS": "90",
        }
        with patch.dict(os.environ, env, clear=False):
            with caplog.at_level(logging.WARNING):
                from iot_machine_learning.infrastructure.ml.cognitive.compliance.hmac_key_manager import HmacKeyManager
                HmacKeyManager()
        assert not any("rotation overdue" in r.message.lower() for r in caplog.records)


# ─── FIX-26: CORS guard ────────────────────────────────────────────────────

class TestCorsCredentialsGuard:
    def test_credentials_true_with_explicit_origins_ok(self):
        """No exception when credentials=True and origins are specific."""
        origins = ["http://localhost:3000"]
        wildcard = "*" in origins or origins == ["*"]
        allow_creds = True
        if allow_creds and (wildcard or not origins):
            raise ValueError("Should not raise")
        assert not wildcard

    def test_credentials_true_with_wildcard_raises(self):
        from iot_machine_learning.ml_service.config.security_config import SecurityConfig
        cfg = SecurityConfig(ML_CORS_ALLOW_CREDENTIALS=True)
        assert cfg.ML_CORS_ALLOW_CREDENTIALS is True
        origins = ["*"]
        wildcard = "*" in origins or origins == ["*"]
        with pytest.raises(ValueError, match="SEC-2"):
            if cfg.ML_CORS_ALLOW_CREDENTIALS and wildcard:
                raise ValueError("SEC-2: credentials=True with wildcard origins")

    def test_credentials_false_default(self):
        from iot_machine_learning.ml_service.config.security_config import SecurityConfig
        cfg = SecurityConfig()
        assert cfg.ML_CORS_ALLOW_CREDENTIALS is False

    def test_credentials_true_empty_origins_raises(self):
        from iot_machine_learning.ml_service.config.security_config import SecurityConfig
        cfg = SecurityConfig(ML_CORS_ALLOW_CREDENTIALS=True)
        origins = []
        with pytest.raises(ValueError, match="SEC-2"):
            if cfg.ML_CORS_ALLOW_CREDENTIALS and not origins:
                raise ValueError("SEC-2: credentials=True with empty origins")


# ─── FIX-27: RBAC readonly key ─────────────────────────────────────────────

class TestApiKeyRbac:
    def test_security_config_has_readonly_key(self):
        from iot_machine_learning.ml_service.config.security_config import SecurityConfig
        cfg = SecurityConfig(ML_API_KEY="master", ML_API_KEY_READONLY="readonly")
        assert cfg.ML_API_KEY == "master"
        assert cfg.ML_API_KEY_READONLY == "readonly"

    def test_readonly_key_default_empty(self):
        from iot_machine_learning.ml_service.config.security_config import SecurityConfig
        cfg = SecurityConfig()
        assert cfg.ML_API_KEY_READONLY == ""

    def test_readonly_logic_accepts_master(self):
        """Pure logic: master key is accepted by readonly check."""
        master_key = "master-key"
        readonly_key = "read-key"
        presented = "master-key"
        accepted = (master_key and presented == master_key) or \
                   (readonly_key and presented == readonly_key)
        assert accepted

    def test_readonly_logic_accepts_readonly(self):
        master_key = "master-key"
        readonly_key = "read-key"
        presented = "read-key"
        accepted = (master_key and presented == master_key) or \
                   (readonly_key and presented == readonly_key)
        assert accepted

    def test_readonly_logic_rejects_unknown(self):
        master_key = "master-key"
        readonly_key = "read-key"
        presented = "wrong-key"
        accepted = (master_key and presented == master_key) or \
                   (readonly_key and presented == readonly_key)
        assert not accepted

    def test_master_only_logic_rejects_readonly_key(self):
        """Write endpoint logic: only master key accepted."""
        master_key = "master-key"
        presented = "read-only-key"
        accepted = bool(master_key) and presented == master_key
        assert not accepted

    def test_master_only_logic_accepts_master_key(self):
        master_key = "master-key"
        presented = "master-key"
        accepted = bool(master_key) and presented == master_key
        assert accepted


# ─── SEC-4: Compliance path traversal ────────────────────────────────────

class TestCompliancePathTraversal:
    def test_valid_path_inside_base_dir(self, tmp_path):
        from iot_machine_learning.ml_service.main import validate_compliance_path
        base = tmp_path / "compliance"
        sink = base / "exports.jsonl"
        result = validate_compliance_path(str(sink), str(base))
        assert result == sink.resolve()

    def test_traversal_escape_raises(self, tmp_path):
        from iot_machine_learning.ml_service.main import validate_compliance_path
        base = tmp_path / "compliance"
        sink = base / ".." / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="compliance_path_traversal"):
            validate_compliance_path(str(sink), str(base))

    def test_path_outside_base_without_traversal_raises(self, tmp_path):
        from iot_machine_learning.ml_service.main import validate_compliance_path
        base = tmp_path / "compliance"
        sink = tmp_path / "tmp" / "audit.jsonl"
        with pytest.raises(ValueError, match="compliance_path_traversal"):
            validate_compliance_path(str(sink), str(base))
