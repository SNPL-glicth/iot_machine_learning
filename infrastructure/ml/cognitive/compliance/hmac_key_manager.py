"""HmacKeyManager — SEC-1 / FIX-25.

Manages HMAC key versioning, rotation, and cross-version verification
for compliance export signatures.

Env vars (all optional — falls back to legacy ML_COMPLIANCE_HMAC_KEY):
    ML_COMPLIANCE_HMAC_KEY          Legacy single-key (backward compat)
    ML_COMPLIANCE_HMAC_KEY_V1       Current active key (hex or UTF-8)
    ML_COMPLIANCE_HMAC_KEY_V2       Next key pre-loaded for rotation
    ML_COMPLIANCE_HMAC_KEY_VERSION  Active version number (default: 1)
    ML_HMAC_KEY_ROTATION_DAYS       Rotation schedule in days (default: 90)
    ML_HMAC_KEY_CREATED_AT          ISO-8601 date when key was created
                                    Used to warn when rotation is overdue.

Usage:
    manager = HmacKeyManager()
    key, version = manager.load_key()            # current key + version
    ok = manager.verify_with_any_version(data, sig)  # cross-version verify
    rotated = manager.rotate_to_next()           # promote V2 → V1 in-process
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_ROTATION_WARN_DAYS_DEFAULT = 90


def _parse_key(raw: Optional[str]) -> Optional[bytes]:
    """Decode a key from hex or UTF-8. Returns None if raw is empty."""
    if not raw:
        return None
    try:
        key = bytes.fromhex(raw)
        return key
    except ValueError:
        return raw.encode("utf-8")


class HmacKeyManager:
    """Manages HMAC key versions for compliance export signatures.

    Supports:
    - Single legacy key (ML_COMPLIANCE_HMAC_KEY) — backward compatible
    - Versioned keys V1/V2 for zero-downtime rotation
    - Cross-version signature verification during rotation window
    - Startup warning when key is overdue for rotation
    """

    def __init__(self) -> None:
        self._check_rotation_overdue()

    # ── Public API ─────────────────────────────────────────────────

    def load_key(self, version: Optional[int] = None) -> Tuple[Optional[bytes], int]:
        """Return (key_bytes, version_int) for the requested version.

        Args:
            version: Explicit version to load (1 or 2).
                     If None, uses the active version from env.

        Returns:
            Tuple of (key_bytes, version). key_bytes is None when not
            configured (compliance export will be unsigned).
        """
        if version is None:
            version = self.get_current_version()

        key = self._load_versioned_key(version)
        if key is not None:
            logger.info(
                "hmac_key_loaded",
                extra={"version": version, "length": len(key)},
            )
            return key, version

        # Fallback to legacy single key
        legacy = _parse_key(os.environ.get("ML_COMPLIANCE_HMAC_KEY"))
        if legacy is not None:
            logger.info(
                "hmac_key_loaded_legacy",
                extra={"version": 0, "length": len(legacy)},
            )
            return legacy, 0

        return None, version

    def get_current_version(self) -> int:
        """Return the active key version from ML_COMPLIANCE_HMAC_KEY_VERSION."""
        raw = os.environ.get("ML_COMPLIANCE_HMAC_KEY_VERSION", "1")
        try:
            return int(raw)
        except ValueError:
            logger.warning(
                "hmac_key_version_invalid",
                extra={"raw": raw, "fallback": 1},
            )
            return 1

    def rotate_to_next(self) -> bool:
        """Promote V2 key to V1 in the current process environment.

        Returns True if rotation succeeded (V2 existed and was promoted).
        Returns False if no V2 key is configured.

        Note: This mutates os.environ for the current process only.
              For persistent rotation, update the secrets manager / env
              and restart the process.
        """
        v2 = os.environ.get("ML_COMPLIANCE_HMAC_KEY_V2")
        if not v2:
            logger.warning("hmac_key_rotate_failed: ML_COMPLIANCE_HMAC_KEY_V2 not set")
            return False

        os.environ["ML_COMPLIANCE_HMAC_KEY_V1"] = v2
        os.environ.pop("ML_COMPLIANCE_HMAC_KEY_V2", None)
        os.environ["ML_COMPLIANCE_HMAC_KEY_VERSION"] = "1"
        logger.info("hmac_key_rotated: V2 promoted to V1")
        return True

    def verify_with_any_version(self, data: bytes, signature: bytes) -> bool:
        """Verify a signature against any configured key version.

        Tries V1, V2, and legacy key. Returns True if any matches.
        Useful during rotation window when old signatures are still valid.
        """
        candidates = []

        for version in (1, 2):
            key = self._load_versioned_key(version)
            if key:
                candidates.append(key)

        legacy = _parse_key(os.environ.get("ML_COMPLIANCE_HMAC_KEY"))
        if legacy:
            candidates.append(legacy)

        for key in candidates:
            expected = hmac.new(key, data, hashlib.sha256).digest()
            if hmac.compare_digest(expected, signature):
                return True
        return False

    # ── Private helpers ────────────────────────────────────────────

    def _load_versioned_key(self, version: int) -> Optional[bytes]:
        env_name = f"ML_COMPLIANCE_HMAC_KEY_V{version}"
        return _parse_key(os.environ.get(env_name))

    def _check_rotation_overdue(self) -> None:
        """Warn at startup if key has exceeded the rotation schedule."""
        created_at_raw = os.environ.get("ML_HMAC_KEY_CREATED_AT")
        if not created_at_raw:
            return

        try:
            created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.debug("hmac_key_created_at_parse_failed: %s", created_at_raw)
            return

        rotation_days_raw = os.environ.get(
            "ML_HMAC_KEY_ROTATION_DAYS", str(_ROTATION_WARN_DAYS_DEFAULT)
        )
        try:
            rotation_days = int(rotation_days_raw)
        except ValueError:
            rotation_days = _ROTATION_WARN_DAYS_DEFAULT

        now = datetime.now(timezone.utc)
        age_days = (now - created_at).days

        if age_days >= rotation_days:
            logger.warning(
                "HMAC key rotation overdue: key is %d days old "
                "(rotation schedule: every %d days). "
                "Set ML_COMPLIANCE_HMAC_KEY_V2 and call rotate_to_next().",
                age_days,
                rotation_days,
                extra={
                    "age_days": age_days,
                    "rotation_days": rotation_days,
                    "created_at": created_at_raw,
                },
            )


# Module-level singleton for convenience
_manager: Optional[HmacKeyManager] = None


def get_hmac_key_manager() -> HmacKeyManager:
    """Return the module-level HmacKeyManager singleton."""
    global _manager
    if _manager is None:
        _manager = HmacKeyManager()
    return _manager
