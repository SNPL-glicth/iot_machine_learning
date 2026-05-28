"""API key verifier for ML service.

Fails closed: refuses to start if ML_API_KEY is not configured.
Uses constant-time comparison to prevent timing attacks.
"""

import os
import hmac

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_expected_key: str | None = os.environ.get("ML_API_KEY")
if not _expected_key or len(_expected_key.strip()) == 0:
    raise RuntimeError("ML_API_KEY is not configured. Service will not start.")


def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """Verify the provided API key against the configured secret."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    if not hmac.compare_digest(api_key, _expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    return api_key
