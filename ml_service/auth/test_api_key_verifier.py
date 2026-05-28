import os
import pytest
from fastapi import HTTPException

from .api_key_verifier import verify_api_key


class TestApiKeyVerifier:
    def test_valid_key_returns_key(self):
        os.environ["ML_API_KEY"] = "valid-production-key"
        result = verify_api_key("valid-production-key")
        assert result == "valid-production-key"

    def test_invalid_key_raises_401(self):
        os.environ["ML_API_KEY"] = "valid-production-key"
        with pytest.raises(HTTPException) as exc:
            verify_api_key("wrong-key")
        assert exc.value.status_code == 401

    def test_missing_key_raises_401(self):
        os.environ["ML_API_KEY"] = "valid-production-key"
        with pytest.raises(HTTPException) as exc:
            verify_api_key(None)
        assert exc.value.status_code == 401
