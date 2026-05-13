"""Seguridad — audit logging, RBAC, encryption (ISO 27001).

SEC-CRIT-1: AuthProvider abstraction for DIP.
SEC-CRIT-4: SecretRedactor for preventing credential exposure in logs.
"""

from .auth_provider import AuthProvider, AuthResult, ApiKeyAuthProvider
from .secret_redactor import SecretRedactor

__all__ = [
    "AuthProvider",
    "AuthResult",
    "ApiKeyAuthProvider",
    "SecretRedactor",
]
