"""Authentication provider abstraction (DIP for SEC-CRIT-1).

Defines the contract for authentication without coupling to specific implementations.
Middleware depends on this abstraction, not on SecurityConfig directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AuthResult:
    """Result of authentication attempt.
    
    Attributes:
        authenticated: Whether authentication succeeded.
        user_id: Authenticated user ID (if successful).
        reason: Reason for failure (if unsuccessful).
        is_readonly: Whether this is a read-only authentication.
    
    Applies SRP: AuthResult only represents auth outcome, no logic.
    """
    authenticated: bool
    user_id: Optional[str] = None
    reason: Optional[str] = None
    is_readonly: bool = False


class AuthProvider(ABC):
    """Abstract authentication provider (DIP principle).
    
    Middleware depends on this abstraction, enabling:
    - API key auth (current)
    - JWT auth (future)
    - OAuth2 auth (future)
    - Multi-factor auth (future)
    
    Without modifying middleware code (OCP principle).
    """
    
    @abstractmethod
    def authenticate(self, credentials: str) -> AuthResult:
        """Authenticate using provided credentials.
        
        Args:
            credentials: Authentication credentials (API key, token, etc.).
        
        Returns:
            AuthResult with authentication outcome.
        
        Applies LSP: All implementations must return valid AuthResult.
        """
        pass
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if authentication is enabled.
        
        Returns:
            True if authentication is required, False otherwise.
        
        Applies ISP: Simple boolean check, no complex state.
        """
        pass


class ApiKeyAuthProvider(AuthProvider):
    """API key-based authentication provider.
    
    Implements AuthProvider for simple API key validation.
    
    Attributes:
        _primary_key: Primary API key (full access).
        _readonly_key: Read-only API key (GET endpoints only).
        _enabled: Whether authentication is enabled.
    
    Applies SRP: Only validates API keys, no other concerns.
    """
    
    def __init__(
        self,
        primary_key: str,
        readonly_key: str = "",
        enabled: bool = True,
    ) -> None:
        """Initialize API key auth provider.
        
        Args:
            primary_key: Primary API key for full access.
            readonly_key: Optional read-only API key.
            enabled: Whether authentication is enabled.
        """
        self._primary_key = primary_key
        self._readonly_key = readonly_key
        self._enabled = enabled
    
    def authenticate(self, credentials: str) -> AuthResult:
        """Authenticate using API key.
        
        Args:
            credentials: API key from request header.
        
        Returns:
            AuthResult with authentication outcome.
        
        Applies LSP: Returns valid AuthResult as per contract.
        """
        if not self._enabled:
            return AuthResult(
                authenticated=True,
                user_id="system",
                reason="auth_disabled",
            )
        
        if not credentials:
            return AuthResult(
                authenticated=False,
                reason="missing_credentials",
            )
        
        # Check primary key
        if credentials == self._primary_key:
            return AuthResult(
                authenticated=True,
                user_id="api_user",
                is_readonly=False,
            )
        
        # Check readonly key
        if self._readonly_key and credentials == self._readonly_key:
            return AuthResult(
                authenticated=True,
                user_id="api_user_readonly",
                is_readonly=True,
            )
        
        return AuthResult(
            authenticated=False,
            reason="invalid_credentials",
        )
    
    def is_enabled(self) -> bool:
        """Check if authentication is enabled.
        
        Returns:
            True if authentication is required.
        
        Applies ISP: Simple boolean check.
        """
        return self._enabled
