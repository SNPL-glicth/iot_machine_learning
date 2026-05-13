"""Secret redactor for logging (SEC-CRIT-4).

Prevents credential exposure in logs by redacting sensitive keys.

Applies SRP: SecretRedactor only redacts, does not log.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Set


class SecretRedactor:
    """Redacts sensitive information from dictionaries before logging.
    
    Prevents credential exposure in logs (SEC-CRIT-4).
    
    Attributes:
        REDACTED: Placeholder for redacted values.
        SECRET_KEYS: Set of key patterns that indicate sensitive data.
    
    Applies SRP: Only redacts secrets, no logging or other concerns.
    """
    
    REDACTED: str = "*****REDACTED*****"
    
    # Immutable set of secret key patterns (no magic strings)
    SECRET_KEYS: FrozenSet[str] = frozenset({
        "API_KEY",
        "PASSWORD",
        "SECRET",
        "TOKEN",
        "CONNECTION_STRING",
        "PRIVATE_KEY",
        "AUTH",
        "CREDENTIAL",
        "PASSPHRASE",
        "CERT",
    })
    
    @staticmethod
    def redact(data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive keys from dictionary.
        
        Args:
            data: Dictionary potentially containing secrets.
        
        Returns:
            New dictionary with sensitive values redacted.
        
        Applies SRP: Only redacts, does not modify original or log.
        """
        if not isinstance(data, dict):
            return data
        
        redacted = {}
        for key, value in data.items():
            if SecretRedactor._is_secret_key(key):
                redacted[key] = SecretRedactor.REDACTED
            elif isinstance(value, dict):
                # Recursive redaction for nested dicts
                redacted[key] = SecretRedactor.redact(value)
            elif isinstance(value, list):
                # Redact list items if they are dicts
                redacted[key] = [
                    SecretRedactor.redact(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                redacted[key] = value
        
        return redacted
    
    @staticmethod
    def _is_secret_key(key: str) -> bool:
        """Check if key name indicates sensitive data.
        
        Args:
            key: Dictionary key to check.
        
        Returns:
            True if key matches secret pattern.
        
        Applies ISP: Simple boolean check, no complex logic.
        """
        key_upper = key.upper()
        return any(secret in key_upper for secret in SecretRedactor.SECRET_KEYS)
    
    @staticmethod
    def redact_string(text: str, patterns: Set[str] | None = None) -> str:
        """Redact sensitive patterns from string.
        
        Args:
            text: String potentially containing secrets.
            patterns: Optional set of patterns to redact (uses SECRET_KEYS if None).
        
        Returns:
            String with sensitive patterns redacted.
        
        Applies OCP: Extensible to custom patterns without modifying class.
        """
        if patterns is None:
            patterns = SecretRedactor.SECRET_KEYS
        
        redacted_text = text
        for pattern in patterns:
            # Simple substring replacement (case-insensitive)
            # More sophisticated regex matching could be added
            if pattern.lower() in redacted_text.lower():
                # Find and replace the actual occurrence
                import re
                redacted_text = re.sub(
                    rf'\b{re.escape(pattern)}\b[:\s=]*[^\s,\]}}]+',
                    f'{pattern}={SecretRedactor.REDACTED}',
                    redacted_text,
                    flags=re.IGNORECASE
                )
        
        return redacted_text
