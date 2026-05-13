"""Security and enterprise features configuration.

Extracted from flags.py as part of refactoring Paso 5.

SEC-CRIT-1: Fail-closed authentication with explicit validation.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class SecurityConfig(BaseModel):
    """Configuration for security and enterprise features.
    
    Responsibilities:
    - API authentication
    - CORS origins
    - Enterprise feature flags
    """

    # --- Security (SEC-CRIT-1: fail-closed) ---
    ML_API_KEY: str = ""  # Must be set in production
    ML_API_KEY_READONLY: str = ""  # Read-only key for GET endpoints (SEC-3)
    ML_AUTH_ENABLED: bool = True  # Default to enabled (fail-closed)
    ML_DEV_MODE: bool = False  # Explicit dev mode flag (must opt-in)
    ML_CORS_ORIGINS: str = (
        "http://localhost:3000,"
        "http://localhost:8080,"
        "http://localhost:5000,"
        "http://localhost:8001,"
        "http://localhost:9090"
    )
    ML_CORS_ALLOW_CREDENTIALS: bool = False  # SEC-2: explicit opt-in required

    # --- Enterprise Features (Fase 3) ---
    ML_ENABLE_DELTA_SPIKE_DETECTION: bool = False
    ML_ENABLE_REGIME_DETECTION: bool = False
    ML_ENABLE_ENSEMBLE_PREDICTOR: bool = False
    ML_ENABLE_AUDIT_LOGGING: bool = False
    ML_ENABLE_PREDICTION_CACHE: bool = False
    ML_ENABLE_VOTING_ANOMALY: bool = False
    ML_ENABLE_CHANGE_POINT_DETECTION: bool = False
    ML_ENABLE_EXPLAINABILITY: bool = False

    # --- Anomaly Config ---
    ML_ANOMALY_VOTING_THRESHOLD: float = 0.5
    ML_ANOMALY_CONTAMINATION: float = 0.1
    
    def validate(self) -> None:
        """Validate security configuration (SEC-CRIT-1).
        
        Raises:
            ValueError: If auth is enabled but no API key is set in production.
        
        Applies SRP: SecurityConfig validates its own invariants.
        """
        if self.ML_AUTH_ENABLED and not self.ML_DEV_MODE:
            if not self.ML_API_KEY:
                raise ValueError(
                    "ML_API_KEY must be set when ML_AUTH_ENABLED=True and ML_DEV_MODE=False. "
                    "Set ML_DEV_MODE=True to explicitly disable auth in development."
                )
    
    @field_validator("ML_API_KEY", "ML_API_KEY_READONLY")
    @classmethod
    def _validate_api_key_format(cls, v: str) -> str:
        """Validate API key format if provided.
        
        Applies ISP: Validator is specific to API key format.
        """
        if v and len(v) < 32:
            raise ValueError("API keys must be at least 32 characters for security")
        return v
