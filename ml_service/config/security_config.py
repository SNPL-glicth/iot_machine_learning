"""Security and enterprise features configuration.

Extracted from flags.py as part of refactoring Paso 5.
"""

from __future__ import annotations

from pydantic import BaseModel


class SecurityConfig(BaseModel):
    """Configuration for security and enterprise features.
    
    Responsibilities:
    - API authentication
    - CORS origins
    - Enterprise feature flags
    """

    # --- Security ---
    ML_API_KEY: str = ""  # Empty = auth disabled (dev mode)
    ML_CORS_ORIGINS: str = (
        "http://localhost:3000,"
        "http://localhost:8080,"
        "http://localhost:5000,"
        "http://localhost:8001,"
        "http://localhost:9090"
    )

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
