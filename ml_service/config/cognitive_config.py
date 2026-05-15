"""Cognitive memory, neural, and embedding configuration.

Extracted from flags.py as part of refactoring Paso 5.
"""

from __future__ import annotations

from typing import Dict
from pydantic import BaseModel, Field


class CognitiveConfig(BaseModel):
    """Configuration for cognitive memory, neural engines, and embeddings.
    
    Responsibilities:
    - Weaviate cognitive memory
    - Hybrid embeddings for text
    - Attention and neural network features
    - Pipeline performance budgets
    """

    # --- Pipeline Performance ---
    ML_PIPELINE_BUDGET_MS: float = 25000.0  # 25 second max
    ML_ENABLE_PHASE_TIMING: bool = True

    # --- Feature Priority by Speed (CPU optimization) ---
    # FAST — keep enabled (< 2s each)
    ML_ENABLE_DECISION_ENGINE: bool = True
    ML_ENABLE_MONTE_CARLO: bool = True

    # MEDIUM — keep enabled with timeout (2-5s each)
    ML_HYBRID_TIMEOUT_SECONDS: float = 3.0

    # MEDIUM — Cognitive Memory
    ML_COHERE_API_KEY: str = ""  # inline
    ML_MEMORY_MAX_RESULTS: int = 5

    # SLOW — disable until GPU available (> 5s each)
    ML_ENABLE_ATTENTION: bool = False  # too slow for CPU
    ML_ATTENTION_TIMEOUT_SECONDS: float = 3.0
    ML_ENABLE_SNN_FULL: bool = False  # disabled on CPU
    ML_ATTENTION_CONFIDENCE_THRESHOLD: float = 0.5  # Fallback threshold
    ML_ATTENTION_BUDGET_MS: float = 100.0  # Time budget per document

    # --- Hybrid Embeddings (Text Entity Extraction) ---
    ML_ENABLE_HYBRID_EMBEDDINGS: bool = True  # Master switch
    ML_HYBRID_EMBEDDING_DIMENSION: int = 128
    ML_HYBRID_ENTROPY_THRESHOLD: float = 0.5
    ML_HYBRID_PHRASE_MIN_PERSISTENCE: int = 2
    ML_HYBRID_ENTITY_THRESHOLD: float = 0.3

    # --- Cognitive Memory (Weaviate) ---
    ML_ENABLE_COGNITIVE_MEMORY: bool = False
    ML_COGNITIVE_MEMORY_DRY_RUN: bool = True
    ML_COGNITIVE_MEMORY_ASYNC: bool = True
    ML_COGNITIVE_MEMORY_URL: str = ""
    ML_ENABLE_MEMORY_RECALL: bool = False

    # --- Experiment Tracking (EXP-1 / MLflow) ---
    MLFLOW_ENABLED: bool = False
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "zenin-cognitive-pipeline"

    # --- Probabilistic Confidence Calibration (CAL-1) ---
    ML_PROBABILISTIC_CALIBRATION_ENABLED: bool = False
    ML_CALIBRATION_MIN_SAMPLES_PLATT: int = 50
    ML_CALIBRATION_MIN_SAMPLES_ISOTONIC: int = 200
    ML_CALIBRATION_WINDOW_SIZE_PLATT: int = 500
    ML_CALIBRATION_WINDOW_SIZE_ISOTONIC: int = 1000
    ML_CALIBRATION_UPDATE_FREQUENCY_PLATT: int = 10  # How often to refit Platt params
    ML_CALIBRATION_SPARSE_UPDATE_THRESHOLD: int = 100
    ML_CALIBRATION_LOG_ECE_EVERY_N: int = 100
    ML_CALIBRATION_ECE_THRESHOLD: float = 0.1  # ECE > 0.1 triggers WARNING

    # --- Anomaly tracking (Redis/memory) ---
    ML_ANOMALY_TTL_SECONDS: float = 7200.0
    ML_ANOMALY_MAX_ENTRIES_PER_SERIES: int = 500
    ML_ANOMALY_KEY_TTL_SECONDS: int = 3600
    ML_ANOMALY_TRACKER_BACKEND: str = "memory"  # memory | redis

    # --- Orchestrator Integration (batch runner) ---
    ML_USE_COGNITIVE_ORCHESTRATOR: bool = False  # Safe rollout: False until validated

    # --- Domain Boundary Check (EJE 4 fix) ---
    ML_DOMAIN_BOUNDARY_ENABLED: bool = False

    # --- Signal Coherence Check (EJE 2 fix) ---
    ML_COHERENCE_CHECK_ENABLED: bool = False

    # --- Decision Arbiter (EJE 1 fix) ---
    ML_DECISION_ARBITER_ENABLED: bool = False

    # --- Confidence Calibration (EJE 6 fix) ---
    ML_CONFIDENCE_CALIBRATION_ENABLED: bool = False

    # --- Action Guard (EJE 5 fix) ---
    ML_ACTION_GUARD_ENABLED: bool = False

    # --- Narrative Unification (EJE 7 fix) ---
    ML_NARRATIVE_UNIFICATION_ENABLED: bool = False

    # --- Pipeline optional phases (used by PipelineExecutorFactory) ---
    ML_EXPLAINABILITY_ENABLED: bool = False
    ML_NARRATIVE_ENABLED: bool = False
    
    # --- Zenin Deterministic Mode (Coherence Fix) ---
    ZENIN_DETERMINISTIC_MODE: bool = False
    ZENIN_ANALYSIS_SEED: int = 42
    
    # --- Drift Detection (FASE 1) ---
    # WINDOW SIZE GUIDANCE (FASE-24):
    # Los siguientes parámetros asumen frecuencia de muestreo ~1Hz.
    # Para otras frecuencias, escalar proporcionalmente:
    #   window_size = valor_base * (frecuencia_hz / 1.0)
    # Ejemplo: ML_ADWIN_MAX_WINDOW=1000 a 10Hz → ajustar a 10000
    # PENDING_CALIBRATION: Hacer configurable por serie en Fase futura.
    
    ML_SAMPLING_FREQUENCY_HZ: float = 1.0
    # Frecuencia de muestreo asumida. Usada como referencia para calibración
    # manual de window sizes. No conectado a lógica aún (PENDING: Fase futura
    # automatizará ajuste proporcional).
    
    ML_ENABLE_DRIFT_DETECTION: bool = True  # Master switch for concept drift detection
    ML_DRIFT_DELTA: float = 0.005  # Page-Hinkley delta: magnitude of changes to detect
    ML_DRIFT_LAMBDA: float = 50.0  # Page-Hinkley lambda: detection threshold
    ML_DRIFT_ALPHA: float = 0.9999  # Page-Hinkley alpha: forgetting factor (0=no forgetting)
    ML_ENABLE_ADWIN: bool = False  # Use ADWIN instead of Page-Hinkley
    ML_ADWIN_DELTA: float = 0.002  # ADWIN confidence parameter
    ML_ADWIN_MAX_WINDOW: int = 1000  # ADWIN maximum window size
    ML_DRIFT_COOLDOWN_SECONDS: float = 300.0  # Minimum seconds between drift resets per series
    
    # --- Seasonality (FASE 2) ---
    ML_ENABLE_SEASONALITY: bool = False  # Master switch for seasonal decomposition (backward compat)
    ML_SEASONAL_PERIOD_DEFAULT: int = 24  # Default seasonal period (24 hours for IoT)
    ML_SEASONAL_USE_STL: bool = False  # Use STL instead of FFT (requires statsmodels)
    ML_SEASONAL_MIN_POINTS: int = 48  # Minimum points required (2 cycles)

    # --- Bayesian Weight Tracker Variance Estimation ---
    # WINDOW SIZE MISMATCH RATIONALE (FASE-25):
    # Los siguientes window sizes son INTENCIONALMENTE diferentes:
    # - Baseline window=20: contexto local inmediato (moving average)
    # - ErrorDriftDetector window=100: detección de tendencias largas
    # - AdaptiveScaler volatility_history=20: respuesta rápida a cambios
    # - Kalman innovation_window=20: estimación de ruido local
    # Diferencia Baseline(20) vs ErrorDrift(100) es intencional:
    #   20 obs = respuesta rápida al valor actual
    #   100 obs = detección de drift sistemático de largo plazo
    # PENDING_CALIBRATION: Ajustar proporcionalmente a ML_SAMPLING_FREQUENCY_HZ
    
    ML_BAYES_SIGMA2_OBS_DEFAULT: float = 1.0  # fallback sigma2_obs when insufficient error samples
    ML_BAYES_SIGMA2_OBS_MIN: float = 0.01  # minimum sigma2_obs to prevent zero variance
    ML_BAYES_VARIANCE_WINDOW: int = 20  # number of recent errors per engine for variance estimation
    ML_BAYES_VARIANCE_MIN_SAMPLES: int = 5  # minimum error samples to estimate empirical variance
    
    # --- Inhibition Suppression Decay (FASE-22) ---
    ML_INHIBITION_SUPPRESSION_HALF_LIFE_S: float = 300.0
    # 5 minutos = balance entre respuesta rápida y estabilidad.
    # Para datos de alta frecuencia (>1Hz) considerar reducir a 60s.
    # Pendiente calibración empírica por tipo de sensor.
    
    # --- Plasticity Regime-Specific Alphas (FASE-23) ---
    ML_PLASTICITY_REGIME_ALPHAS: Dict[str, float] = Field(default_factory=lambda: {
        "STABLE": 0.08,      # Memoria ~12 obs, baja varianza
        "TRENDING": 0.18,    # Memoria ~6 obs, adaptación moderada
        "VOLATILE": 0.30,    # Memoria ~3 obs, respuesta rápida
        "NOISY": 0.05,       # Memoria ~20 obs, filtro agresivo
        "DEFAULT": 0.15,     # Punto medio: memoria ~6 obs
    })
    # Regime-specific EMA learning rates. Higher alpha = faster adaptation.
    # Calibration pending on historical data.
    # Source: moved from docs/configuration.md to code (FASE-23)
