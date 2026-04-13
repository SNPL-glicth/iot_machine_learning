"""Batch processing, streaming, and ingest configuration.

Extracted from flags.py as part of refactoring Paso 5.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class BatchConfig(BaseModel):
    """Configuration for batch runners, stream processing, and ingest.
    
    Responsibilities:
    - Batch runner workers and circuit breakers
    - Stream prediction settings
    - Sliding window eviction
    - Ingest circuit breaker
    - MQTT async processing
    """

    # --- Batch Config ---
    ML_BATCH_MAX_WORKERS: int = 4
    ML_BATCH_CIRCUIT_BREAKER_THRESHOLD: int = 10

    # --- Batch Parallelism (E-4 / RC-2 fix) ---
    ML_BATCH_PARALLEL_WORKERS: int = 1  # 1 = sequential (backward compat)

    # --- Stream Prediction Dedup (E-2 / RC-1 fix) ---
    ML_STREAM_PREDICTIONS_ENABLED: bool = False  # Default: batch only

    # --- Sliding Window Eviction (E-1 / RC-3 fix) ---
    ML_SLIDING_WINDOW_MAX_SENSORS: int = 1000
    ML_SLIDING_WINDOW_TTL_SECONDS: int = 3600

    # --- Ingest Circuit Breaker (E-3 / RC-4 fix) ---
    ML_INGEST_CIRCUIT_BREAKER_ENABLED: bool = True
    ML_INGEST_CB_FAILURE_THRESHOLD: int = 5
    ML_INGEST_CB_TIMEOUT_SECONDS: int = 30

    # --- Batch Runner Enterprise Bridge ---
    ML_BATCH_USE_ENTERPRISE: bool = False
    ML_BATCH_ENTERPRISE_SENSORS: Optional[str] = None
    ML_BATCH_BASELINE_ONLY_SENSORS: Optional[str] = None

    # --- Performance Fixes (H-ML-4, H-ML-8, H-ING-2) ---
    ML_ENTERPRISE_USE_PRELOADED_DATA: bool = True
    ML_STREAM_USE_SLIDING_WINDOW: bool = True
    ML_MQTT_ASYNC_PROCESSING: bool = True
    ML_MQTT_QUEUE_SIZE: int = 1000
    ML_MQTT_NUM_WORKERS: int = 4
