"""Anomaly explanation narration — human-readable text generation.

Components:
    - build_anomaly_explanation: Generates text from detector votes
"""

from .builder import build_anomaly_explanation

__all__ = ["build_anomaly_explanation"]
