"""FFT-based dominant cycle detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .engine import SeasonalConfig

logger = logging.getLogger(__name__)


def detect_cycle(values: list[float], config: "SeasonalConfig") -> Tuple[Optional[int], float]:
    """Detect dominant cycle period using FFT.

    Args:
        values: Time series values
        config: SeasonalConfig with min_period, max_period, fft_threshold

    Returns:
        (period, confidence) or (None, 0) if no clear cycle
    """
    n = len(values)

    # Detrend to remove DC component
    mean_val = np.mean(values)
    detrended = np.array(values) - mean_val

    # FFT
    fft = np.fft.rfft(detrended)
    power = np.abs(fft) ** 2

    # Find peaks in frequency domain (excluding DC)
    freqs = np.fft.rfftfreq(n)

    # Consider only frequencies within min/max period range
    min_freq = 1.0 / config.max_period
    max_freq = 1.0 / config.min_period

    valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not valid_mask.any():
        return None, 0.0

    valid_power = power[valid_mask]
    valid_freqs = freqs[valid_mask]

    if len(valid_power) == 0:
        return None, 0.0

    # Find dominant frequency
    peak_idx = np.argmax(valid_power)
    peak_power = valid_power[peak_idx]
    total_power = np.sum(power[1:])  # Exclude DC

    if total_power == 0:
        return None, 0.0

    # Confidence based on peak prominence
    confidence = peak_power / total_power

    if confidence < config.fft_threshold:
        return None, 0.0

    dominant_freq = valid_freqs[peak_idx]
    period = int(round(1.0 / dominant_freq))

    return period, min(confidence, 0.95)
