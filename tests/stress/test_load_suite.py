"""Load testing suite — configurable sensors, frequency, throughput, memory, CPU.

No external frameworks. Pure stdlib + project code.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pytest

from iot_machine_learning.domain.entities.iot.sensor_reading import Reading
from iot_machine_learning.ml_service.consumers.sliding_window import (
    SlidingWindowStore,
)
from iot_machine_learning.infrastructure.ml.cognitive.fusion.engine_selector import (
    WeightedFusion,
)
from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception, InhibitionState,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.detector import (
    VotingAnomalyDetector,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.config import (
    AnomalyDetectorConfig,
)
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow

logger = logging.getLogger(__name__)

_RANDOM = random.Random(42)


@dataclass(frozen=True)
class LoadConfig:
    n_sensors: int = 100
    duration_seconds: float = 5.0
    frequency_hz: float = 10.0
    window_size: int = 20
    max_sensors: int = 2000


@dataclass(frozen=True)
class LoadResult:
    total_events: int
    total_time_seconds: float
    throughput_hz: float
    p99_latency_ms: float
    max_latency_ms: float
    error_count: int
    memory_growth_bytes: int
    peak_cpu_fraction: float


class LoadHarness:
    """Runs configurable load against ZENIN subsystems."""

    def __init__(self, config: LoadConfig) -> None:
        self.config = config
        self._store = SlidingWindowStore(
            max_size=config.window_size,
            max_sensors=config.max_sensors,
            enable_proactive_cleanup=False,
        )
        self._fusion = WeightedFusion()
        self._detector = VotingAnomalyDetector(
            config=AnomalyDetectorConfig(min_training_points=10),
        )
        self._errors = 0
        self._latencies: List[float] = []

    def run(self) -> LoadResult:
        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()

        events_target = int(self.config.duration_seconds * self.config.frequency_hz * self.config.n_sensors)
        interval = 1.0 / self.config.frequency_hz

        t_start = time.monotonic()
        events = 0

        def sensor_loop(sensor_idx: int) -> None:
            nonlocal events
            sid = f"load_sensor_{sensor_idx}"
            deadline = t_start + self.config.duration_seconds
            next_t = t_start
            while time.monotonic() < deadline and events < events_target:
                t0 = time.monotonic()
                try:
                    reading = Reading(
                        series_id=sid,
                        value=_RANDOM.gauss(22.0, 1.5),
                        timestamp=t0,
                    )
                    self._store.append(reading)
                    win = self._store.get_window(sid)
                    if len(win) >= self.config.window_size:
                        sw = SensorWindow(
                            series_id=str(sensor_idx),
                            readings=[
                                Reading(
                                    series_id=str(sensor_idx),
                                    value=r.value,
                                    timestamp=r.timestamp,
                                )
                                for r in win
                            ],
                        )
                        self._detector.detect(sw)
                except Exception:
                    self._errors += 1
                finally:
                    elapsed = time.monotonic() - t0
                    self._latencies.append(elapsed)
                    events += 1
                next_t += interval
                sleep_time = next_t - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.config.n_sensors, 64)) as pool:
            list(pool.map(sensor_loop, range(self.config.n_sensors)))

        total_time = time.monotonic() - t_start
        snap_after = tracemalloc.take_snapshot()
        diff = snap_after.compare_to(snap_before, "lineno")
        growth = sum(stat.size_diff for stat in diff if stat.size_diff > 0)
        tracemalloc.stop()

        sorted_lat = sorted(self._latencies)
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if sorted_lat else 0.0
        p_max = sorted_lat[-1] if sorted_lat else 0.0
        throughput = events / total_time if total_time > 0 else 0.0

        self._store.close()

        return LoadResult(
            total_events=events,
            total_time_seconds=total_time,
            throughput_hz=throughput,
            p99_latency_ms=p99 * 1000.0,
            max_latency_ms=p_max * 1000.0,
            error_count=self._errors,
            memory_growth_bytes=growth,
            peak_cpu_fraction=0.0,  # Could parse /proc/stat if needed; skip for portability
        )


class TestLoadSuite:
    """Parameterized load tests."""

    @pytest.mark.parametrize("n_sensors,frequency_hz,duration", [
        (100, 10.0, 3.0),
        (500, 5.0, 3.0),
        (1000, 2.0, 3.0),
    ])
    def test_load_varied_scale(
        self,
        n_sensors: int,
        frequency_hz: float,
        duration: float,
    ) -> None:
        cfg = LoadConfig(
            n_sensors=n_sensors,
            frequency_hz=frequency_hz,
            duration_seconds=duration,
            max_sensors=n_sensors + 500,
        )
        harness = LoadHarness(cfg)
        result = harness.run()

        expected_events = int(duration * frequency_hz * n_sensors)
        assert result.total_events >= expected_events * 0.8, (
            f"events={result.total_events} < 80% of target={expected_events}"
        )
        assert result.error_count / max(result.total_events, 1) < 0.01
        assert result.p99_latency_ms < 500.0, (
            f"p99_latency={result.p99_latency_ms:.1f}ms exceeds 500ms"
        )
        assert result.memory_growth_bytes < 20 * 1024 * 1024, (
            f"memory_growth={result.memory_growth_bytes / 1024 / 1024:.1f}MB"
        )

    @pytest.mark.slow
    def test_sustained_throughput_100_sensors(self) -> None:
        cfg = LoadConfig(n_sensors=100, frequency_hz=20.0, duration_seconds=10.0)
        harness = LoadHarness(cfg)
        result = harness.run()
        assert result.throughput_hz >= 1000.0, (
            f"throughput={result.throughput_hz:.1f}Hz < 1000Hz"
        )
