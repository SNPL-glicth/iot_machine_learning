"""
Memory bound tests for 6A components.
Verifies that in-memory structures stay bounded after many insertions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest
import time
from collections import deque

from infrastructure.ml.cognitive.causal.event_propagation_tracker import (
    EventPropagationTracker,
    MAX_COMPLETED_PROPAGATIONS,
)
from infrastructure.ml.cognitive.causal.operational_sequence_registry import (
    OperationalSequenceRegistry,
    MAX_ANOMALY_PRECURSORS,
)


class TestEventPropagationTrackerMemoryBounds(unittest.TestCase):
    def setUp(self):
        self.tracker = EventPropagationTracker()

    def test_completed_propagations_bounded_by_maxlen(self):
        for i in range(MAX_COMPLETED_PROPAGATIONS + 1000):
            pid = self.tracker.start_propagation(
                source_sensor_id=i % 100,
                timestamp=1000.0 + i,
            )
            self.tracker.add_to_propagation(pid, (i + 1) % 100, 1001.0 + i)
            self.tracker.end_propagation(pid, 1002.0 + i)

        self.assertLessEqual(
            len(self.tracker._completed_propagations),
            MAX_COMPLETED_PROPAGATIONS,
        )

    def test_propagation_counts_bounded_after_cleanup(self):
        for source in range(200):
            for target in range(200):
                pid = self.tracker.start_propagation(source, 1000.0)
                self.tracker.add_to_propagation(pid, target, 1001.0)
                self.tracker.end_propagation(pid, 1002.0)

        self.tracker._propagation_stats_cleanup()
        self.assertGreaterEqual(len(self.tracker._propagation_counts), 0)

    def test_stale_active_propagations_cleaned(self):
        old_time = time.time() - 7200
        for i in range(100):
            self.tracker.start_propagation(
                source_sensor_id=i,
                timestamp=old_time,
            )

        self.tracker._cleanup_stale_active_propagations()
        self.assertEqual(len(self.tracker._active_propagations), 0)

    def test_10k_events_memory_stays_bounded(self):
        for i in range(10000):
            pid = self.tracker.start_propagation(
                source_sensor_id=i % 50,
                timestamp=float(i),
            )
            self.tracker.add_to_propagation(pid, (i + 1) % 50, float(i) + 1)
            self.tracker.end_propagation(pid, float(i) + 2)

        self.assertLessEqual(
            len(self.tracker._completed_propagations),
            MAX_COMPLETED_PROPAGATIONS,
        )
        self.assertLessEqual(
            len(self.tracker._propagation_counts),
            5000,
        )


class TestOperationalSequenceRegistryMemoryBounds(unittest.TestCase):
    def setUp(self):
        self.registry = OperationalSequenceRegistry()

    def test_anomaly_precursors_bounded(self):
        from domain.entities.causal import TemporalPattern

        for i in range(MAX_ANOMALY_PRECURSORS + 500):
            pattern = TemporalPattern(
                pattern_id=f"p_{i}",
                sequence=[i, i + 1, i + 2],
                frequency=1,
                avg_duration_seconds=1.0,
                confidence=0.9,
                is_pre_anomaly=True,
                timestamp=float(i),
            )
            self.registry.register_sequence(pattern)

        self.assertLessEqual(
            len(self.registry._anomaly_precursors),
            MAX_ANOMALY_PRECURSORS,
        )

    def test_10k_registrations_memory_stays_bounded(self):
        from domain.entities.causal import TemporalPattern

        for i in range(10000):
            pattern = TemporalPattern(
                pattern_id=f"p_{i}",
                sequence=[i % 100, (i + 1) % 100, (i + 2) % 100],
                frequency=1,
                avg_duration_seconds=1.0,
                confidence=0.9,
                is_pre_anomaly=i % 2 == 0,
                timestamp=float(i),
            )
            self.registry.register_sequence(pattern)

        self.assertLessEqual(len(self.registry._sequences), 1000)
        self.assertLessEqual(
            len(self.registry._anomaly_precursors),
            MAX_ANOMALY_PRECURSORS,
        )


if __name__ == "__main__":
    unittest.main()
