"""PROD-3 Escenario 1: 1000 sensores ingestando asincrónicamente.

Objetivo: verificar throughput del stream consumer con carga real.
"""
from __future__ import annotations

import random
import time


class SensorIngestionTasks:
    """Simula ingestión masiva de lecturas de sensores."""

    def __init__(self, user):
        self.client = user.client

    def run(self):
        sensor_id = random.randint(1, 1000)
        payload = {
            "sensor_id": sensor_id,
            "value": round(random.uniform(15.0, 35.0), 2),
            "timestamp": time.time(),
            "metadata": {"source": "locust", "batch": random.randint(1, 10)},
        }
        with self.client.post("/telemetry/ingest", json=payload, catch_response=True) as resp:
            if resp.status_code in (200, 202):
                resp.success()
            else:
                resp.failure(f"ingest_failed {resp.status_code}")
