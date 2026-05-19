"""PROD-3 Escenario 2: carga cognitiva intensiva.

Objetivo: saturar el pipeline de predicción y medir latencias por fase.
"""
from __future__ import annotations

import random


class CognitiveLoadTasks:
    """Ejecuta predicciones ML con ventanas variadas."""

    def __init__(self, user):
        self.client = user.client

    def run(self):
        sensor_id = random.randint(1, 100)
        payload = {
            "sensor_id": sensor_id,
            "horizon_minutes": random.choice([5, 10, 30]),
            "window": random.choice([12, 24, 48]),
            "dedupe_minutes": 0,
        }
        with self.client.post("/ml/predict", json=payload, catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                proc = data.get("processing_time_ms", 0)
                if proc > 500:
                    resp.failure(f"slow_prediction {proc}ms")
                else:
                    resp.success()
            elif resp.status_code == 202:
                resp.success()
            else:
                resp.failure(f"predict_failed {resp.status_code}")
