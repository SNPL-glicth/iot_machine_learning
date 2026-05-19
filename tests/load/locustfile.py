"""Locust entry point — PROD-3 load testing.

Run: locust -f tests/load/locustfile.py --host http://localhost:8000
"""
from __future__ import annotations

import os

from locust import HttpUser, between, task

from scenarios.scenario_1000_sensors import SensorIngestionTasks
from scenarios.scenario_cognitive_load import CognitiveLoadTasks
from scenarios.scenario_redis_failover import RedisFailoverTasks

API_KEY = os.environ.get("ML_API_KEY", "dev-key")


class MLLoadUser(HttpUser):
    """Composite load user mixing all three scenarios."""

    wait_time = between(0.5, 2.0)
    abstract = False

    def on_start(self):
        self.client.headers["X-API-Key"] = API_KEY

    @task(70)
    def sensor_ingest(self):
        SensorIngestionTasks(self).run()

    @task(20)
    def cognitive_predict(self):
        CognitiveLoadTasks(self).run()

    @task(10)
    def redis_failover(self):
        RedisFailoverTasks(self).run()
