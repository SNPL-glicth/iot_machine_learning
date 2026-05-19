"""PROD-3 Escenario 3: Redis failover.

Objetivo: validar que circuit breaker abre y sistema sigue respondiendo.
"""
from __future__ import annotations


class RedisFailoverTasks:
    """Verifica health y circuit breaker durante caída de Redis."""

    def __init__(self, user):
        self.client = user.client

    def run(self):
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"health_down {resp.status_code}")
                return
            data = resp.json()
            cb = data.get("tsdb_circuit", "unknown")
            degraded = data.get("degraded", False)
            if cb == "open" and not degraded:
                resp.failure("circuit_open_but_not_degraded")
            else:
                resp.success()
        with self.client.get("/ml/metrics", catch_response=True) as m:
            if m.status_code != 200:
                m.failure(f"metrics_down {m.status_code}")
            else:
                m.success()
