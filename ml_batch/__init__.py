"""ML Batch Runner facade (E-13).

Re-exports from ml_service.runners for independent deployment.
Existing imports via ml_service.runners continue to work unchanged.

Usage:
    from ml_batch import run_batch_cycle
    run_batch_cycle()
"""

from __future__ import annotations


def run_batch_cycle():
    """Execute one batch prediction cycle."""
    from iot_ingest_services.jobs.batch.runner import run_once
    run_once()
