"""ml_batch — Facade del runner batch ML.

Expone run_batch_cycle() como punto de entrada del ciclo batch.
El runner de producción real está en iot_ingest_services/jobs/ml_batch_runner.py.
Este módulo es el facade interno para dev/test.
"""

from __future__ import annotations


def run_batch_cycle() -> dict:
    """Ejecuta un ciclo batch de predicciones ML.

    Returns:
        Dict con resultados del ciclo: sensors procesados, predicciones, errores.
    """
    try:
        from iot_machine_learning.ml_service.runners.ml_batch_runner import run_once
        return run_once() or {}
    except Exception as exc:
        return {"error": str(exc), "sensors_processed": 0}


__all__ = ["run_batch_cycle"]
