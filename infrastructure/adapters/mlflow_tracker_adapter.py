"""Adapter MLflow para ExperimentTrackerPort.

Implementa el port usando MLflow real. Fail-safe: nunca crashea el pipeline.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, Optional

from iot_machine_learning.domain.ports.experiment_tracker_port import (
    ExperimentTrackerPort,
)

logger = logging.getLogger(__name__)


class MlflowTrackerAdapter(ExperimentTrackerPort):
    """Adapter que implementa ExperimentTrackerPort con MLflow.

    Fail-safe: si MLflow falla, loguea warning pero no propaga excepción.
    Esto garantiza que el pipeline nunca falle por problemas de tracking.

    Attributes:
        _tracking_uri: URI del servidor MLflow (ej: http://localhost:5000).
        _experiment_name: Nombre del experimento en MLflow.
        _enabled: Flag para desactivar sin rewire.
        _run_id: ID del run activo (None si no hay run).
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "zenin-cognitive-pipeline",
        enabled: bool = True,
    ) -> None:
        """Inicializa el adapter MLflow.

        Args:
            tracking_uri: MLflow tracking server URI.
                Default: MLFLOW_TRACKING_URI env var o "http://localhost:5000".
            experiment_name: Nombre del experimento.
            enabled: Si False, todos los métodos son no-op.
        """
        self._tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self._experiment_name = experiment_name
        self._enabled = enabled
        self._run_id: Optional[str] = None
        self._mlflow: Optional[Any] = None

        if not self._enabled:
            logger.info("mlflow_tracker_disabled")
            return

        # Lazy import de MLflow (no fallar en import si no está instalado)
        try:
            import mlflow as _mlflow_module

            self._mlflow = _mlflow_module
            self._mlflow.set_tracking_uri(self._tracking_uri)
            self._mlflow.set_experiment(self._experiment_name)
            logger.info(
                "mlflow_tracker_initialized",
                extra={
                    "tracking_uri": self._tracking_uri,
                    "experiment_name": self._experiment_name,
                },
            )
        except ImportError:
            logger.warning("mlflow_not_installed_tracker_disabled")
            self._enabled = False
        except Exception as exc:
            logger.warning(
                "mlflow_init_failed_tracker_disabled",
                extra={"error": str(exc)},
            )
            self._enabled = False

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Inicia run en MLflow (fail-safe)."""
        if not self._enabled or self._mlflow is None:
            return f"disabled_run_{id(self)}"

        try:
            run = self._mlflow.start_run(run_name=run_name)
            self._run_id = run.info.run_id

            if tags:
                self._mlflow.set_tags(tags)

            logger.debug(
                "mlflow_run_started",
                extra={"run_id": self._run_id, "run_name": run_name},
            )
            return self._run_id

        except Exception as exc:
            logger.warning(
                "mlflow_start_run_failed",
                extra={"error": str(exc), "run_name": run_name},
            )
            return f"failed_run_{id(self)}"

    def end_run(self, status: Optional[str] = None) -> None:
        """Finaliza run en MLflow (fail-safe)."""
        if not self._enabled or self._mlflow is None:
            self._run_id = None
            return

        try:
            self._mlflow.end_run(status=status)
            logger.debug(
                "mlflow_run_ended",
                extra={"run_id": self._run_id, "status": status},
            )
        except Exception as exc:
            logger.warning("mlflow_end_run_failed", extra={"error": str(exc)})
        finally:
            self._run_id = None

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Loguea métrica en MLflow (fail-safe)."""
        if not self._enabled or self._mlflow is None:
            return

        try:
            self._mlflow.log_metric(key, value, step=step)
        except Exception as exc:
            logger.debug(
                "mlflow_log_metric_failed",
                extra={"key": key, "error": str(exc)},
            )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Loguea métricas batch en MLflow (fail-safe)."""
        if not self._enabled or self._mlflow is None:
            return

        try:
            self._mlflow.log_metrics(metrics, step=step)
        except Exception as exc:
            logger.debug(
                "mlflow_log_metrics_failed",
                extra={"keys": list(metrics.keys()), "error": str(exc)},
            )

    def log_param(self, key: str, value: Any) -> None:
        """Loguea parámetro en MLflow (fail-safe)."""
        if not self._enabled or self._mlflow is None:
            return

        try:
            # MLflow solo acepta strings, ints, floats
            if isinstance(value, (int, float, str, bool)):
                self._mlflow.log_param(key, value)
            else:
                self._mlflow.log_param(key, str(value))
        except Exception as exc:
            logger.debug(
                "mlflow_log_param_failed",
                extra={"key": key, "error": str(exc)},
            )

    def log_params(self, params: Dict[str, Any]) -> None:
        """Loguea parámetros batch en MLflow (fail-safe)."""
        if not self._enabled or self._mlflow is None:
            return

        try:
            # Convertir valores no primitivos a string
            safe_params = {
                k: v if isinstance(v, (int, float, str, bool)) else str(v)
                for k, v in params.items()
            }
            self._mlflow.log_params(safe_params)
        except Exception as exc:
            logger.debug(
                "mlflow_log_params_failed",
                extra={"keys": list(params.keys()), "error": str(exc)},
            )

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Loguea artefacto en MLflow (fail-safe)."""
        if not self._enabled or self._mlflow is None:
            return

        try:
            self._mlflow.log_artifact(local_path, artifact_path)
        except Exception as exc:
            logger.warning(
                "mlflow_log_artifact_failed",
                extra={"local_path": local_path, "error": str(exc)},
            )

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Setea tags en MLflow (fail-safe)."""
        if not self._enabled or self._mlflow is None:
            return

        try:
            safe_tags = {
                k: v if isinstance(v, (int, float, str, bool)) else str(v)
                for k, v in tags.items()
            }
            self._mlflow.set_tags(safe_tags)
        except Exception as exc:
            logger.debug(
                "mlflow_set_tags_failed",
                extra={"keys": list(tags.keys()), "error": str(exc)},
            )

    def __enter__(self) -> ExperimentTrackerPort:
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: auto-end_run."""
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end_run(status)
