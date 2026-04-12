"""Puerto abstracto para experiment tracking (MLflow, Weights&Biases, etc.).

Responsabilidad: Definir interfaz para logging de experimentos sin
acoplarse a implementación específica (hexagonal architecture).

Zero dependencias de MLflow o cualquier librería de tracking aquí.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ExperimentTrackerPort(ABC):
    """Puerto abstracto para tracking de experimentos ML.

    Implementaciones concretas:
    - MlflowTrackerAdapter: MLflow real (infrastructure)
    - NullExperimentTracker: No-op para tests y modo sin tracking (domain)

    Design notes:
    - Todos los métodos son fail-safe: nunca lanzan excepciones al caller.
    - Context manager support para uso con ``with`` statement.
    - Tags y métricas son Dict[str, Any] para flexibilidad.
    """

    @abstractmethod
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Inicia un nuevo run de experimento.

        Args:
            run_name: Nombre descriptivo del run (ej: "predict_sensor_123").
            tags: Tags/metadata para el run (ej: {"model_version": "0.2.1"}).

        Returns:
            run_id: Identificador único del run iniciado.
        """
        ...

    @abstractmethod
    def end_run(self, status: Optional[str] = None) -> None:
        """Finaliza el run actual.

        Args:
            status: Estado final ("FINISHED", "FAILED", "KILLED").
                Default: "FINISHED".
        """
        ...

    @abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Loguea una métrica numérica.

        Args:
            key: Nombre de la métrica (ej: "mae", "confidence_score").
            value: Valor numérico.
            step: Step opcional (ej: número de predicción/plasticity update).
        """
        ...

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Loguea múltiples métricas en batch.

        Args:
            metrics: Dict {nombre: valor}.
            step: Step opcional.
        """
        ...

    @abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        """Loguea un parámetro (string, int, float, bool).

        Args:
            key: Nombre del parámetro (ej: "engine_name", "regime").
            value: Valor del parámetro.
        """
        ...

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Loguea múltiples parámetros en batch.

        Args:
            params: Dict {nombre: valor}.
        """
        ...

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Loguea un artefacto (archivo).

        Args:
            local_path: Ruta local al archivo.
            artifact_path: Ruta destino opcional dentro del artifact store.
        """
        ...

    @abstractmethod
    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Setea tags en el run actual (sobrescribe si existen).

        Args:
            tags: Dict {nombre: valor}.
        """
        ...

    @abstractmethod
    def __enter__(self) -> ExperimentTrackerPort:
        """Context manager entry."""
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: auto-end_run with appropriate status."""
        ...


class NullExperimentTracker(ExperimentTrackerPort):
    """No-op tracker para tests y modo sin tracking.

    Implementa el port sin hacer nada (fail-safe por diseño).
    Todos los métodos son no-op pero mantienen la interfaz.
    """

    def __init__(self) -> None:
        self._run_id: Optional[str] = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> str:
        """No-op: retorna run_id dummy."""
        self._run_id = f"null_run_{id(self)}"
        return self._run_id

    def end_run(self, status: Optional[str] = None) -> None:
        """No-op."""
        self._run_id = None

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """No-op."""
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """No-op."""
        pass

    def log_param(self, key: str, value: Any) -> None:
        """No-op."""
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        """No-op."""
        pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """No-op."""
        pass

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """No-op."""
        pass

    def __enter__(self) -> ExperimentTrackerPort:
        """Context manager: start_run()."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager: end_run() with status based on exception."""
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end_run(status)
