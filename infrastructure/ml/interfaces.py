"""Interfaces base para UTSAE — contratos que todo motor/filtro debe cumplir.

Decisiones de diseño:
- ABC en lugar de Protocol: queremos forzar herencia explícita para que
  errores de implementación se detecten al instanciar, no al llamar.
- PredictionResult es frozen dataclass: inmutable una vez creado, seguro
  para logging y serialización sin copias defensivas.
- IdentityFilter incluido aquí como fallback canónico (no en archivo aparte)
  para evitar imports circulares y mantener el contrato junto al no-op.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass(frozen=True)
class PredictionResult:
    """Resultado unificado de cualquier motor de predicción.

    Attributes:
        predicted_value: Valor predicho para el siguiente paso temporal.
        confidence: Confianza del motor en la predicción (0.0–1.0).
        trend: Dirección de la tendencia detectada.
        metadata: Información específica del motor (derivadas, orden, flags).
    """

    predicted_value: float
    confidence: float  # 0.0 (sin confianza) a 1.0 (total confianza)
    trend: Literal["up", "down", "stable"]
    metadata: dict = field(default_factory=dict)


class PredictionEngine(ABC):
    """Interfaz abstracta para motores de predicción.

    Todo motor nuevo debe:
    1. Implementar ``name``, ``predict``, ``can_handle``.
    2. Retornar ``PredictionResult`` con metadata relevante.
    3. Documentar en ``can_handle`` el mínimo de puntos requerido.
    4. Manejar edge cases (pocos datos, NaN) de forma explícita.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre identificador del motor (para logging/métricas)."""
        ...

    @abstractmethod
    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        """Genera predicción a partir de una ventana de valores.

        Args:
            values: Serie temporal ordenada cronológicamente
                (más antiguo primero, más reciente al final).
            timestamps: Timestamps Unix opcionales correspondientes a
                cada valor.  Si ``None``, se asume muestreo uniforme
                con Δt = 1.

        Returns:
            ``PredictionResult`` con valor predicho, confianza, trend y
            metadata específica del motor.

        Raises:
            ValueError: Si ``values`` está vacía o contiene NaN/Inf.
        """
        ...

    @abstractmethod
    def can_handle(self, n_points: int) -> bool:
        """Indica si el motor puede operar con *n_points* datos.

        Args:
            n_points: Número de puntos disponibles en la ventana.

        Returns:
            ``True`` si el motor puede generar una predicción válida.
        """
        ...

    def supports_uncertainty(self) -> bool:
        """Indica si el motor provee intervalos de confianza.

        Cuando retorna ``True``, ``PredictionResult.metadata`` debe
        incluir la clave ``"confidence_interval"`` con una tupla
        ``(lower, upper)``.

        Returns:
            ``True`` si metadata incluye ``confidence_interval``.
        """
        return False


class SignalFilter(ABC):
    """Filtro de señal aplicable pre o post predicción.

    Contrato:
    - ``filter_value``: para procesamiento online (una lectura a la vez).
    - ``filter``: para procesamiento batch (serie completa).
    - ``reset``: limpia estado interno (warmup, buffers).

    Implementaciones deben mantener estado por ``series_id`` para que
    series distintas no interfieran entre sí.
    """

    @abstractmethod
    def filter_value(self, series_id: str, value: float) -> float:
        """Filtra un valor individual (stream online).

        Args:
            series_id: Identificador de la serie (estado independiente por serie).
            value: Valor crudo de la observación.

        Returns:
            Valor filtrado/suavizado.
        """
        ...

    @abstractmethod
    def filter(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> List[float]:
        """Filtra una serie completa (batch).

        Args:
            values: Serie temporal completa.
            timestamps: Timestamps Unix correspondientes.

        Returns:
            Serie filtrada con el mismo tamaño que la entrada.
        """
        ...

    @abstractmethod
    def reset(self, series_id: Optional[str] = None) -> None:
        """Reinicia estado interno del filtro.

        Args:
            series_id: Si se provee, resetea solo esa serie.
                Si ``None``, resetea todas las series.
        """
        ...


class IdentityFilter(SignalFilter):
    """Filtro no-op: retorna la entrada sin modificar.

    Se usa como fallback cuando los feature flags desactivan Kalman
    u otros filtros.  No mantiene estado.
    """

    def filter_value(self, series_id: str, value: float) -> float:
        """Retorna el valor sin modificar."""
        return value

    def filter(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> List[float]:
        """Retorna la serie sin modificar."""
        return list(values)

    def reset(self, series_id: Optional[str] = None) -> None:
        """No-op: no hay estado que resetear."""
        pass
