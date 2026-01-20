from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineMetadata:
    """Metadatos para el modelo baseline.

    Se almacena en la tabla ``ml_models``.
    """

    name: str
    model_type: str
    version: str


BASELINE_MOVING_AVERAGE = BaselineMetadata(
    name="baseline_moving_average",
    model_type="baseline",
    version="1.0.0",
)
