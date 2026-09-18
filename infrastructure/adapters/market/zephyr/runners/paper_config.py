"""Configuration and predictor/calibrator persistence for PaperBotRunner."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from iot_machine_learning.domain.entities.market.calibration.pipeline import (
    AdaptiveCalibrator,
    export_calibrator_state,
    import_calibrator_state,
)
from iot_machine_learning.domain.entities.market.replay.baselines import (
    EmaCrossoverPredictor,
    MomentumPredictor,
    NaivePredictor,
)


@dataclass(frozen=True)
class PaperBotConfig:
    """Configuración del experimento paper (todo explícito, nada oculto)."""

    symbol: str = "BTC-USD"
    interval_seconds: int = 60
    horizons_seconds: tuple[int, ...] = (60, 300, 900)
    predictor_name: str = "momentum"
    neutral_margin: float = 0.05
    require_calibrated: bool = True
    window_candles: int = 180
    require_positive_net: bool = True

    def __post_init__(self) -> None:
        if not self.horizons_seconds:
            raise ValueError("horizons no puede ser vacío")
        span = self.window_candles * self.interval_seconds
        needed = max(self.horizons_seconds) + 2 * self.interval_seconds
        if span <= needed:
            raise ValueError(
                f"ventana insuficiente: {self.window_candles} velas × "
                f"{self.interval_seconds}s cubre {span}s pero el horizonte "
                f"máximo {max(self.horizons_seconds)}s necesita > {needed}s"
            )


def make_predictor(name: str):
    """Predictor crudo por nombre (los tres baselines del replay)."""
    predictors = {
        "naive": NaivePredictor,
        "momentum": MomentumPredictor,
        "ema-crossover": EmaCrossoverPredictor,
    }
    if name not in predictors:
        raise ValueError(f"predictor desconocido: {name!r} ({sorted(predictors)})")
    return predictors[name]()


def load_calibrator_state(path: Path) -> AdaptiveCalibrator:
    """Carga el artefacto JSON del calibrador aceptado offline."""
    state = json.loads(path.read_text(encoding="utf-8"))
    return import_calibrator_state(state)


def save_calibrator_state(calibrator: AdaptiveCalibrator, path: Path) -> None:
    """Exporta el artefacto JSON (post-refit offline aceptado)."""
    path.write_text(
        json.dumps(export_calibrator_state(calibrator), indent=2),
        encoding="utf-8",
    )
