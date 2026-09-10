"""Estados latentes de mercado (FASE 2).

El mercado no se comporta igual todo el tiempo. Estos estados NO están
etiquetados: se infieren por similitud a prototipos documentados
(``model``) y se suavizan en el tiempo (``filter``). El clasificador
determinista de ``replay/regime.py`` sigue intacto como baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = ["LatentRegime", "RegimePosterior"]


class LatentRegime(Enum):
    """Taxonomía latente (agnóstica a dirección: TRENDING puede ser↗ o ↘)."""

    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    LOW_VOL = "LOW_VOL"
    HIGH_VOL = "HIGH_VOL"
    BREAKOUT = "BREAKOUT"
    CRASH = "CRASH"


@dataclass(frozen=True, slots=True, kw_only=True)
class RegimePosterior:
    """Posterior blanda sobre los 6 estados (orden de ``LatentRegime``)."""

    probs: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.probs) != len(LatentRegime):
            raise ValueError(
                f"probs debe tener {len(LatentRegime)} elementos, "
                f"hay {len(self.probs)}"
            )
        if any(not math.isfinite(p) or p < 0 for p in self.probs):
            raise ValueError(f"probs inválidas: {self.probs!r}")
        total = sum(self.probs)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"probs deben sumar 1.0, suman {total!r}")

    def __getitem__(self, regime: LatentRegime) -> float:
        return self.probs[list(LatentRegime).index(regime)]

    @property
    def most_likely(self) -> LatentRegime:
        """Estado MAP (desempate por orden del enum, determinista)."""
        best = max(range(len(self.probs)), key=lambda i: self.probs[i])
        return list(LatentRegime)[best]

    @property
    def confidence(self) -> float:
        """Probabilidad del estado MAP."""
        return self.probs[list(LatentRegime).index(self.most_likely)]

    @property
    def entropy(self) -> float:
        """Entropía (nats): 0 = certeza, ln(6) ≈ 1.79 = uniforme."""
        return -sum(p * math.log(p) for p in self.probs if p > 0)

    def to_dict(self) -> dict[str, float]:
        """Nombre → probabilidad (para ContextKey/dashboard)."""
        return {r.value: self.probs[i] for i, r in enumerate(LatentRegime)}
