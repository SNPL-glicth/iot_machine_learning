"""AdaptationGuard (FASE 8) — ACCEPT / REJECT antes de tocar el modelo.

Peso propuesto ≠ peso aplicado: el guard decide con reglas explícitas y
registra CADA chequeo (quién, qué, por qué), para que el sistema sea
auditable, reversible y estadísticamente defendible:

    ✓ min_n                        (muestra mínima por contexto)
    ✓ suficiente historial         (días distintos observados)
    ✓ no INVALIDATED / no STALE    (solo outcomes reales, fuente limpia)
    ✓ reward válido                (finito, muestra > 0)
    ✓ mejora estadísticamente razonable (Wilson lower bound de accuracy > 0.5)
    ✓ cambio máximo de peso        (|Δ| <= max_change)
    ✓ suma de pesos = 1            (renormalización exacta por contexto)
    ✓ ningún experto desaparece    (piso min_weight)
    ✓ versión anterior preservada  (parent_version + append-only)

Regla de piedra de FASE 8: nada de esto ocurre sin haber observado el
outcome externo (el análisis solo consume filas evaluadas del store).
"""

from __future__ import annotations

import math
from typing import Optional

from .guard_types import GuardCheck, GuardResult
from .proposer import WeightProposal
from ..calibration import compute_wilson_lb as wilson_lower_bound

__all__ = ["AdaptationGuard", "GuardCheck", "GuardResult", "wilson_lower_bound"]


class AdaptationGuard:
    """Aplica los guardrails de FASE 8 a una WeightProposal."""

    def __init__(
        self,
        *,
        min_n: int = 10,
        min_history_days: int = 2,
        max_change: float = 0.10,
        min_weight: float = 0.05,
        wilson_z: float = 1.96,
        sum_tolerance: float = 1e-6,
    ) -> None:
        if min_n < 1:
            raise ValueError(f"min_n debe ser >= 1: {min_n}")
        if min_history_days < 1:
            raise ValueError(f"min_history_days debe ser >= 1: {min_history_days}")
        if not 0.0 < max_change <= 1.0:
            raise ValueError(f"max_change inválida: {max_change}")
        if not 0.0 < min_weight <= 0.5:
            raise ValueError(f"min_weight inválida: {min_weight}")
        if wilson_z <= 0.0:
            raise ValueError(f"wilson_z debe ser > 0: {wilson_z}")
        self.min_n = min_n
        self.min_history_days = min_history_days
        self.max_change = max_change
        self.min_weight = min_weight
        self.wilson_z = wilson_z
        self.sum_tolerance = sum_tolerance

    def evaluate(
        self,
        proposal: WeightProposal,
        *,
        history_days: int,
        data_quality: str = "clean",
        context_weights_after: Optional[dict[str, float]] = None,
    ) -> GuardResult:
        """Evalúa la propuesta; ``context_weights_after`` es el vector de
        pesos del contexto si se aplicara (para suma=1 y piso)."""
        checks: list[GuardCheck] = []

        # 1) min_n
        ok = proposal.sample_size >= self.min_n
        checks.append(
            GuardCheck(
                "min_n",
                ok,
                f"muestra {proposal.sample_size} >= {self.min_n}"
                if ok
                else f"muestra {proposal.sample_size} < {self.min_n}: no concluye",
            )
        )

        # 2) suficiente historial
        ok = history_days >= self.min_history_days
        checks.append(
            GuardCheck(
                "history",
                ok,
                f"{history_days} día(s) observado(s) >= {self.min_history_days}"
                if ok
                else f"{history_days} día(s) < {self.min_history_days}: historial insuficiente",
            )
        )

        # 3) solo outcomes reales (no INVALIDATED, no STALE)
        ok = data_quality == "clean"
        checks.append(
            GuardCheck(
                "clean_source",
                ok,
                "solo filas evaluadas (status=rewarded, sin STALE)"
                if ok
                else f"origen no limpio: {data_quality!r} — ZENIN nunca aprende "
                "de su propia predicción sin observar el outcome",
            )
        )

        # 4) reward válido
        ok = math.isfinite(proposal.observed_reward) and proposal.sample_size > 0
        checks.append(
            GuardCheck(
                "reward_valid",
                ok,
                f"reward {proposal.observed_reward:+.4f} finito, n={proposal.sample_size}"
                if ok
                else f"reward inválido: {proposal.observed_reward!r}",
            )
        )

        # 5) mejora estadísticamente razonable
        wilson = wilson_lower_bound(
            round(proposal.accuracy * proposal.sample_size),
            proposal.sample_size,
            self.wilson_z,
        )
        ok = wilson > 0.5
        checks.append(
            GuardCheck(
                "statistical",
                ok,
                f"Wilson lower bound {wilson:.3f} > 0.5 (acc {proposal.accuracy:.1%}, "
                f"n={proposal.sample_size})"
                if ok
                else f"Wilson lower bound {wilson:.3f} <= 0.5: sin evidencia "
                f"de mejora (acc {proposal.accuracy:.1%}, n={proposal.sample_size})",
            )
        )

        # 6) cambio máximo de peso
        delta = abs(proposal.weight_delta)
        ok = delta <= self.max_change + 1e-12
        checks.append(
            GuardCheck(
                "max_change",
                ok,
                f"|Δ| = {delta:.4f} <= {self.max_change}"
                if ok
                else f"|Δ| = {delta:.4f} > {self.max_change}: cambio excesivo",
            )
        )

        # 7) suma de pesos = 1 (contexto completo si se aplica)
        weights = context_weights_after or {}
        total = sum(weights.values())
        ok = bool(weights) and abs(total - 1.0) <= self.sum_tolerance
        checks.append(
            GuardCheck(
                "sum_weights",
                ok,
                f"suma {total:.6f} ≈ 1 ({len(weights)} expertos)"
                if ok
                else f"suma {total:.6f} ≠ 1: renomalización rota",
            )
        )

        # 8) ningún experto desaparece
        ok = bool(weights) and all(
            v >= self.min_weight - 1e-12 for v in weights.values()
        )
        checks.append(
            GuardCheck(
                "min_weight",
                ok,
                f"todo experto >= {self.min_weight}"
                if ok
                else f"algún experto < {self.min_weight}: no puede desaparecer",
            )
        )

        # 9) versión anterior preservada
        ok = proposal.parent_version is not None
        checks.append(
            GuardCheck(
                "parent_preserved",
                ok,
                f"parent_version={proposal.parent_version} (append-only)"
                if ok
                else "sin parent_version: el modelo anterior no se preservaría",
            )
        )

        return GuardResult(passed=all(c.ok for c in checks), checks=tuple(checks))