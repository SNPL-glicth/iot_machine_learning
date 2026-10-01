"""Despachador asimétrico para el Mixture of Experts por afinidad de escala.

SRP: Selecciona y ejecuta exclusivamente los expertos cuya afinidad coincide
con el nivel de representación activo (10X, 2X, RAW).
No contiene condicionales algorítmicos acoplados (cero 'if expert == ...').
"""

from __future__ import annotations

from typing import Any, Sequence

from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    RepresentationLevel,
)
from iot_machine_learning.domain.ports.asymmetric_expert_port import (
    AsymmetricExpertPort,
)


class AsymmetricDispatcher:
    """Enrutador dinámico que despacha slices a expertos según afinidad de escala."""

    def __init__(
        self,
        experts: Sequence[AsymmetricExpertPort] | None = None,
    ) -> None:
        self._experts_by_level: dict[RepresentationLevel, list[AsymmetricExpertPort]] = {
            level: [] for level in RepresentationLevel
        }
        self._all_experts: dict[str, AsymmetricExpertPort] = {}

        # Telemetría de cómputo y ejecuciones
        self._dispatches_by_level: dict[RepresentationLevel, int] = {
            level: 0 for level in RepresentationLevel
        }
        self._evaluations_by_expert: dict[str, int] = {}
        self._total_cost_expended: float = 0.0
        self._total_cost_hypothetical_full: float = 0.0

        if experts:
            for expert in experts:
                self.register_expert(expert)

    def register_expert(self, expert: AsymmetricExpertPort) -> None:
        """Registra un experto en el catálogo asimétrico verificando su contrato."""
        if not hasattr(expert, "name") or not hasattr(expert, "affinity"):
            raise TypeError(
                f"El experto {expert} no cumple con los atributos requeridos por AsymmetricExpertPort"
            )
        if not callable(getattr(expert, "evaluate", None)):
            raise TypeError(f"El experto {expert} debe implementar el método evaluate()")

        self._experts_by_level[expert.affinity].append(expert)
        self._all_experts[expert.name] = expert
        self._evaluations_by_expert[expert.name] = 0

    @property
    def registered_experts(self) -> tuple[AsymmetricExpertPort, ...]:
        """Tupla inmutable de todos los expertos registrados."""
        return tuple(self._all_experts.values())

    def get_experts_for_level(
        self, level: RepresentationLevel
    ) -> tuple[AsymmetricExpertPort, ...]:
        """Obtiene los expertos registrados para un nivel de representación específico."""
        return tuple(self._experts_by_level.get(level, []))

    def dispatch(
        self,
        level: RepresentationLevel,
        series_slice: Sequence[float],
    ) -> list[EvidenceScore]:
        """Despacha el slice únicamente a los expertos con afinidad al nivel activo.

        Args:
            level: Nivel de resolución temporal activo (10X, 2X, RAW).
            series_slice: Fragmento de la serie muestreado a la escala activa.

        Returns:
            Lista de EvidenceScore emitidos por los expertos afines.
        """
        if not series_slice:
            return []

        # Encontrar expertos afines comparando value del enum
        lvl_val = level.value if hasattr(level, "value") else str(level)
        active_experts = [
            e for e in self._all_experts.values()
            if (e.affinity.value if hasattr(e.affinity, "value") else str(e.affinity)) == lvl_val
        ]
        if not active_experts:
            return []

        # Registrar despacho
        for k in self._dispatches_by_level:
            if k.value == lvl_val:
                self._dispatches_by_level[k] += 1
                break

        # Cómputo hipotético si corriéramos TODOS los expertos
        full_catalog_cost = sum(
            getattr(e, "compute_cost_estimate", 1.0)
            for e in self._all_experts.values()
        )
        self._total_cost_hypothetical_full += full_catalog_cost

        scores: list[EvidenceScore] = []
        for expert in active_experts:
            score = expert.evaluate(series_slice)
            if not isinstance(score, EvidenceScore):
                raise TypeError(
                    f"El experto {expert.name} retornó {type(score)}, se esperaba EvidenceScore"
                )
            score_val = (
                score.representation_affinity.value
                if hasattr(score.representation_affinity, "value")
                else str(score.representation_affinity)
            )
            if score_val != lvl_val:
                raise ValueError(
                    f"Inconsistencia de contrato: experto {expert.name} con afinidad "
                    f"{expert.affinity} emitió score para nivel {score.representation_affinity}"
                )

            scores.append(score)
            self._evaluations_by_expert[expert.name] += 1
            self._total_cost_expended += score.compute_cost_estimate

        return scores

    def get_compute_savings_ratio(self) -> float:
        """Calcula la fracción de costo computacional ahorrada respecto a ejecución plena."""
        if self._total_cost_hypothetical_full <= 1e-9:
            return 0.0
        saved = self._total_cost_hypothetical_full - self._total_cost_expended
        return max(0.0, saved / self._total_cost_hypothetical_full)

    def get_telemetry(self) -> dict[str, Any]:
        """Retorna telemetría consolidada de despacho y cómputo."""
        return {
            "dispatches_by_level": {k.value: v for k, v in self._dispatches_by_level.items()},
            "evaluations_by_expert": dict(self._evaluations_by_expert),
            "total_cost_expended": round(self._total_cost_expended, 4),
            "total_cost_hypothetical_full": round(self._total_cost_hypothetical_full, 4),
            "compute_savings_ratio": round(self.get_compute_savings_ratio(), 4),
        }
