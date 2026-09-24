"""Entidad de predicción de mercado (FASE 3).

Contract v1 — dominio puro, en memoria: no conoce proveedores, MySQL,
key-value stores ni Weaviate. La matemática y el ciclo temporal se validan aquí
antes de cualquier persistencia.

Responsabilidad única: la entidad ``Prediction`` y sus transiciones.
Los tipos de apoyo viven en ``types``, la validación numérica en
``validation`` y las reglas de estado en ``lifecycle``.
"""

from __future__ import annotations

from .prediction_entity import Prediction
from .prediction_guards import guard_outcome
from .prediction_transitions import (
    activate,
    archive,
    can_produce_reward,
    evaluate,
    invalidate,
    issue_reward,
    to_waiting_outcome,
)
from .types import InputContext, PredictionInterval, Regime

# Attach methods to Prediction class
Prediction.activate = activate
Prediction.to_waiting_outcome = to_waiting_outcome
Prediction.evaluate = evaluate
Prediction.issue_reward = issue_reward
Prediction.invalidate = invalidate
Prediction.archive = archive
Prediction.can_produce_reward = property(can_produce_reward)
Prediction._guard_outcome = staticmethod(lambda self, outcome: guard_outcome(self, outcome))

__all__ = ["Prediction", "Regime", "PredictionInterval", "InputContext"]
