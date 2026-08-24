"""Walk-forward de ZENIN (FASE 9.1) — validación, no tuning.

La pregunta no es "¿ajusta bien?" sino "¿sobrevive fuera de muestra?":

    TRAIN       TEST
    ███████████ │ ██
                │
       TRAIN       TEST
       ███████████ │ ██
                   │
          TRAIN       TEST
          ███████████ │ ██

Reglas del harness (todas puras y auditables):
- ventanas contiguas y disjuntas (TRAIN ∩ TEST = ∅), origin rolling;
- el TRAIN alimenta la adaptación MoE (proposals + guardrail + versión)
  usando SOLO los outcomes conocidos dentro de ese TRAIN (since/until
  acotan el historial: nada del futuro entra);
- el TEST se evalúa con los pesos de la versión creada en su TRAIN;
- el régimen de cada ventana se etiqueta con ``classify_window`` sobre
  la cola del TRAIN (el estado que el modelo "ve" al entrar al TEST);
- el modelo compuesto es una media ponderada: por contexto y horizonte,
  reward_modelo = Σ w_experto × reward_experto (nada mágico: los pesos
  vienen de model_versions, la recompensa de outcomes reales).

Este módulo es puro (sin SQL, sin red): el runner en el script inyecta
el store real. Un resultado honesto (p. ej. ZENIN 51.8% vs EMA 51.4%)
es un resultado válido: la meta es evidencia, no victoria.
"""

from __future__ import annotations

from .windows import WfWindow, wf_windows, window_regime
from .wf_metrics import ModelMetrics, EdgeMetrics, weighted_model_metrics
from .eval import HorizonEval, WfRow, evaluate_window
from .render import render_wf_report

__all__ = [
    "WfWindow",
    "wf_windows",
    "window_regime",
    "ModelMetrics",
    "EdgeMetrics",
    "HorizonEval",
    "WfRow",
    "weighted_model_metrics",
    "evaluate_window",
    "render_wf_report",
]