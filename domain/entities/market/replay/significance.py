"""FASE 9.5 — Statistical Reality Check: ¿es la señal estadísticamente real?

FASE 9.4 dejó la pregunta abierta: el edge DECLARADO no sobrevive al
TEST, y el +0.03% de AMD (9.3) puede ser pura fluctuación. 9.5 somete
el pipeline de selección a pruebas de significancia, todas puras
(sin SQL, sin engine — solo datos persistidos):

    permutation test
        Barajar los OUTCOMES (movimientos firmados) entre timestamps,
        dejando las PREDICCIONES intactas. La dirección predicha se
        recupera de (direction_correct, move) y se re-evalúa contra el
        movimiento barajado. Bajo el nulo, el PnL esperado es
        exactamente 0 antes de costos (las distribuciones marginales
        se conservan, la asociación temporal predicción ↔ outcome se
        destruye). Si el edge real sigue apareciendo bajo el nulo →
        ruido/estructura del dataset, no capacidad predictiva.

    bootstrap por ventana (block bootstrap)
        Remuestreo de ventanas con reemplazo, IC 95% percentil para
        accuracy, net edge, sharpe y maxDD del agregado real. Responde
        "¿ese +0.03% es algo o fluctuación?".

    diferencia contra baselines
        ZENIN − (Naive | EMA) por ventana con IC 95% por bootstrap.
        Si el intervalo cruza cero, no hay superioridad demostrable.

    permutación del ganador (contexto → experto)
        Destruir SOLO la asociación contexto → experto ganador
        (ganador aleatorio por ventana, peso 1.0) y comprobar si el
        edge de la selección desaparece. FASE 9.3 dijo que la señal
        vive en la selección: esta prueba lo verifica.

    bootstrap por experto
        IC 95% para accuracy, mean_reward y ECE de cada estrategia
        sobre sus filas recompensadas (remuestreo de predicciones).
"""

from __future__ import annotations

from .permutation import (
    PermWindow,
    PermutationResult,
    recover_predicted_direction,
    permutation_test,
)
from .bootstrap import (
    WindowRecord,
    BootstrapCi,
    weighted_acc,
    weighted_net,
    pooled_sharpe,
    window_cumsum_maxdd,
    block_bootstrap,
    difference_ci,
)
from .random_winner import RandomWinnerResult, random_winner_test
from .expert_metrics import ExpertMetricsCi, bootstrap_expert_metrics

__all__ = [
    "PermWindow",
    "PermutationResult",
    "recover_predicted_direction",
    "permutation_test",
    "WindowRecord",
    "BootstrapCi",
    "weighted_acc",
    "weighted_net",
    "pooled_sharpe",
    "window_cumsum_maxdd",
    "block_bootstrap",
    "difference_ci",
    "RandomWinnerResult",
    "random_winner_test",
    "ExpertMetricsCi",
    "bootstrap_expert_metrics",
]