"""Narrador de anomalías — capa Narrative.

Extraído de voting_anomaly_detector.py.
Responsabilidad ÚNICA: generar texto legible para humanos
a partir de votos de detección de anomalías.

No contiene cálculos matemáticos.
No contiene I/O ni sklearn.
"""

from __future__ import annotations

from typing import Dict, List


def build_anomaly_explanation(
    votes: Dict[str, float],
    z_score: float = 0.0,
) -> str:
    """Genera explicación legible a partir de votos individuales.

    Args:
        votes: Dict método → voto [0, 1].
        z_score: Z-score absoluto (para incluir en texto).

    Returns:
        Texto legible. Ej: "Z-score alto (3.2σ) + Fuera de rango IQR"
    """
    explanations: List[str] = []

    # Z-score
    z_vote = votes.get("z_score", 0.0)
    if z_vote >= 1.0:
        explanations.append(f"Z-score alto ({z_score:.1f}σ)")

    # IQR
    iqr_vote = votes.get("iqr", 0.0)
    if iqr_vote > 0.5:
        explanations.append("Fuera de rango IQR")

    # IsolationForest
    if_vote = votes.get("isolation_forest", 0.0)
    if if_vote > 0.5:
        explanations.append("Aislado globalmente (IF)")

    # LocalOutlierFactor
    lof_vote = votes.get("local_outlier_factor", 0.0)
    if lof_vote > 0.5:
        explanations.append("Outlier local (LOF)")

    return " + ".join(explanations) if explanations else "Valor normal"
