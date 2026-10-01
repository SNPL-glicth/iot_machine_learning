"""Métricas multidimensionales de separabilidad, preservación y degradación de evidencia.

Implementa el vector:
S = { S_stat, S_detector, S_event, S_temporal }

- S_stat: Cohen's d, Wasserstein distance, SNR, KS-test
- S_detector: ROC-AUC, PR-AUC, margen de separación (Peak - P95 normal)
- S_event: Cobertura de ventana, retención de amplitud pico, masa de evidencia
- S_temporal: Retraso de primer disparo (onset delay), delta respecto al baseline crudo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatisticalSeparability:
    """Métricas puras de distribución (detector-agnostic)."""

    cohens_d: float
    wasserstein_dist: float
    snr_db: float
    ks_statistic: float
    ks_pvalue: float


@dataclass(frozen=True)
class DetectorSeparability:
    """Capacidad discriminativa del detector sobre la representación."""

    roc_auc: float
    pr_auc: float
    peak_anomaly_score: float
    p95_normal_score: float
    p99_normal_score: float
    separation_margin_p95: float  # peak_anomaly - p95_normal
    separation_margin_p99: float  # peak_anomaly - p99_normal


@dataclass(frozen=True)
class EventPreservationMetrics:
    """Preservación intrínseca de una ventana anómala específica."""

    event_id: int
    detected: bool
    window_coverage_pct: float  # % de puntos en ventana que superan umbral
    peak_score: float
    peak_retention_ratio: float  # peak_transformed / peak_raw
    mean_window_score: float
    total_window_energy: float


@dataclass(frozen=True)
class TemporalPreservationMetrics:
    """Desplazamiento y retardo en la detección temporal del evento."""

    event_id: int
    detected: bool
    onset_idx_transformed: int | None
    onset_idx_raw_equiv: int | None
    delay_seconds: float | None
    delay_points_equiv: int | None
    delta_delay_vs_raw_sec: float | None  # delay_trans - delay_raw
    window_elapsed_pct: float | None  # % transcurrido de ventana al detectar


def compute_statistical_separability(
    normal_values: np.ndarray,
    anomaly_values: np.ndarray,
) -> StatisticalSeparability:
    """Calcula separabilidad estadística entre la distribución normal y anómala."""
    if len(normal_values) == 0 or len(anomaly_values) == 0:
        return StatisticalSeparability(0.0, 0.0, 0.0, 0.0, 1.0)

    # Limpiar posibles NaN/Inf
    norm_clean = normal_values[np.isfinite(normal_values)]
    ano_clean = anomaly_values[np.isfinite(anomaly_values)]

    mu_norm = float(np.mean(norm_clean))
    mu_ano = float(np.mean(ano_clean))
    var_norm = float(np.var(norm_clean, ddof=1)) if len(norm_clean) > 1 else 1e-9
    var_ano = float(np.var(ano_clean, ddof=1)) if len(ano_clean) > 1 else 1e-9

    pooled_std = float(np.sqrt(max((var_norm + var_ano) / 2.0, 1e-12)))
    cohens_d = abs(mu_ano - mu_norm) / pooled_std

    # Wasserstein distance
    w_dist = float(stats.wasserstein_distance(norm_clean, ano_clean))

    # SNR en dB: 10 * log10((mu_ano - mu_norm)^2 / var_norm)
    diff_sq = (mu_ano - mu_norm) ** 2
    snr_ratio = diff_sq / max(var_norm, 1e-12)
    snr_db = float(10.0 * np.log10(max(snr_ratio, 1e-12)))

    # Kolmogorov-Smirnov test
    ks_res = stats.ks_2samp(norm_clean, ano_clean)

    return StatisticalSeparability(
        cohens_d=round(cohens_d, 4),
        wasserstein_dist=round(w_dist, 4),
        snr_db=round(snr_db, 2),
        ks_statistic=round(float(ks_res.statistic), 4),
        ks_pvalue=float(ks_res.pvalue),
    )


def compute_detector_separability(
    scores: np.ndarray,
    binary_labels: np.ndarray,
) -> DetectorSeparability:
    """Calcula la separabilidad de los scores generados por el detector."""
    if len(scores) != len(binary_labels) or len(scores) == 0:
        return DetectorSeparability(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    scores_clean = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

    normal_scores = scores_clean[binary_labels == 0]
    anomaly_scores = scores_clean[binary_labels == 1]

    if len(anomaly_scores) == 0:
        return DetectorSeparability(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    p95_norm = float(np.percentile(normal_scores, 95)) if len(normal_scores) > 0 else 0.0
    p99_norm = float(np.percentile(normal_scores, 99)) if len(normal_scores) > 0 else 0.0
    peak_ano = float(np.max(anomaly_scores))

    # ROC-AUC y PR-AUC vía rank-sum / trapezoidal rule
    roc_auc = _compute_roc_auc(scores_clean, binary_labels)
    pr_auc = _compute_pr_auc(scores_clean, binary_labels)

    return DetectorSeparability(
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
        peak_anomaly_score=round(peak_ano, 4),
        p95_normal_score=round(p95_norm, 4),
        p99_normal_score=round(p99_norm, 4),
        separation_margin_p95=round(peak_ano - p95_norm, 4),
        separation_margin_p99=round(peak_ano - p99_norm, 4),
    )


def _compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Calcula ROC-AUC vía estadístico U de Mann-Whitney (vectorizado)."""
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = stats.rankdata(scores)
    pos_ranks = ranks[labels == 1]
    u_stat = float(np.sum(pos_ranks) - (n_pos * (n_pos + 1)) / 2.0)
    return float(u_stat / (n_pos * n_neg))


def _compute_pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Calcula PR-AUC por integración trapezoidal de precisión y recall."""
    n_pos = int(np.sum(labels == 1))
    if n_pos == 0:
        return 0.0

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp_cumsum = np.cumsum(sorted_labels == 1)
    fp_cumsum = np.cumsum(sorted_labels == 0)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / n_pos

    # Integración trapezoidal
    recalls_padded = np.concatenate(([0.0], recalls))
    precisions_padded = np.concatenate(([1.0], precisions))
    pr_auc = float(np.trapezoid(precisions_padded, recalls_padded))
    return max(0.0, min(1.0, pr_auc))
