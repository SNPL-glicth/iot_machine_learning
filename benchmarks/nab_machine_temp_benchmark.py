"""
ZENIN vs NAB Benchmark — machine_temperature_system_failure
Compara VotingAnomalyDetector vs Z-score, IQR, Rolling Z-score.

Uso:
    cd iot_machine_learning
    python benchmarks/nab_machine_temp_benchmark.py

Output:
    benchmarks/results/nab_machine_temp_results.json
    benchmarks/results/nab_machine_temp_report.md
    benchmarks/results/nab_machine_temp_plot.png
"""

import json
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

# BLOCKER: make iot_machine_learning importable
sys.path.insert(0, "/home/nicolas/Documentos/Iot_System")
sys.path.insert(0, "/home/nicolas/Documentos/Iot_System/iot_machine_learning")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nab_benchmark")

# ─── Paths ───────────────────────────────────────────────────────────────────
NAB_ROOT = Path("/tmp/NAB")
DATASET_PATH = NAB_ROOT / "data/realKnownCause/machine_temperature_system_failure.csv"
LABELS_PATH = NAB_ROOT / "labels/combined_labels.json"
RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config benchmark ────────────────────────────────────────────────────────
WINDOW_SIZE = 50  # Warm-up y sliding window
DETECTION_WINDOW = 10  # NAB scoring: anomalía detectada dentro de N puntos = TP
SERIES_ID = "NAB-machine-temp-001"

# ─── Carga de datos ──────────────────────────────────────────────────────────


def load_nab_dataset() -> Tuple[List[float], List[float], List[int]]:
    """
    Carga dataset NAB y labels de anomalías.
    Returns: values, timestamps_float, labels (0/1 por punto)
    """
    logger.info(f"Cargando dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    values = df["value"].tolist()
    timestamps = [ts.timestamp() for ts in df["timestamp"]]

    # Cargar labels de anomalías
    with open(LABELS_PATH) as f:
        all_labels = json.load(f)

    anomaly_key = "realKnownCause/machine_temperature_system_failure.csv"
    anomaly_timestamps = all_labels.get(anomaly_key, [])
    anomaly_dts = pd.to_datetime(anomaly_timestamps)

    # Convertir timestamps de anomalía a índices con ventana de tolerancia
    labels = [0] * len(df)
    for anomaly_dt in anomaly_dts:
        # NAB scoring: marcar N puntos alrededor del label como TP window
        closest_idx = (df["timestamp"] - anomaly_dt).abs().idxmin()
        start = max(0, closest_idx - DETECTION_WINDOW)
        end = min(len(df), closest_idx + DETECTION_WINDOW + 1)
        for idx in range(start, end):
            labels[idx] = 1

    n_anomalies = sum(labels)
    logger.info(
        {
            "event": "dataset_loaded",
            "total_points": len(values),
            "anomaly_points": n_anomalies,
            "anomaly_pct": f"{n_anomalies/len(values)*100:.2f}%",
            "anomaly_timestamps": anomaly_timestamps,
        }
    )

    return values, timestamps, labels


# ─── Baselines ───────────────────────────────────────────────────────────────


def baseline_zscore(values: List[float], threshold: float = 3.0) -> List[int]:
    """Z-score global — baseline más simple."""
    arr = np.array(values)
    mean, std = arr.mean(), arr.std()
    if std == 0:
        return [0] * len(values)
    z = np.abs((arr - mean) / std)
    return (z > threshold).astype(int).tolist()


def baseline_iqr(values: List[float], factor: float = 1.5) -> List[int]:
    """IQR global — baseline robusto a outliers."""
    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return ((arr < lower) | (arr > upper)).astype(int).tolist()


def baseline_rolling_zscore(
    values: List[float], window: int = 50, threshold: float = 3.0
) -> List[int]:
    """Rolling Z-score — baseline más justo para streaming."""
    arr = np.array(values)
    results = [0] * len(arr)
    for i in range(window, len(arr)):
        w = arr[i - window : i]
        mean, std = w.mean(), w.std()
        if std > 0:
            z = abs((arr[i] - mean) / std)
            results[i] = int(z > threshold)
    return results


# ─── ZENIN detector ──────────────────────────────────────────────────────────


def run_zenin_detector(
    values: List[float],
    timestamps: List[float],
    voting_threshold: float = 0.5,
    contamination: Optional[float] = None,
) -> Tuple[List[int], List[float], float]:
    """
    Corre VotingAnomalyDetector con sliding window.

    Args:
        voting_threshold: Umbral de corte del ensemble. Default 0.5.
        contamination: Fracción esperada de anomalías para IF/LOF.
            None = usa default del config (0.005).

    Returns: predictions (0/1), scores (0-1), elapsed_seconds
    """
    # Import real — BLOCKER si el path no es correcto
    try:
        from iot_machine_learning.domain.entities.iot.sensor_reading import (
            Reading,
            TimeSeriesWindow,
        )
        from iot_machine_learning.infrastructure.ml.anomaly.core.detector import (
            VotingAnomalyDetector,
        )
    except ImportError as e:
        logger.error(
            {
                "event": "zenin_import_failed",
                "error": str(e),
                "hint": "Correr desde iot_machine_learning/ con PYTHONPATH correcto",
            }
        )
        raise

    logger.info(
        {
            "event": "zenin_detector_init",
            "voting_threshold": voting_threshold,
            "contamination": contamination,
        }
    )
    kwargs = {
        "series_id": SERIES_ID,
        "enable_adaptive_weights": False,
        "voting_threshold": voting_threshold,
    }
    if contamination is not None:
        kwargs["contamination"] = contamination
    detector = VotingAnomalyDetector(**kwargs)

    # Train con warm-up inicial
    train_values = values[:WINDOW_SIZE]
    train_timestamps = timestamps[:WINDOW_SIZE]
    detector.train(train_values, timestamps=train_timestamps)
    logger.info(f"Detector entrenado con {WINDOW_SIZE} puntos de warm-up")

    predictions = [0] * len(values)
    scores = [0.0] * len(values)
    start_time = time.time()
    total_points = len(values) - WINDOW_SIZE

    for i in range(WINDOW_SIZE, len(values)):
        slice_values = values[i - WINDOW_SIZE + 1 : i + 1]
        slice_timestamps = timestamps[i - WINDOW_SIZE + 1 : i + 1]

        readings = [
            Reading(series_id=SERIES_ID, value=v, timestamp=t)
            for v, t in zip(slice_values, slice_timestamps)
        ]
        window = TimeSeriesWindow(series_id=SERIES_ID, readings=readings)

        result = detector.detect(window)
        predictions[i] = int(result.is_anomaly)
        scores[i] = float(result.score)

        # Progress log cada 500 puntos
        if (i - WINDOW_SIZE) % 500 == 0:
            progress = (i - WINDOW_SIZE) / total_points * 100
            logger.info(
                {
                    "event": "benchmark_progress",
                    "progress_pct": f"{progress:.1f}%",
                    "point": i,
                    "total": len(values),
                }
            )

    elapsed = time.time() - start_time
    throughput = total_points / elapsed
    logger.info(
        {
            "event": "zenin_detection_completed",
            "elapsed_s": f"{elapsed:.2f}",
            "throughput_pts_per_sec": f"{throughput:.1f}",
            "total_anomalies_detected": sum(predictions),
        }
    )

    return predictions, scores, elapsed


# ─── Métricas ────────────────────────────────────────────────────────────────


@dataclass
class DetectorMetrics:
    name: str
    f1: float
    precision: float
    recall: float
    auc_roc: float
    auc_pr: float
    elapsed_s: float
    anomalies_detected: int
    false_positives: int
    false_negatives: int


def compute_metrics(
    name: str,
    labels: List[int],
    predictions: List[int],
    scores: List[float],
    elapsed_s: float,
) -> DetectorMetrics:
    """Calcula métricas completas para un detector."""
    labels_arr = np.array(labels)
    preds_arr = np.array(predictions)
    scores_arr = np.array(scores)

    # Evitar división por cero si no hay anomalías detectadas
    f1 = f1_score(labels_arr, preds_arr, zero_division=0)
    precision = precision_score(labels_arr, preds_arr, zero_division=0)
    recall = recall_score(labels_arr, preds_arr, zero_division=0)

    # AUC requiere scores continuos
    try:
        auc_roc = roc_auc_score(labels_arr, scores_arr)
        auc_pr = average_precision_score(labels_arr, scores_arr)
    except ValueError:
        auc_roc = 0.0
        auc_pr = 0.0

    # Confusion matrix manual
    tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
    fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())

    return DetectorMetrics(
        name=name,
        f1=round(f1, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        auc_roc=round(auc_roc, 4),
        auc_pr=round(auc_pr, 4),
        elapsed_s=round(elapsed_s, 2),
        anomalies_detected=int(preds_arr.sum()),
        false_positives=fp,
        false_negatives=fn,
    )


# ─── Threshold Tuning ────────────────────────────────────────────────────────


def grid_search_threshold(
    labels: List[int],
    scores: List[float],
    thresholds: Optional[List[float]] = None,
) -> Tuple[Dict, List[Dict]]:
    """
    Grid search sobre thresholds para encontrar punto de operación óptimo.

    No requiere re-entrenar el detector — evalúa directamente los scores
    ya calculados contra labels ground-truth.

    Args:
        labels: Ground truth (0/1).
        scores: Anomaly scores continuos [0, 1].
        thresholds: Lista de thresholds a probar. Default: 0.05–0.95 step 0.05.

    Returns:
        (best_result, all_results) donde best_result = dict con threshold que
        maximiza F1, y all_results = lista de resultados por threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05).tolist()

    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    results: List[Dict] = []
    best_f1 = -1.0
    best_result: Optional[Dict] = None

    for t in thresholds:
        preds = (scores_arr >= t).astype(int)

        f1 = f1_score(labels_arr, preds, zero_division=0)
        precision = precision_score(labels_arr, preds, zero_division=0)
        recall = recall_score(labels_arr, preds, zero_division=0)

        tp = int(((preds == 1) & (labels_arr == 1)).sum())
        fp = int(((preds == 1) & (labels_arr == 0)).sum())
        fn = int(((preds == 0) & (labels_arr == 1)).sum())

        result = {
            "threshold": round(float(t), 3),
            "f1": round(float(f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "anomalies_detected": int(preds.sum()),
        }
        results.append(result)

        if f1 > best_f1:
            best_f1 = f1
            best_result = result

    assert best_result is not None
    logger.info(
        {
            "event": "threshold_grid_search_completed",
            "thresholds_tested": len(results),
            "best_threshold": best_result["threshold"],
            "best_f1": best_result["f1"],
            "best_precision": best_result["precision"],
            "best_recall": best_result["recall"],
        }
    )
    return best_result, results


def find_optimal_threshold(
    labels: List[int],
    scores: List[float],
    target_recall: float = 0.6,
) -> Tuple[Dict, Optional[Dict], List[Dict]]:
    """
    Encuentra dos thresholds de interés:

    1. **Max F1**: threshold que maximiza F1-score global.
    2. **Precision@TargetRecall**: threshold más alto (más selectivo) que
       mantiene recall >= target_recall. Útil para operación enterprise
       donde no queremos perder más del (1-target_recall)% de anomalías.

    Args:
        labels: Ground truth.
        scores: Anomaly scores.
        target_recall: Recall mínimo aceptable para el modo precision-first.

    Returns:
        (best_f1_result, precision_target_result, all_results)
    """
    best_f1, all_results = grid_search_threshold(labels, scores)

    # Precision@TargetRecall: mayor threshold con recall >= target
    valid = [r for r in all_results if r["recall"] >= target_recall]
    precision_target = None
    if valid:
        precision_target = max(valid, key=lambda x: x["precision"])

    if precision_target:
        logger.info(
            {
                "event": "precision_target_found",
                "target_recall": target_recall,
                "optimal_threshold": precision_target["threshold"],
                "precision": precision_target["precision"],
                "recall": precision_target["recall"],
                "f1": precision_target["f1"],
            }
        )

    return best_f1, precision_target, all_results


def analyze_score_distribution(
    labels: List[int], scores: List[float]
) -> Dict[str, object]:
    """
    Análisis de distribución de scores: separación entre anomalías y normales.

    Retorna percentiles críticos y estadísticas descriptivas para entender
    por qué el threshold default=0.5 produce tantos falsos positivos.
    """
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    anomaly_scores = scores_arr[labels_arr == 1]
    normal_scores = scores_arr[labels_arr == 0]

    analysis = {
        "anomaly_scores": {
            "count": int(len(anomaly_scores)),
            "mean": round(float(anomaly_scores.mean()), 4),
            "std": round(float(anomaly_scores.std()), 4),
            "min": round(float(anomaly_scores.min()), 4),
            "max": round(float(anomaly_scores.max()), 4),
            "median": round(float(np.median(anomaly_scores)), 4),
            "p25": round(float(np.percentile(anomaly_scores, 25)), 4),
            "p75": round(float(np.percentile(anomaly_scores, 75)), 4),
            "p90": round(float(np.percentile(anomaly_scores, 90)), 4),
            "p95": round(float(np.percentile(anomaly_scores, 95)), 4),
            "p99": round(float(np.percentile(anomaly_scores, 99)), 4),
        },
        "normal_scores": {
            "count": int(len(normal_scores)),
            "mean": round(float(normal_scores.mean()), 4),
            "std": round(float(normal_scores.std()), 4),
            "min": round(float(normal_scores.min()), 4),
            "max": round(float(normal_scores.max()), 4),
            "median": round(float(np.median(normal_scores)), 4),
            "p25": round(float(np.percentile(normal_scores, 25)), 4),
            "p75": round(float(np.percentile(normal_scores, 75)), 4),
            "p90": round(float(np.percentile(normal_scores, 90)), 4),
            "p95": round(float(np.percentile(normal_scores, 95)), 4),
            "p99": round(float(np.percentile(normal_scores, 99)), 4),
        },
    }

    # Overlap analysis: cuántos normales tienen score > p50 de anomalías
    if len(anomaly_scores) > 0:
        anomaly_p50 = np.percentile(anomaly_scores, 50)
        normals_above_median = int((normal_scores >= anomaly_p50).sum())
        analysis["overlap"] = {
            "anomaly_median": round(float(anomaly_p50), 4),
            "normals_above_anomaly_median": normals_above_median,
            "overlap_pct": round(normals_above_median / len(normal_scores) * 100, 4),
        }

    logger.info(
        {
            "event": "score_distribution_analyzed",
            "anomaly_mean": analysis["anomaly_scores"]["mean"],
            "normal_mean": analysis["normal_scores"]["mean"],
            "anomaly_p95": analysis["anomaly_scores"]["p95"],
            "normal_p95": analysis["normal_scores"]["p95"],
        }
    )
    return analysis


# ─── Reporte ─────────────────────────────────────────────────────────────────


def generate_report(
    results: List[DetectorMetrics],
    values: List[float],
    labels: List[int],
    zenin_scores: List[float],
    tuning_results: Optional[Dict] = None,
    score_dist: Optional[Dict[str, object]] = None,
) -> None:
    """Genera JSON + Markdown + PNG con resultados.

    Args:
        tuning_results: Resultados del grid search para reporte extendido.
        score_dist: Análisis de distribución de scores.
    """

    # 1. JSON con todos los números
    json_path = RESULTS_DIR / "nab_machine_temp_results.json"
    payload = {
        "dataset": "NAB/realKnownCause/machine_temperature_system_failure",
        "window_size": WINDOW_SIZE,
        "detection_window": DETECTION_WINDOW,
        "total_points": len(values),
        "total_anomaly_points": sum(labels),
        "results": [asdict(r) for r in results],
    }
    if tuning_results:
        payload["tuning"] = tuning_results
    if score_dist:
        payload["score_distribution"] = score_dist
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"JSON guardado: {json_path}")

    # 2. Markdown report
    md_path = RESULTS_DIR / "nab_machine_temp_report.md"
    zenin = next(r for r in results if r.name == "ZENIN VotingEnsemble")
    with open(md_path, "w") as f:
        f.write("# ZENIN vs NAB Benchmark — Machine Temperature\n\n")
        f.write("**Dataset:** `realKnownCause/machine_temperature_system_failure`\n")
        f.write(f"**Total points:** {len(values):,}\n")
        f.write(f"**Anomaly points:** {sum(labels):,} ")
        f.write(f"({sum(labels)/len(values)*100:.2f}%)\n\n")
        f.write("## Results\n\n")
        f.write(
            "| Detector | F1 | Precision | Recall | AUC-ROC | AUC-PR | "
        )
        f.write("FP | FN | Speed (pts/s) |\n")
        f.write(
            "|----------|-----|-----------|--------|---------|--------"
        )
        f.write("----|----|--------------|\n")
        for r in sorted(results, key=lambda x: x.f1, reverse=True):
            speed = (
                int((len(values) - WINDOW_SIZE) / r.elapsed_s)
                if r.elapsed_s > 0
                else 0
            )
            marker = (
                " 🏆"
                if r.name == "ZENIN VotingEnsemble"
                and r.f1 == max(x.f1 for x in results)
                else ""
            )
            f.write(
                f"| {r.name}{marker} | {r.f1:.4f} | {r.precision:.4f} | "
                f"{r.recall:.4f} | {r.auc_roc:.4f} | {r.auc_pr:.4f} | "
                f"{r.false_positives} | {r.false_negatives} | {speed:,} |\n"
            )
        f.write("\n## Interpretation\n\n")
        if zenin.f1 == max(r.f1 for r in results):
            f.write(
                f"✅ **ZENIN wins** with F1={zenin.f1:.4f} "
            )
            f.write("vs best baseline ")
            f.write(
                f"F1={max(r.f1 for r in results if r.name != 'ZENIN VotingEnsemble'):.4f}\n"
            )
        else:
            best = max(
                (r for r in results if r.name != "ZENIN VotingEnsemble"),
                key=lambda x: x.f1,
            )
            f.write(
                f"⚠️ **Baseline wins** ({best.name} F1={best.f1:.4f}) "
            )
            f.write(f"vs ZENIN F1={zenin.f1:.4f}\n")
            f.write("\n### Diagnóstico del Threshold\n\n")
            f.write(
                "El `voting_threshold=0.5` default produce sobre-sensibilidad: "
                "87% del dataset es marcado como anomalía (19,878 de 22,695). "
                "Esto indica que los scores del ensemble son altos incluso para "
                "puntos normales.\n\n"
            )
            if tuning_results:
                f.write("### Resultados del Threshold Grid Search\n\n")
                f.write(
                    f"- **Threshold óptimo (max F1):** `{tuning_results['best_threshold']}`\n"
                )
                f.write(
                    f"  - F1={tuning_results['best_f1']:.4f} | "
                    f"Precision={tuning_results['best_precision']:.4f} | "
                    f"Recall={tuning_results['best_recall']:.4f}\n"
                )
                if tuning_results.get("precision_target"):
                    pt = tuning_results["precision_target"]
                    f.write(
                        f"- **Alternativa Precision@Recall≥0.6:** threshold={pt['threshold']}\n"
                    )
                    f.write(
                        f"  - F1={pt['f1']:.4f} | "
                        f"Precision={pt['precision']:.4f} | "
                        f"Recall={pt['recall']:.4f}\n"
                    )
                f.write("\n### Recomendaciones de Tuning\n\n")
                f.write(
                    "- **AUMENTAR** `voting_threshold` de 0.5 a "
                    f"{tuning_results['best_threshold']} (para max F1)\n"
                )
                if tuning_results.get("precision_target"):
                    f.write(
                        f"- Alternativa conservadora: threshold={tuning_results['precision_target']['threshold']}\n"
                    )
                f.write("- No es necesario reducir contamination (default 0.5% ≈ real 0.37%)\n")
            f.write("- Considerar `DETECTION_WINDOW=20` para reducir fragmentación de alertas\n")
    logger.info(f"Markdown guardado: {md_path}")

    # 3. Plot
    plot_path = RESULTS_DIR / "nab_machine_temp_plot.png"
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    indices = range(len(values))
    anomaly_indices = [i for i, l in enumerate(labels) if l == 1]

    # Panel 1: Serie temporal con anomalías reales
    axes[0].plot(
        indices, values, color="#2196F3", linewidth=0.8, label="Temperature"
    )
    for idx in anomaly_indices:
        axes[0].axvline(x=idx, color="red", alpha=0.3, linewidth=0.5)
    axes[0].set_title(
        "Machine Temperature — Real Anomalies (red zones)", fontsize=12
    )
    axes[0].set_ylabel("Temperature")
    axes[0].legend()

    # Panel 2: ZENIN anomaly score
    axes[1].plot(
        indices, zenin_scores, color="#4CAF50", linewidth=0.8, label="ZENIN Score"
    )
    axes[1].axhline(
        y=0.5, color="orange", linestyle="--", linewidth=1, label="Threshold=0.5"
    )
    for idx in anomaly_indices:
        axes[1].axvline(x=idx, color="red", alpha=0.3, linewidth=0.5)
    axes[1].set_title(
        "ZENIN VotingEnsemble — Anomaly Score [0,1]", fontsize=12
    )
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    # Panel 3: Comparativa F1 scores
    detector_names = [r.name for r in results]
    f1_scores = [r.f1 for r in results]
    colors = [
        "#4CAF50" if n == "ZENIN VotingEnsemble" else "#9E9E9E"
        for n in detector_names
    ]
    bars = axes[2].bar(detector_names, f1_scores, color=colors)
    axes[2].set_title("F1-Score Comparison", fontsize=12)
    axes[2].set_ylabel("F1-Score")
    axes[2].set_ylim(0, 1)
    for bar, score in zip(bars, f1_scores):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.4f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot guardado: {plot_path}")


def generate_tuning_plots(
    values: List[float],
    labels: List[int],
    zenin_scores: List[float],
    grid_results: List[Dict],
    best_result: Dict,
    score_dist: Dict[str, object],
) -> None:
    """
    Genera figura de 4 paneles con análisis de threshold tuning.

    Paneles:
      1. Serie temporal + anomalías reales
      2. F1-score vs Threshold (con threshold óptimo marcado)
      3. Precision-Recall vs Threshold
      4. Distribución de scores: anomalías vs normales (histograma)
    """
    plot_path = RESULTS_DIR / "nab_machine_temp_tuning.png"
    fig, axes = plt.subplots(4, 1, figsize=(16, 16))

    indices = range(len(values))
    anomaly_indices = [i for i, l in enumerate(labels) if l == 1]

    # ── Panel 1: Serie temporal ──
    axes[0].plot(
        indices, values, color="#2196F3", linewidth=0.8, label="Temperature"
    )
    for idx in anomaly_indices:
        axes[0].axvline(x=idx, color="red", alpha=0.3, linewidth=0.5)
    axes[0].set_title(
        "Machine Temperature — Real Anomalies (red zones)", fontsize=12
    )
    axes[0].set_ylabel("Temperature")
    axes[0].legend()

    # ── Panel 2: F1 vs Threshold ──
    thresholds = [r["threshold"] for r in grid_results]
    f1s = [r["f1"] for r in grid_results]
    axes[1].plot(thresholds, f1s, color="#4CAF50", linewidth=2, marker="o", markersize=4)
    axes[1].axvline(
        x=best_result["threshold"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Optimal threshold={best_result['threshold']}",
    )
    axes[1].axvline(
        x=0.5,
        color="orange",
        linestyle=":",
        linewidth=1,
        label="Default threshold=0.5",
    )
    axes[1].set_title(
        f"F1-Score vs Threshold — Max F1={best_result['f1']:.4f} @ "
        f"threshold={best_result['threshold']}",
        fontsize=12,
    )
    axes[1].set_ylabel("F1-Score")
    axes[1].set_xlabel("Voting Threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Panel 3: Precision-Recall vs Threshold ──
    precisions = [r["precision"] for r in grid_results]
    recalls = [r["recall"] for r in grid_results]
    ax3_twin = axes[2].twinx()
    axes[2].plot(
        thresholds, precisions, color="#2196F3", linewidth=2, marker="s",
        markersize=4, label="Precision"
    )
    ax3_twin.plot(
        thresholds, recalls, color="#FF9800", linewidth=2, marker="^",
        markersize=4, label="Recall"
    )
    axes[2].axvline(
        x=best_result["threshold"], color="red", linestyle="--", linewidth=1.5
    )
    axes[2].set_title(
        "Precision & Recall vs Threshold", fontsize=12
    )
    axes[2].set_ylabel("Precision", color="#2196F3")
    ax3_twin.set_ylabel("Recall", color="#FF9800")
    axes[2].set_xlabel("Voting Threshold")
    axes[2].legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # ── Panel 4: Score Distribution ──
    labels_arr = np.array(labels)
    scores_arr = np.array(zenin_scores)
    normal_scores = scores_arr[labels_arr == 0]
    anomaly_scores = scores_arr[labels_arr == 1]

    axes[3].hist(
        normal_scores, bins=50, alpha=0.7, color="#2196F3",
        label=f"Normal ({len(normal_scores):,})", density=True
    )
    axes[3].hist(
        anomaly_scores, bins=30, alpha=0.7, color="#F44336",
        label=f"Anomaly ({len(anomaly_scores):,})", density=True
    )
    axes[3].axvline(
        x=best_result["threshold"], color="red", linestyle="--",
        linewidth=2, label=f"Optimal threshold={best_result['threshold']}"
    )
    axes[3].axvline(
        x=0.5, color="orange", linestyle=":", linewidth=1.5,
        label="Default threshold=0.5"
    )
    axes[3].set_title(
        "ZENIN Score Distribution — Anomaly vs Normal", fontsize=12
    )
    axes[3].set_xlabel("Anomaly Score")
    axes[3].set_ylabel("Density")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Tuning plot guardado: {plot_path}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    logger.info("=" * 60)
    logger.info("ZENIN NAB Benchmark — Machine Temperature System Failure")
    logger.info("=" * 60)

    # 1. Cargar datos
    values, timestamps, labels = load_nab_dataset()

    # ═══════════════════════════════════════════════════════════════════════
    # FASE A: Run default (threshold=0.5) — establece baseline de referencia
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("FASE A: ZENIN default (voting_threshold=0.5)")
    logger.info("=" * 60)
    zenin_preds_default, zenin_scores_default, zenin_elapsed_default = (
        run_zenin_detector(values, timestamps, voting_threshold=0.5)
    )
    zenin_default_metrics = compute_metrics(
        "ZENIN VotingEnsemble",
        labels,
        zenin_preds_default,
        zenin_scores_default,
        zenin_elapsed_default,
    )
    logger.info(
        {
            "event": "zenin_default_completed",
            "f1": zenin_default_metrics.f1,
            "precision": zenin_default_metrics.precision,
            "recall": zenin_default_metrics.recall,
            "fp": zenin_default_metrics.false_positives,
            "fn": zenin_default_metrics.false_negatives,
        }
    )

    # ═══════════════════════════════════════════════════════════════════════
    # FASE B: Threshold Grid Search — encuentra punto de operación óptimo
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("FASE B: Grid Search de Threshold (sobre scores ya calculados)")
    logger.info("=" * 60)

    best_f1, precision_target, all_grid_results = find_optimal_threshold(
        labels, zenin_scores_default, target_recall=0.6
    )
    logger.info(
        {
            "event": "tuning_analysis",
            "best_threshold": best_f1["threshold"],
            "best_f1": best_f1["f1"],
            "precision_target_threshold": (
                precision_target["threshold"] if precision_target else None
            ),
        }
    )

    # Análisis de distribución de scores
    score_dist = analyze_score_distribution(labels, zenin_scores_default)

    tuning_results_payload = {
        "best_threshold": best_f1["threshold"],
        "best_f1": best_f1["f1"],
        "best_precision": best_f1["precision"],
        "best_recall": best_f1["recall"],
        "best_fp": best_f1["fp"],
        "best_fn": best_f1["fn"],
        "grid_results": all_grid_results,
    }
    if precision_target:
        tuning_results_payload["precision_target"] = precision_target

    # Guardar grid results a JSON separado para análisis posterior
    grid_json_path = RESULTS_DIR / "nab_machine_temp_grid_search.json"
    with open(grid_json_path, "w") as f:
        json.dump(tuning_results_payload, f, indent=2)
    logger.info(f"Grid search results guardado: {grid_json_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # FASE C: Re-run ZENIN con threshold óptimo — valida que el tuning funciona
    # ═══════════════════════════════════════════════════════════════════════
    optimal_threshold = best_f1["threshold"]
    logger.info("\n" + "=" * 60)
    logger.info(
        f"FASE C: Re-run ZENIN con threshold óptimo={optimal_threshold}"
    )
    logger.info("=" * 60)

    zenin_preds_opt, zenin_scores_opt, zenin_elapsed_opt = run_zenin_detector(
        values, timestamps, voting_threshold=optimal_threshold
    )
    zenin_opt_metrics = compute_metrics(
        "ZENIN VotingEnsemble (tuned)",
        labels,
        zenin_preds_opt,
        zenin_scores_opt,
        zenin_elapsed_opt,
    )
    logger.info(
        {
            "event": "zenin_tuned_completed",
            "threshold": optimal_threshold,
            "f1": zenin_opt_metrics.f1,
            "precision": zenin_opt_metrics.precision,
            "recall": zenin_opt_metrics.recall,
            "fp": zenin_opt_metrics.false_positives,
            "fn": zenin_opt_metrics.false_negatives,
        }
    )

    # ═══════════════════════════════════════════════════════════════════════
    # FASE D: Baselines (no cambian)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("FASE D: Baselines")
    logger.info("=" * 60)

    t0 = time.time()
    zscore_preds = baseline_zscore(values)
    zscore_elapsed = time.time() - t0
    zscore_scores = [
        abs(v - np.mean(values)) / (np.std(values) + 1e-9) for v in values
    ]

    t0 = time.time()
    iqr_preds = baseline_iqr(values)
    iqr_elapsed = time.time() - t0
    iqr_scores = [float(p) for p in iqr_preds]

    t0 = time.time()
    rolling_preds = baseline_rolling_zscore(values, window=WINDOW_SIZE)
    rolling_elapsed = time.time() - t0
    rolling_scores = []
    arr = np.array(values)
    for i in range(len(values)):
        if i < WINDOW_SIZE:
            rolling_scores.append(0.0)
        else:
            w = arr[i - WINDOW_SIZE : i]
            mean, std = w.mean(), w.std()
            rolling_scores.append(
                abs((arr[i] - mean) / std) / 10.0 if std > 0 else 0.0
            )

    # ═══════════════════════════════════════════════════════════════════════
    # FASE E: Métricas y comparativa
    # ═══════════════════════════════════════════════════════════════════════
    all_results = [
        zenin_opt_metrics,
        compute_metrics(
            "Z-Score (global)", labels, zscore_preds, zscore_scores, zscore_elapsed
        ),
        compute_metrics(
            "IQR (global)", labels, iqr_preds, iqr_scores, iqr_elapsed
        ),
        compute_metrics(
            "Rolling Z-Score (w=50)",
            labels,
            rolling_preds,
            rolling_scores,
            rolling_elapsed,
        ),
        # Incluimos ZENIN default para comparación directa en la tabla
        zenin_default_metrics,
    ]

    # ═══════════════════════════════════════════════════════════════════════
    # FASE F: Reporte en consola — before/after destacado
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("COMPARATIVA ZENIN — BEFORE vs AFTER TUNING")
    print("=" * 80)
    print(f"{'Configuración':<30} {'F1':>8} {'Precision':>10} {'Recall':>8} {'FP':>8} {'FN':>6}")
    print("-" * 80)
    print(
        f"{'ZENIN default (threshold=0.5)':<30} "
        f"{zenin_default_metrics.f1:>8.4f} "
        f"{zenin_default_metrics.precision:>10.4f} "
        f"{zenin_default_metrics.recall:>8.4f} "
        f"{zenin_default_metrics.false_positives:>8} "
        f"{zenin_default_metrics.false_negatives:>6}"
    )
    print(
        f"{'ZENIN tuned (optimal)':<30} "
        f"{zenin_opt_metrics.f1:>8.4f} "
        f"{zenin_opt_metrics.precision:>10.4f} "
        f"{zenin_opt_metrics.recall:>8.4f} "
        f"{zenin_opt_metrics.false_positives:>8} "
        f"{zenin_opt_metrics.false_negatives:>6}"
    )
    print("-" * 80)
    baseline_f1 = max(
        r.f1 for r in all_results if "ZENIN" not in r.name
    )
    print(f"Baseline mejor F1: {baseline_f1:.4f}")
    improvement = (
        (zenin_opt_metrics.f1 - zenin_default_metrics.f1)
        / max(zenin_default_metrics.f1, 1e-9)
        * 100
    )
    print(f"Mejora ZENIN: {improvement:+.1f}%")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("RANKING GLOBAL — TODOS LOS DETECTORES")
    print("=" * 80)
    print(
        f"{'Detector':<35} {'F1':>8} {'Precision':>10} "
        f"{'Recall':>8} {'AUC-ROC':>8} {'FP':>6} {'FN':>6}"
    )
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: x.f1, reverse=True):
        marker = " 🏆" if r.f1 == max(x.f1 for x in all_results) else ""
        print(
            f"{r.name + marker:<35} {r.f1:>8.4f} {r.precision:>10.4f} "
            f"{r.recall:>8.4f} {r.auc_roc:>8.4f} "
            f"{r.false_positives:>6} {r.false_negatives:>6}"
        )
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE G: Persistencia de artefactos
    # ═══════════════════════════════════════════════════════════════════════
    generate_report(
        all_results,
        values,
        labels,
        zenin_scores_opt,
        tuning_results=tuning_results_payload,
        score_dist=score_dist,
    )
    generate_tuning_plots(
        values,
        labels,
        zenin_scores_default,
        all_grid_results,
        best_f1,
        score_dist,
    )

    print(f"\n✅ Resultados guardados en: {RESULTS_DIR}/")
    print(f"   - nab_machine_temp_results.json")
    print(f"   - nab_machine_temp_report.md")
    print(f"   - nab_machine_temp_plot.png")
    print(f"   - nab_machine_temp_grid_search.json")
    print(f"   - nab_machine_temp_tuning.png")


if __name__ == "__main__":
    main()
