"""
ZENIN NAB Forensic Analysis — FASE 1 + FASE 2

Objetivos:
  1. Clasificar los FN (falsos negativos) por tipo de anomalía.
  2. Contribution analysis por detector (TP/FP/FN/TN individual).
  3. Ablation study: quitar un detector a la vez, medir delta F1.

Uso:
    cd iot_machine_learning
    python benchmarks/nab_forensic_analysis.py

Output:
    benchmarks/results/nab_forensic_report.md
    benchmarks/results/nab_forensic_fn_details.json
    benchmarks/results/nab_forensic_contribution.json
    benchmarks/results/nab_forensic_ablation.json
"""

from __future__ import annotations

import json
import sys
import time
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("nab_forensic")

# BLOCKER: make iot_machine_learning importable
sys.path.insert(0, "/home/nicolas/Documentos/Iot_System")
sys.path.insert(0, "/home/nicolas/Documentos/Iot_System/iot_machine_learning")

from sklearn.metrics import f1_score, precision_score, recall_score

# ─── Paths ───────────────────────────────────────────────────────────────────
NAB_ROOT = Path("/tmp/NAB")
DATASET_PATH = NAB_ROOT / "data/realKnownCause/machine_temperature_system_failure.csv"
LABELS_PATH = NAB_ROOT / "labels/combined_labels.json"
RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 50
DETECTION_WINDOW = 10
SERIES_ID = "NAB-machine-temp-001"
VOTING_THRESHOLD = 0.75  # óptimo del grid search previo

# Pesos default del ensemble — siempre leer de AnomalyDetectorConfig para estar sincronizado
from iot_machine_learning.infrastructure.ml.anomaly.core.config import AnomalyDetectorConfig
DEFAULT_WEIGHTS = dict(AnomalyDetectorConfig().weights)
ALL_DETECTORS = list(DEFAULT_WEIGHTS.keys())


# ─── Carga de datos ─────────────────────────────────────────────────────────

def load_nab_dataset() -> Tuple[List[float], List[float], List[int], pd.DataFrame]:
    """Carga dataset NAB con labels. Retorna values, timestamps, labels, df."""
    logger.info(f"Cargando dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    values = df["value"].tolist()
    timestamps = [ts.timestamp() for ts in df["timestamp"]]

    with open(LABELS_PATH) as f:
        all_labels = json.load(f)

    anomaly_key = "realKnownCause/machine_temperature_system_failure.csv"
    anomaly_timestamps = all_labels.get(anomaly_key, [])
    anomaly_dts = pd.to_datetime(anomaly_timestamps)

    labels = [0] * len(df)
    for anomaly_dt in anomaly_dts:
        closest_idx = (df["timestamp"] - anomaly_dt).abs().idxmin()
        start = max(0, closest_idx - DETECTION_WINDOW)
        end = min(len(df), closest_idx + DETECTION_WINDOW + 1)
        for idx in range(start, end):
            labels[idx] = 1

    logger.info(
        f"Dataset cargado: {len(values)} puntos, {sum(labels)} anomaly-points ({sum(labels)/len(values)*100:.2f}%)"
    )
    return values, timestamps, labels, df


# ─── Ejecución ZENIN con per-point metadata ─────────────────────────────────

@dataclass
class PointResult:
    index: int
    value: float
    timestamp: float
    score: float
    prediction: int
    label: int
    votes: Dict[str, float]
    is_fn: bool
    is_fp: bool
    is_tp: bool
    is_tn: bool


def run_zenin_detailed(
    values: List[float],
    timestamps: List[float],
    labels: List[int],
    voting_threshold: float = VOTING_THRESHOLD,
    weights: Optional[Dict[str, float]] = None,
    exclude_detector: Optional[str] = None,
) -> Tuple[List[PointResult], float]:
    """
    Corre VotingAnomalyDetector y retorna resultados punto a punto
    con votos individuales por detector.
    """
    from iot_machine_learning.domain.entities.iot.sensor_reading import (
        Reading,
        TimeSeriesWindow,
    )
    from iot_machine_learning.infrastructure.ml.anomaly.core.detector import (
        VotingAnomalyDetector,
    )
    from iot_machine_learning.infrastructure.ml.anomaly.core.config import (
        AnomalyDetectorConfig,
    )

    cfg = AnomalyDetectorConfig(
        voting_threshold=voting_threshold,
        weights=weights or dict(DEFAULT_WEIGHTS),
        contamination=0.005,
    )

    detector = VotingAnomalyDetector(
        config=cfg,
        series_id=SERIES_ID,
        enable_adaptive_weights=False,
    )

    # Si se pide excluir un detector, filtramos sub_detectors
    if exclude_detector:
        detector._sub_detectors = [
            d for d in detector._sub_detectors if d.method_name != exclude_detector
        ]
        # Reconstruir outcome trackers
        detector._detector_outcomes = {
            d.method_name: detector._detector_outcomes.get(d.method_name, __import__('collections').deque(maxlen=50))
            for d in detector._sub_detectors
        }
        # Renormalizar pesos restantes
        remaining = [d.method_name for d in detector._sub_detectors]
        new_weights = {k: v for k, v in cfg.weights.items() if k in remaining}
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}
        detector._adaptive_weights = new_weights
        detector._strategy = detector._strategy.__class__(
            weights=new_weights, threshold=voting_threshold
        )
        logger.info(f"Ablation: excluyendo '{exclude_detector}', pesos renormalizados: {new_weights}")

    # Train
    train_values = values[:WINDOW_SIZE]
    train_timestamps = timestamps[:WINDOW_SIZE]
    detector.train(train_values, timestamps=train_timestamps)

    results: List[PointResult] = []
    start_time = time.time()

    for i in range(WINDOW_SIZE, len(values)):
        slice_values = values[i - WINDOW_SIZE + 1 : i + 1]
        slice_timestamps = timestamps[i - WINDOW_SIZE + 1 : i + 1]

        readings = [
            Reading(series_id=SERIES_ID, value=v, timestamp=t)
            for v, t in zip(slice_values, slice_timestamps)
        ]
        window = TimeSeriesWindow(series_id=SERIES_ID, readings=readings)

        result = detector.detect(window)
        pred = int(result.is_anomaly)
        label = labels[i]

        pr = PointResult(
            index=i,
            value=values[i],
            timestamp=timestamps[i],
            score=float(result.score),
            prediction=pred,
            label=label,
            votes=dict(result.method_votes) if result.method_votes else {},
            is_fn=(label == 1 and pred == 0),
            is_fp=(label == 0 and pred == 1),
            is_tp=(label == 1 and pred == 1),
            is_tn=(label == 0 and pred == 0),
        )
        results.append(pr)

    elapsed = time.time() - start_time
    return results, elapsed


# ─── Métricas por detector individual ────────────────────────────────────────

def compute_detector_contribution(results: List[PointResult]) -> Dict[str, Dict]:
    """
    Para cada detector, calcula cuántos TP/FP/FN/TN hubiera generado
    si ese detector votara SOLO (threshold 0.5 sobre su voto individual).
    """
    contrib: Dict[str, Dict] = {}

    for det_name in ALL_DETECTORS:
        tp = fp = fn = tn = 0
        vote_sum = 0.0
        vote_count = 0

        for pr in results:
            vote = pr.votes.get(det_name, 0.0)
            vote_sum += vote
            vote_count += 1

            # Si el detector vota > 0.5 => predice anomalía
            pred = 1 if vote > 0.5 else 0
            label = pr.label

            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            else:
                tn += 1

        total_anomalies = tp + fn
        total_normals = fp + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_anomalies if total_anomalies > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        contrib[det_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_vote": round(vote_sum / vote_count, 4) if vote_count else 0.0,
            "vote_rate": round((tp + fp) / (tp + fp + fn + tn), 4),  # tasa de votos positivos
        }

    return contrib


# ─── Clasificación de FN ────────────────────────────────────────────────────

def classify_fn(results: List[PointResult], values: List[float], labels: List[int], df: pd.DataFrame) -> List[Dict]:
    """
    Clasifica cada FN por tipo de anomalía:
      - spike: cambio brusco > 3σ local en ventana de ±5 puntos
      - drift: cambio gradual persistente > 2σ durante ≥10 puntos
      - short: anomalía que dura < 5 puntos
      - gradual: anomalía lenta sin spike claro
      - noise: ruido de alta frecuencia antes del FN
    """
    fn_points = [pr for pr in results if pr.is_fn]
    classified = []

    arr = np.array(values)
    rolling_std = pd.Series(arr).rolling(window=21, center=True, min_periods=5).std().values
    rolling_mean = pd.Series(arr).rolling(window=21, center=True, min_periods=5).mean().values

    for pr in fn_points:
        idx = pr.index
        local_std = rolling_std[idx] if not np.isnan(rolling_std[idx]) else np.std(arr[max(0, idx-10):idx+10])
        local_mean = rolling_mean[idx] if not np.isnan(rolling_mean[idx]) else np.mean(arr[max(0, idx-10):idx+10])

        # Ventana local
        win_start = max(0, idx - 5)
        win_end = min(len(values), idx + 6)
        local_vals = arr[win_start:win_end]

        # Spike: máxima desviación local > 3σ
        max_dev = np.max(np.abs(local_vals - local_mean)) if local_std and local_std > 0 else 0
        is_spike = max_dev > 3 * local_std if local_std > 0 else False

        # Drift: tendencia persistente
        trend_start = max(0, idx - 10)
        trend_end = min(len(values), idx + 11)
        trend_vals = arr[trend_start:trend_end]
        trend = np.polyfit(np.arange(len(trend_vals)), trend_vals, 1)[0] if len(trend_vals) > 2 else 0
        is_drift = abs(trend) > 0.5 * local_std if local_std > 0 else False

        # Short: contexto de anomalía breve
        # Buscamos cuántos puntos con label=1 consecutivos alrededor
        label_window = labels[max(0, idx-5):min(len(labels), idx+6)]
        anomaly_run = sum(label_window)
        is_short = anomaly_run <= 5

        # Noise: alta variabilidad previa
        prev_vals = arr[max(0, idx-10):idx]
        is_noise = np.std(prev_vals) > 2 * np.std(arr) if len(prev_vals) > 2 else False

        # Clasificación jerárquica
        if is_spike:
            fn_type = "spike"
        elif is_drift and not is_short:
            fn_type = "drift"
        elif is_short:
            fn_type = "short"
        elif is_noise:
            fn_type = "noise_masked"
        else:
            fn_type = "gradual"

        classified.append({
            "index": idx,
            "value": round(float(pr.value), 4),
            "score": round(float(pr.score), 4),
            "votes": {k: round(v, 3) for k, v in pr.votes.items()},
            "fn_type": fn_type,
            "local_std": round(float(local_std), 4) if not np.isnan(local_std) else None,
            "max_dev_sigma": round(float(max_dev / local_std), 2) if local_std and local_std > 0 else None,
            "trend": round(float(trend), 4) if not np.isnan(trend) else None,
            "anomaly_run_length": int(anomaly_run),
            "is_noise_before": bool(is_noise),
        })

    return classified


# ─── Ablation study ─────────────────────────────────────────────────────────

def run_ablation_study(
    values: List[float],
    timestamps: List[float],
    labels: List[float],
) -> List[Dict]:
    """
    Quita un detector a la vez, re-run ZENIN, mide delta F1.
    """
    # Baseline: todos los detectores
    baseline_results, _ = run_zenin_detailed(
        values, timestamps, labels, voting_threshold=VOTING_THRESHOLD
    )
    baseline_preds = [0] * len(values)
    for pr in baseline_results:
        baseline_preds[pr.index] = pr.prediction

    baseline_f1 = f1_score(labels[WINDOW_SIZE:], [baseline_preds[i] for i in range(WINDOW_SIZE, len(values))], zero_division=0)
    baseline_precision = precision_score(labels[WINDOW_SIZE:], [baseline_preds[i] for i in range(WINDOW_SIZE, len(values))], zero_division=0)
    baseline_recall = recall_score(labels[WINDOW_SIZE:], [baseline_preds[i] for i in range(WINDOW_SIZE, len(values))], zero_division=0)

    logger.info(f"Ablation baseline — F1={baseline_f1:.4f} P={baseline_precision:.4f} R={baseline_recall:.4f}")

    ablation_results = []
    for det_name in ALL_DETECTORS:
        logger.info(f"Ablation: excluyendo {det_name}...")
        ab_results, _ = run_zenin_detailed(
            values, timestamps, labels,
            voting_threshold=VOTING_THRESHOLD,
            exclude_detector=det_name,
        )
        ab_preds = [0] * len(values)
        for pr in ab_results:
            ab_preds[pr.index] = pr.prediction

        ab_f1 = f1_score(labels[WINDOW_SIZE:], [ab_preds[i] for i in range(WINDOW_SIZE, len(values))], zero_division=0)
        ab_precision = precision_score(labels[WINDOW_SIZE:], [ab_preds[i] for i in range(WINDOW_SIZE, len(values))], zero_division=0)
        ab_recall = recall_score(labels[WINDOW_SIZE:], [ab_preds[i] for i in range(WINDOW_SIZE, len(values))], zero_division=0)

        delta_f1 = ab_f1 - baseline_f1
        delta_precision = ab_precision - baseline_precision
        delta_recall = ab_recall - baseline_recall

        impact = "toxic" if delta_f1 > 0.01 else "neutral" if abs(delta_f1) < 0.01 else "helpful"

        ablation_results.append({
            "excluded_detector": det_name,
            "f1": round(ab_f1, 4),
            "precision": round(ab_precision, 4),
            "recall": round(ab_recall, 4),
            "delta_f1": round(delta_f1, 4),
            "delta_precision": round(delta_precision, 4),
            "delta_recall": round(delta_recall, 4),
            "impact": impact,
        })

    return ablation_results


# ─── Reporte Markdown ────────────────────────────────────────────────────────

def generate_forensic_report(
    results: List[PointResult],
    fn_classified: List[Dict],
    contribution: Dict[str, Dict],
    ablation: List[Dict],
    values: List[float],
    labels: List[int],
) -> None:
    """Genera reporte Markdown con todo el análisis."""

    # Métricas globales
    preds = [0] * len(values)
    scores = [0.0] * len(values)
    for pr in results:
        preds[pr.index] = pr.prediction
        scores[pr.index] = pr.score

    global_f1 = f1_score(labels[WINDOW_SIZE:], [preds[i] for i in range(WINDOW_SIZE, len(values))], zero_division=0)
    global_precision = precision_score(labels[WINDOW_SIZE:], [preds[i] for i in range(WINDOW_SIZE, len(values))], zero_division=0)
    global_recall = recall_score(labels[WINDOW_SIZE:], [preds[i] for i in range(WINDOW_SIZE, len(values))], zero_division=0)

    n_fn = sum(1 for pr in results if pr.is_fn)
    n_fp = sum(1 for pr in results if pr.is_fp)
    n_tp = sum(1 for pr in results if pr.is_tp)
    n_tn = sum(1 for pr in results if pr.is_tn)

    md_path = RESULTS_DIR / "nab_forensic_report.md"
    with open(md_path, "w") as f:
        f.write("# ZENIN NAB Forensic Analysis\n\n")
        f.write(f"**Dataset:** `realKnownCause/machine_temperature_system_failure`\n")
        f.write(f"**Threshold:** {VOTING_THRESHOLD}\n")
        f.write(f"**Window size:** {WINDOW_SIZE}\n\n")

        f.write("## 1. Métricas Globales (post warm-up)\n\n")
        f.write(f"| Métrica | Valor |\n")
        f.write(f"|---------|-------|\n")
        f.write(f"| F1 | {global_f1:.4f} |\n")
        f.write(f"| Precision | {global_precision:.4f} |\n")
        f.write(f"| Recall | {global_recall:.4f} |\n")
        f.write(f"| TP | {n_tp} |\n")
        f.write(f"| FP | {n_fp} |\n")
        f.write(f"| FN | {n_fn} |\n")
        f.write(f"| TN | {n_tn} |\n\n")

        # ── FN Classification ──
        f.write("## 2. Análisis de Falsos Negativos (FN)\n\n")
        f.write(f"**Total FN:** {n_fn}\n\n")

        if fn_classified:
            # Agrupar por tipo
            type_counts = defaultdict(int)
            for fn in fn_classified:
                type_counts[fn["fn_type"]] += 1

            f.write("### 2.1 Distribución por tipo\n\n")
            f.write("| Tipo | Count | % del total FN |\n")
            f.write("|------|-------|----------------|\n")
            for fn_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                pct = count / n_fn * 100
                f.write(f"| {fn_type} | {count} | {pct:.1f}% |\n")
            f.write("\n")

            f.write("### 2.2 Detalle de cada FN\n\n")
            f.write("| Idx | Valor | Score | Tipo | Votes |\n")
            f.write("|-----|-------|-------|------|-------|\n")
            for fn in fn_classified[:30]:  # top 30
                votes_str = ", ".join(f"{k}={v:.2f}" for k, v in fn["votes"].items())
                f.write(
                    f"| {fn['index']} | {fn['value']} | {fn['score']} | {fn['fn_type']} | {votes_str} |\n"
                )
            if len(fn_classified) > 30:
                f.write(f"\n*({len(fn_classified) - 30} FN adicionales en JSON)*\n")
            f.write("\n")

        # ── Contribution Analysis ──
        f.write("## 3. Contribution Analysis por Detector\n\n")
        f.write("| Detector | TP | FP | FN | TN | Precision | Recall | F1 | Vote Rate | Impact |\n")
        f.write("|----------|----|----|----|----|-----------|--------|----|-----------|--------|\n")

        for det_name in ALL_DETECTORS:
            c = contribution[det_name]
            # Buscar impacto del ablation
            ab = next((a for a in ablation if a["excluded_detector"] == det_name), None)
            impact = ab["impact"] if ab else "unknown"
            delta_f1 = ab["delta_f1"] if ab else 0.0
            f.write(
                f"| {det_name} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | "
                f"{c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | "
                f"{c['vote_rate']:.4f} | {impact} ({delta_f1:+.4f}) |\n"
            )
        f.write("\n")

        # ── Ablation Study ──
        f.write("## 4. Ablation Study\n\n")
        f.write(f"**Baseline F1:** {ablation[0]['f1'] - ablation[0]['delta_f1']:.4f}\n\n")
        f.write("| Excluido | F1 | Delta F1 | Delta P | Delta R | Impacto |\n")
        f.write("|----------|----|----------|---------|---------|---------|\n")
        for a in sorted(ablation, key=lambda x: x["delta_f1"]):
            marker = " 🔥" if a["impact"] == "toxic" else " ✅" if a["impact"] == "helpful" else ""
            f.write(
                f"| {a['excluded_detector']}{marker} | {a['f1']:.4f} | "
                f"{a['delta_f1']:+.4f} | {a['delta_precision']:+.4f} | {a['delta_recall']:+.4f} | "
                f"{a['impact']} |\n"
            )
        f.write("\n")

        f.write("## 5. Interpretación y Recomendaciones\n\n")
        f.write("### Detectores tóxicos (quitarlos mejora F1)\n")
        toxic = [a for a in ablation if a["impact"] == "toxic"]
        if toxic:
            for t in toxic:
                f.write(f"- **{t['excluded_detector']}**: quitarlo mejora F1 en {t['delta_f1']:+.4f}\n")
        else:
            f.write("- Ninguno. Todos los detectores aportan (o son neutrales).\n")
        f.write("\n")

        f.write("### Detectores críticos (quitarlos empeora F1)\n")
        helpful = [a for a in ablation if a["impact"] == "helpful"]
        for h in helpful:
            f.write(f"- **{h['excluded_detector']}**: quitarlo empeora F1 en {h['delta_f1']:+.4f}\n")
        f.write("\n")

        if fn_classified:
            dominant_fn_type = max(type_counts.items(), key=lambda x: x[1])[0]
            f.write(f"### Tipo de FN dominante: `{dominant_fn_type}`\n")
            f.write(f"{type_counts[dominant_fn_type]} de {n_fn} FN son de tipo `{dominant_fn_type}`.\n")
            if dominant_fn_type == "spike":
                f.write("→ Los spikes pequeños no están siendo capturados. Considerar bajar thresholds de Z-score/velocity en régimen stable.\n")
            elif dominant_fn_type == "drift":
                f.write("→ Los drift lentos escapan. Considerar activar drift-aware detection o usar Kalman residual como input al ensemble.\n")
            elif dominant_fn_type == "short":
                f.write("→ Anomalías cortas (< 5 pts) se pierden. Considerar aumentar DETECTION_WINDOW o reducir voting_threshold momentáneamente.\n")
            elif dominant_fn_type == "noise_masked":
                f.write("→ Ruido previo enmascara anomalías. Considerar pre-filtrado (Hampel) antes del ensemble.\n")
            else:
                f.write("→ Anomalías graduales sin spike claro. Considerar IF con ventanas más cortas o LOF con menor contamination.\n")

    logger.info(f"Reporte guardado: {md_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("ZENIN NAB FORENSIC ANALYSIS — FASE 1 + FASE 2")
    logger.info("=" * 70)

    # 1. Cargar datos
    values, timestamps, labels, df = load_nab_dataset()

    # 2. Correr ZENIN con per-point metadata
    logger.info("\nEjecutando ZENIN con votos detallados (threshold=0.75)...")
    results, elapsed = run_zenin_detailed(values, timestamps, labels, voting_threshold=VOTING_THRESHOLD)
    logger.info(f"Completado en {elapsed:.2f}s — {len(results)} puntos analizados")

    # 3. Contribution analysis
    logger.info("\nCalculando contribution analysis por detector...")
    contribution = compute_detector_contribution(results)
    for det_name, c in contribution.items():
        logger.info(
            f"  {det_name}: F1={c['f1']:.4f} P={c['precision']:.4f} R={c['recall']:.4f} "
            f"(TP={c['tp']} FP={c['fp']} FN={c['fn']})"
        )

    # 4. Clasificar FN
    logger.info("\nClasificando FN por tipo de anomalía...")
    fn_classified = classify_fn(results, values, labels, df)
    type_counts = defaultdict(int)
    for fn in fn_classified:
        type_counts[fn["fn_type"]] += 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {t}: {c} FN")

    # 5. Ablation study
    logger.info("\nIniciando ablation study (esto toma tiempo)...")
    ablation = run_ablation_study(values, timestamps, labels)
    for a in ablation:
        logger.info(
            f"  Sin {a['excluded_detector']}: F1={a['f1']:.4f} delta={a['delta_f1']:+.4f} [{a['impact']}]"
        )

    # 6. Persistir JSONs
    json_path = RESULTS_DIR / "nab_forensic_fn_details.json"
    with open(json_path, "w") as f:
        json.dump(fn_classified, f, indent=2)
    logger.info(f"FN details guardado: {json_path}")

    contrib_path = RESULTS_DIR / "nab_forensic_contribution.json"
    with open(contrib_path, "w") as f:
        json.dump(contribution, f, indent=2)
    logger.info(f"Contribution guardado: {contrib_path}")

    ablation_path = RESULTS_DIR / "nab_forensic_ablation.json"
    with open(ablation_path, "w") as f:
        json.dump(ablation, f, indent=2)
    logger.info(f"Ablation guardado: {ablation_path}")

    # 7. Reporte Markdown
    generate_forensic_report(results, fn_classified, contribution, ablation, values, labels)

    logger.info("\n✅ Forensic analysis completo.")


if __name__ == "__main__":
    main()
