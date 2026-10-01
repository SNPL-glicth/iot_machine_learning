"""Autopsia profunda del score de ZENIN en machine_temperature_system_failure.csv.

Analiza:
1. Telemetría interna en cada una de las 4 ventanas canónicas de falla (TPs).
2. Telemetría interna en cada uno de los 70 clusters de falsos positivos (FPs).
3. Desglose detallado por experto (CUSUM, Isolation Forest, Z-score, Rolling Z, etc.).
4. Análisis de la frontera de decisión: por qué falla a 0.65 y triunfa a 0.9120.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from iot_machine_learning.domain.entities.iot.sensor_reading import Reading, SensorWindow
from iot_machine_learning.infrastructure.ml.anomaly.calibration.layer import (
    AdaptiveDetectorCalibrationLayer,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.config import (
    AnomalyDetectorConfig,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.detector import (
    VotingAnomalyDetector,
)

logging.basicConfig(level=logging.WARNING)

BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR.parent / "data" / "NAB"
CSV_PATH = DATA_DIR / "data" / "realKnownCause" / "machine_temperature_system_failure.csv"
WINDOWS_PATH = DATA_DIR / "labels" / "combined_windows.json"
SERIES_ID = "machine_temp_sensor_01"
WINDOW_SIZE = 50
WARMUP_POINTS = 1000
PROBATION_POINTS = 750  # 15% canonical probation period
OPT_THRESHOLD = 0.9119654641138405  # Official Numenta ThresholdSweeper result


def load_dataset() -> tuple[list[str], list[float], list[float]]:
    timestamps_raw: list[str] = []
    timestamps_sec: list[float] = []
    values: list[float] = []

    from datetime import datetime

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row["timestamp"]
            val = float(row["value"])
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            timestamps_raw.append(ts_str)
            timestamps_sec.append(dt.timestamp())
            values.append(val)
    return timestamps_raw, timestamps_sec, values


def load_canonical_windows() -> list[tuple[str, str]]:
    with open(WINDOWS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("realKnownCause/machine_temperature_system_failure.csv", [])


def run_autopsy():
    timestamps_raw, timestamps_sec, values = load_dataset()
    canonical_windows = load_canonical_windows()

    # Mapear ventanas a rangos de índices
    window_ranges: list[tuple[int, int, str, str]] = []
    for w_start_str, w_end_str in canonical_windows:
        # buscar primer indice >= w_start y ultimo <= w_end
        idx_start = next(i for i, ts in enumerate(timestamps_raw) if ts >= w_start_str[:19])
        idx_end = next(i for i in range(len(timestamps_raw) - 1, -1, -1) if timestamps_raw[i] <= w_end_str[:19])
        window_ranges.append((idx_start, idx_end, w_start_str, w_end_str))

    # Inicializar detector exactamente igual a nab_machine_temp_benchmark.py
    cfg = AnomalyDetectorConfig(
        voting_threshold=0.65,
        contamination=0.005,
    )
    detector = VotingAnomalyDetector(
        config=cfg,
        series_id=SERIES_ID,
        enable_adaptive_weights=False,
        calibration_layer=AdaptiveDetectorCalibrationLayer(),
    )

    train_values = values[:WARMUP_POINTS]
    train_timestamps = timestamps_sec[:WARMUP_POINTS]
    detector.train(train_values, timestamps=train_timestamps)

    # Recolectar datos paso a paso
    records: list[dict[str, Any]] = []

    print(f"Ejecutando inferencia sobre {len(values)} puntos...")
    for i in range(WARMUP_POINTS, len(values)):
        slice_values = values[i - WINDOW_SIZE + 1 : i + 1]
        slice_timestamps = timestamps_sec[i - WINDOW_SIZE + 1 : i + 1]
        readings = [
            Reading(series_id=SERIES_ID, value=v, timestamp=t)
            for v, t in zip(slice_values, slice_timestamps, strict=False)
        ]
        window = SensorWindow(series_id=SERIES_ID, readings=readings)
        res = detector.detect(window)

        # Determinar si cae en ventana canónica
        in_win = 0
        for w_idx, (w_s, w_e, _, _) in enumerate(window_ranges, start=1):
            if w_s <= i <= w_e:
                in_win = w_idx
                break

        is_probation = (i < PROBATION_POINTS)
        is_fp = (res.is_anomaly and in_win == 0 and not is_probation)
        is_tp = (res.is_anomaly and in_win > 0)

        record = {
            "idx": i,
            "timestamp": timestamps_raw[i],
            "value": values[i],
            "in_window": in_win,
            "is_probation": is_probation,
            "score": round(res.score, 5),
            "is_anomaly_065": res.is_anomaly,
            "is_anomaly_0912": (res.score >= OPT_THRESHOLD),
            "confidence": round(res.confidence, 4),
            "calib_state": detector._calibration_layer.state.value if detector._calibration_layer else "N/A",
            "votes": {k: round(v, 4) for k, v in res.method_votes.items()},
            "is_fp": is_fp,
            "is_tp": is_tp,
        }
        records.append(record)

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. ANÁLISIS DE LAS 4 VENTANAS CANÓNICAS DE FALLA (TPs)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("AUTOMAQ: ANÁLISIS DE LOS 4 EVENTOS DE FALLA CRÍTICA (VENTANAS CANÓNICAS)")
    print("=" * 90)

    events_summary = []
    for w_idx, (w_s, w_e, w_start_str, w_end_str) in enumerate(window_ranges, start=1):
        win_records = [r for r in records if r["idx"] >= w_s and r["idx"] <= w_e]
        if not win_records:
            continue

        scores = [r["score"] for r in win_records]
        values_win = [r["value"] for r in win_records]
        det_065 = [r for r in win_records if r["is_anomaly_065"]]
        det_0912 = [r for r in win_records if r["is_anomaly_0912"]]

        first_det_065 = det_065[0] if det_065 else None
        first_det_0912 = det_0912[0] if det_0912 else None

        # Promedio de votos por experto dentro de la ventana
        expert_means = defaultdict(list)
        for r in win_records:
            for exp, v in r["votes"].items():
                expert_means[exp].append(v)
        expert_avg = {exp: round(float(np.mean(vals)), 4) for exp, vals in expert_means.items()}
        expert_max = {exp: round(float(np.max(vals)), 4) for exp, vals in expert_means.items()}

        delay_pts_065 = (first_det_065["idx"] - w_s) if first_det_065 else None
        delay_pts_0912 = (first_det_0912["idx"] - w_s) if first_det_0912 else None

        ev_info = {
            "event_id": w_idx,
            "start": w_start_str,
            "end": w_end_str,
            "points": len(win_records),
            "temp_range": (round(min(values_win), 2), round(max(values_win), 2)),
            "score_min": round(min(scores), 4),
            "score_max": round(max(scores), 4),
            "score_mean": round(float(np.mean(scores)), 4),
            "score_median": round(float(np.median(scores)), 4),
            "first_det_065": {
                "idx": first_det_065["idx"] if first_det_065 else None,
                "ts": first_det_065["timestamp"] if first_det_065 else None,
                "delay_pts": delay_pts_065,
                "score": first_det_065["score"] if first_det_065 else None,
                "calib_state": first_det_065["calib_state"] if first_det_065 else None,
            },
            "first_det_0912": {
                "idx": first_det_0912["idx"] if first_det_0912 else None,
                "ts": first_det_0912["timestamp"] if first_det_0912 else None,
                "delay_pts": delay_pts_0912,
                "score": first_det_0912["score"] if first_det_0912 else None,
            },
            "expert_avg": expert_avg,
            "expert_max": expert_max,
        }
        events_summary.append(ev_info)

        print(f"\n--- EVENTO {w_idx}: {w_start_str[:19]} a {w_end_str[:19]} ({len(win_records)} pts) ---")
        print(f"  Rango Temperatura: {ev_info['temp_range'][0]}° a {ev_info['temp_range'][1]}°")
        print(f"  Score ZENIN: Min={ev_info['score_min']:.4f}, Median={ev_info['score_median']:.4f}, Mean={ev_info['score_mean']:.4f}, Max={ev_info['score_max']:.4f}")
        delay_str_065 = f"{delay_pts_065} pts ({delay_pts_065/len(win_records)*100:.1f}% ventana)" if delay_pts_065 is not None else "NO DETECTADO"
        delay_str_0912 = f"{delay_pts_0912} pts ({delay_pts_0912/len(win_records)*100:.1f}% ventana)" if delay_pts_0912 is not None else "NO DETECTADO"
        print(f"  Detección @ 0.65: Retraso={delay_str_065} | Primer TS={first_det_065['timestamp'] if first_det_065 else 'NONE'}")
        print(f"  Detección @ 0.912: Retraso={delay_str_0912} | Primer TS={first_det_0912['timestamp'] if first_det_0912 else 'NONE'}")
        print("  Votos Máximos por Experto:")
        for exp, mv in sorted(expert_max.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {exp:<22}: Max={mv:.4f}, Mean={expert_avg[exp]:.4f}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. ANÁLISIS DE LOS FALSOS POSITIVOS (70 CLUSTERS)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("AUTOMAQ: ANÁLISIS DE LOS FALSOS POSITIVOS FUERA DE VENTANA")
    print("=" * 90)

    fp_records = [r for r in records if r["is_fp"]]
    print(f"Total FP points fuera de ventana: {len(fp_records)}")

    # Agrupar FP en clusters contiguos
    clusters: list[list[dict[str, Any]]] = []
    curr_cluster: list[dict[str, Any]] = []

    for r in records:
        if r["is_fp"]:
            curr_cluster.append(r)
        else:
            if curr_cluster:
                clusters.append(curr_cluster)
                curr_cluster = []
    if curr_cluster:
        clusters.append(curr_cluster)

    print(f"Total FP clusters formados: {len(clusters)}")

    # Analizar características de los clusters FP
    cluster_stats = []
    for c_idx, c in enumerate(clusters, start=1):
        c_scores = [r["score"] for r in c]
        c_vals = [r["value"] for r in c]
        c_max_score = max(c_scores)
        c_mean_score = float(np.mean(c_scores))
        c_len = len(c)
        c_ts_start = c[0]["timestamp"]
        c_ts_end = c[-1]["timestamp"]

        # Que expertos empujaron el score en este cluster?
        c_exp_max = defaultdict(float)
        for r in c:
            for exp, v in r["votes"].items():
                if v > c_exp_max[exp]:
                    c_exp_max[exp] = v

        survives_0912 = any(r["is_anomaly_0912"] for r in c)

        stat = {
            "cluster_id": c_idx,
            "start": c_ts_start,
            "end": c_ts_end,
            "length": c_len,
            "temp_min": round(min(c_vals), 2),
            "temp_max": round(max(c_vals), 2),
            "max_score": round(c_max_score, 4),
            "mean_score": round(c_mean_score, 4),
            "survives_0912": survives_0912,
            "top_experts": sorted(c_exp_max.items(), key=lambda x: x[1], reverse=True)[:4],
        }
        cluster_stats.append(stat)

    # Distribución de max_scores en los clusters FP
    max_scores = [cs["max_score"] for cs in cluster_stats]
    lengths = [cs["length"] for cs in cluster_stats]
    surviving_clusters = [cs for cs in cluster_stats if cs["survives_0912"]]

    print(f"\nLongitud de clusters FP: Min={min(lengths)}, Median={np.median(lengths):.1f}, Max={max(lengths)}")
    print(f"Max Scores en clusters FP: Min={min(max_scores):.4f}, P50={np.percentile(max_scores, 50):.4f}, P90={np.percentile(max_scores, 90):.4f}, Max={max(max_scores):.4f}")
    print(f"\n¿Cuántos clusters FP sobreviven a theta* = 0.9120?: {len(surviving_clusters)} de {len(clusters)} ({len(surviving_clusters)/len(clusters)*100:.1f}%)")
    print(f"¿Cuántos puntos FP sobreviven a theta* = 0.9120?: {sum(1 for r in fp_records if r['is_anomaly_0912'])} de {len(fp_records)}")

    # Expertos que causaron las falsas alarmas
    expert_fp_contribution = defaultdict(list)
    for c in cluster_stats:
        for exp, val in c["top_experts"]:
            expert_fp_contribution[exp].append(val)

    print("\nImpacto promedio de cada experto en los clusters FP:")
    for exp, vals in sorted(expert_fp_contribution.items(), key=lambda x: np.mean(x[1]), reverse=True):
        print(f"  - {exp:<22}: Presencia en {len(vals)}/{len(clusters)} clusters | Score medio={np.mean(vals):.4f} | Score max={np.max(vals):.4f}")

    print("\nTop 5 clusters FP con mayor score:")
    top_5 = sorted(cluster_stats, key=lambda x: x["max_score"], reverse=True)[:5]
    for cs in top_5:
        print(f"  Cluster {cs['cluster_id']} ({cs['start'][:16]} a {cs['end'][:16]}, {cs['length']} pts): MaxScore={cs['max_score']} (Temp: {cs['temp_min']}°-{cs['temp_max']}°)")
        print(f"    Top Experts: {cs['top_experts']}")

    # Guardar resumen en JSON
    out_path = BENCHMARK_DIR / "results" / "zenin_score_autopsy.json"
    autopsy_data = {
        "events": events_summary,
        "fp_clusters_total": len(clusters),
        "fp_points_total": len(fp_records),
        "fp_clusters_surviving_0912": len(surviving_clusters),
        "fp_points_surviving_0912": sum(1 for r in fp_records if r['is_anomaly_0912']),
        "cluster_stats": cluster_stats,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(autopsy_data, f, indent=2)
    print(f"\nReporte de autopsia guardado en: {out_path}")


if __name__ == "__main__":
    run_autopsy()
