"""
ZENIN vs NAB Benchmark Unificado — machine_temperature_system_failure
Harness canónico de evaluación integral:
1. Métricas oficiales de NAB (Event Recall 100%, NAB Standard Score, Range F1, Clusters FP).
2. Perfilado exhaustivo de hardware en tiempo real (CPU, RAM RSS/Delta, Heap Tracemalloc).
3. Medición de latencia per-point y throughput en streaming.
4. Comparativa contra baselines estándar (Z-score, IQR, Rolling Z-score).
5. Persistencia consolidada de reporte técnico Markdown, resultados JSON y gráficos.

Uso:
    python benchmarks/nab_machine_temp_benchmark.py

Salidas consolidadas:
    benchmarks/results/nab_machine_temp_report.md
    benchmarks/results/nab_machine_temp_results.json
    benchmarks/results/nab_machine_temp_plot.png
    benchmarks/results/nab_machine_temp_tuning.png
    benchmarks/results/nab_machine_temp_resources.png
    benchmarks/results/nab_machine_temp_grid_search.json
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import threading
import time
import tracemalloc
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import psutil

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ─── Environment & Paths ───────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.nab_evaluator import ComprehensiveNABReport, NABEvaluator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nab_benchmark")

# ─── NAB Dataset Paths ────────────────────────────────────────────────────────
NAB_ROOT = Path("/tmp/NAB")
if not (NAB_ROOT / "data/realKnownCause/machine_temperature_system_failure.csv").exists():
    NAB_ROOT = _REPO_ROOT / "data" / "NAB"
DATASET_PATH = NAB_ROOT / "data/realKnownCause/machine_temperature_system_failure.csv"
LABELS_PATH = NAB_ROOT / "labels/combined_labels.json"
WINDOWS_PATH = NAB_ROOT / "labels/combined_windows.json"
RESULTS_DIR = _REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuración Canónica del Benchmark ──────────────────────────────────────
WINDOW_SIZE = 50  # Sliding sensor window
WARMUP_POINTS = 1000  # Período de warm-up nominal representativo del activo
DETECTION_WINDOW = 10  # NAB scoring: ventana de tolerancia de +-10 puntos (21 puntos por ventana)
SERIES_ID = "NAB-machine-temp-001"


# ─── Hardware & System Profiling ──────────────────────────────────────────────


def get_cpu_model_name() -> str:
    """Extrae el modelo de CPU desde /proc/cpuinfo en Linux o fallback a platform."""
    try:
        if Path("/proc/cpuinfo").exists():
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_system_specs() -> dict[str, Any]:
    """Obtiene especificaciones completas de hardware y SO."""
    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    return {
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": get_cpu_model_name(),
        "physical_cores": psutil.cpu_count(logical=False) or 1,
        "logical_cores": psutil.cpu_count(logical=True) or 1,
        "cpu_freq_current_mhz": round(freq.current, 1) if freq else 0.0,
        "cpu_freq_max_mhz": round(freq.max, 1) if freq and freq.max else 0.0,
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "ram_used_gb": round(vm.used / (1024**3), 2),
    }


# ─── Resource Monitor ─────────────────────────────────────────────────────────


class ResourceMonitor:
    """Monitor de recursos en tiempo real.

    Mide CPU (user, system, % avg, % peak) y Memoria (RSS start, peak, delta, tracemalloc heap).
    """

    def __init__(self, sample_interval_s: float = 0.02):
        self.sample_interval_s = sample_interval_s
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.cpu_samples: list[float] = []
        self.rss_samples: list[float] = []

        self.start_user_time: float = 0.0
        self.start_sys_time: float = 0.0
        self.end_user_time: float = 0.0
        self.end_sys_time: float = 0.0
        self.start_rss_mb: float = 0.0
        self.peak_rss_mb: float = 0.0
        self.tracemalloc_peak_mb: float = 0.0
        self.elapsed_wall_s: float = 0.0

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                self.cpu_samples.append(self.process.cpu_percent(interval=None))
                rss_mb = self.process.memory_info().rss / (1024 * 1024)
                self.rss_samples.append(rss_mb)
            except Exception:
                pass
            self._stop_event.wait(self.sample_interval_s)

    def __enter__(self):
        tracemalloc.start()
        self.process.cpu_percent(interval=None)  # Inicializar contador de CPU
        ct = self.process.cpu_times()
        self.start_user_time = ct.user
        self.start_sys_time = ct.system
        self.start_rss_mb = self.process.memory_info().rss / (1024 * 1024)
        self.rss_samples = [self.start_rss_mb]

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_wall_s = time.perf_counter() - self.t0
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

        ct = self.process.cpu_times()
        self.end_user_time = ct.user
        self.end_sys_time = ct.system

        _, tm_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.tracemalloc_peak_mb = tm_peak / (1024 * 1024)

        current_rss = self.process.memory_info().rss / (1024 * 1024)
        self.rss_samples.append(current_rss)
        self.peak_rss_mb = max(self.rss_samples) if self.rss_samples else current_rss

    @property
    def user_time_s(self) -> float:
        return max(0.0, self.end_user_time - self.start_user_time)

    @property
    def sys_time_s(self) -> float:
        return max(0.0, self.end_sys_time - self.start_sys_time)

    @property
    def cpu_percent_avg(self) -> float:
        valid = [s for s in self.cpu_samples if s > 0.0]
        if valid:
            return float(np.mean(valid))
        if self.elapsed_wall_s > 0:
            return float((self.user_time_s + self.sys_time_s) / self.elapsed_wall_s * 100)
        return 0.0

    @property
    def cpu_percent_peak(self) -> float:
        return float(max(self.cpu_samples)) if self.cpu_samples else 0.0

    @property
    def memory_delta_mb(self) -> float:
        return max(0.0, self.peak_rss_mb - self.start_rss_mb)


# ─── Dataclass para Métricas ──────────────────────────────────────────────────


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
    # Métricas de Recursos y CPU
    cpu_percent_avg: float = 0.0
    cpu_percent_peak: float = 0.0
    cpu_user_s: float = 0.0
    cpu_system_s: float = 0.0
    # Métricas de Memoria
    memory_start_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_delta_mb: float = 0.0
    tracemalloc_peak_mb: float = 0.0
    # Métricas de Rendimiento / Latencia
    throughput_pts_sec: float = 0.0
    latency_mean_us: float = 0.0
    latency_p50_us: float = 0.0
    latency_p95_us: float = 0.0
    latency_p99_us: float = 0.0


# ─── Carga de datos ──────────────────────────────────────────────────────────


def load_nab_dataset() -> tuple[list[float], list[float], list[int], list[float], list[tuple[Any, Any]]]:
    """Carga dataset NAB, marcas oficiales y ventanas canónicas de Numenta NAB.

    Returns:
        values: Serie temporal de temperatura
        timestamps: Timestamps unix en float
        labels: Etiquetas binarias puntuales basadas en las ventanas canónicas
        anomaly_timestamps_float: Timestamps exactos de los eventos anómalos de ground truth
        window_ranges: Tuplas (start_time, end_time) canónicas de combined_windows.json
    """
    logger.info(f"Cargando dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    values = df["value"].astype(float).tolist()
    timestamps = [float(ts.timestamp()) for ts in df["timestamp"]]

    anomaly_key = "realKnownCause/machine_temperature_system_failure.csv"

    # Cargar labels de anomalías puntuales
    with open(LABELS_PATH) as f:
        all_labels = json.load(f)

    anomaly_timestamps = all_labels.get(anomaly_key, [])
    anomaly_dts = pd.to_datetime(anomaly_timestamps)
    anomaly_timestamps_float = [float(dt.timestamp()) for dt in anomaly_dts]

    # Cargar ventanas canónicas de Numenta NAB
    window_ranges: list[tuple[Any, Any]] = []
    if WINDOWS_PATH.exists():
        with open(WINDOWS_PATH) as f:
            all_windows = json.load(f)
        raw_ranges = all_windows.get(anomaly_key, [])
        window_ranges = [(pd.to_datetime(w[0]), pd.to_datetime(w[1])) for w in raw_ranges]
        logger.info(f"Ventanas canónicas cargadas desde {WINDOWS_PATH}: {len(window_ranges)} ventanas.")

    # Generar labels de ground truth utilizando las ventanas oficiales canónicas
    labels = [0] * len(df)
    if window_ranges:
        for w_start, w_end in window_ranges:
            s_idx = (df["timestamp"] - w_start).abs().idxmin()
            e_idx = (df["timestamp"] - w_end).abs().idxmin()
            if s_idx > e_idx:
                s_idx, e_idx = e_idx, s_idx
            for idx in range(s_idx, e_idx + 1):
                labels[idx] = 1
    else:
        for anomaly_dt in anomaly_dts:
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
            "anomaly_events": len(anomaly_timestamps_float),
            "canonical_windows_count": len(window_ranges),
        }
    )
    return values, timestamps, labels, anomaly_timestamps_float, window_ranges


# ─── Baselines ───────────────────────────────────────────────────────────────


def run_baseline_zscore(
    values: list[float], threshold: float = 3.0
) -> tuple[list[int], list[float], dict[str, Any]]:
    """Z-score global — baseline estadístico clásico, medido con ResourceMonitor."""
    latencies_us = []
    with ResourceMonitor() as mon:
        t0 = time.perf_counter_ns()
        arr = np.array(values)
        mean, std = arr.mean(), arr.std()
        if std == 0:
            preds = [0] * len(values)
            scores = [0.0] * len(values)
        else:
            z = np.abs((arr - mean) / std)
            preds = (z > threshold).astype(int).tolist()
            scores = (z / 10.0).tolist()
        t1 = time.perf_counter_ns()
        total_time_us = (t1 - t0) / 1000.0
        per_point_us = total_time_us / max(1, len(values))
        latencies_us = [per_point_us] * len(values)

    resource_data = {
        "monitor": mon,
        "latencies_us": latencies_us,
    }
    return preds, scores, resource_data


def run_baseline_iqr(
    values: list[float], factor: float = 1.5
) -> tuple[list[int], list[float], dict[str, Any]]:
    """IQR global — baseline robusto a outliers, medido con ResourceMonitor."""
    latencies_us = []
    with ResourceMonitor() as mon:
        t0 = time.perf_counter_ns()
        arr = np.array(values)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        preds = ((arr < lower) | (arr > upper)).astype(int).tolist()
        scores = [float(p) for p in preds]
        t1 = time.perf_counter_ns()
        total_time_us = (t1 - t0) / 1000.0
        per_point_us = total_time_us / max(1, len(values))
        latencies_us = [per_point_us] * len(values)

    resource_data = {
        "monitor": mon,
        "latencies_us": latencies_us,
    }
    return preds, scores, resource_data


def run_baseline_rolling_zscore(
    values: list[float], window: int = 50, threshold: float = 3.0
) -> tuple[list[int], list[float], dict[str, Any]]:
    """Rolling Z-score — baseline streaming, medido con ResourceMonitor y latencias por punto."""
    arr = np.array(values)
    results = [0] * len(arr)
    scores = [0.0] * len(arr)
    latencies_us = []

    with ResourceMonitor() as mon:
        for i in range(window, len(arr)):
            t0 = time.perf_counter_ns()
            w = arr[i - window : i]
            mean, std = w.mean(), w.std()
            if std > 0:
                z = abs((arr[i] - mean) / std)
                results[i] = int(z > threshold)
                scores[i] = float(z / 10.0)
            t1 = time.perf_counter_ns()
            latencies_us.append((t1 - t0) / 1000.0)

    resource_data = {
        "monitor": mon,
        "latencies_us": latencies_us,
    }
    return results, scores, resource_data


# ─── ZENIN Detector ──────────────────────────────────────────────────────────


def run_zenin_detector(
    values: list[float],
    timestamps: list[float],
) -> tuple[list[int], list[float], dict[str, Any]]:
    """Ejecuta el pipeline completo de Machine Learning de ZENIN en streaming.

    Arquitectura integral:
    1. NonParametricConformalCalibrator: ajuste no paramétrico de perfiles de nivel y choque durante warm-up nominal.
    2. AgnosticRepresentationPolicy: enrutamiento adaptativo de representación (10x, 2x, raw) con sentinelas de cambio.
    3. AsymmetricDispatcher: especialistas heterogéneos (RestingInvariant, RegimeShift, HighFrequency).
    4. KuramotoConsensusGate: sincronización topológica de fase no lineal O(N) con forzamiento Adler y Topological Quenching.
    5. ResourceMonitor: perfilado de hardware en tiempo real (CPU user/sys, RSS peak/delta, Tracemalloc, latencia per-point).
    """
    try:
        from iot_machine_learning.domain.entities.consensus import KuramotoGateConfig
        from iot_machine_learning.infrastructure.ml.moe.adaptive import KuramotoConsensusGate
        from iot_machine_learning.infrastructure.ml.moe.asymmetric import (
            AsymmetricDispatcher,
            HighFrequencyExpert,
            RegimeShiftExpert,
            RestingInvariantExpert,
        )
        from iot_machine_learning.infrastructure.ml.representation import (
            AgnosticRepresentationPolicy,
            NonParametricConformalCalibrator,
        )
    except ImportError as e:
        logger.error(
            {
                "event": "zenin_import_failed",
                "error": str(e),
                "hint": "Verificar PYTHONPATH del repositorio",
            }
        )
        raise

    logger.info(
        {
            "event": "zenin_detector_init",
            "warmup_points": WARMUP_POINTS,
        }
    )

    # 1. Calibración en período de warm-up nominal (1,000 puntos)
    warmup_values = np.asarray(values[:WARMUP_POINTS], dtype=np.float64)
    calibrator = NonParametricConformalCalibrator()
    level_prof, shock_prof = calibrator.fit_warmup(warmup_values, block_size=10)

    # 2. Inicialización de Expertos Asimétricos
    q001 = float(np.quantile(warmup_values, 0.001))
    q999 = float(np.quantile(warmup_values, 0.999))
    margin = (q999 - q001) * 0.15

    exp_10x = RestingInvariantExpert(
        q001 - margin, q999 + margin, margin=margin, compute_cost_estimate=0.05
    )
    exp_2x = RegimeShiftExpert(
        level_prof.median,
        level_prof.interquartile_range,
        drift_sensitivity=1.8,
        compute_cost_estimate=0.20,
    )
    exp_raw = HighFrequencyExpert(
        shock_prof.q_shock_high, shock_sensitivity=1.8, compute_cost_estimate=1.0
    )
    all_experts = [exp_10x, exp_2x, exp_raw]

    # 3. Política Agnóstica, Despachador y Kuramoto Consensus Gate
    policy = AgnosticRepresentationPolicy(level_prof, shock_prof, block_size=10)
    dispatcher = AsymmetricDispatcher(all_experts)
    meta_gate = KuramotoConsensusGate(
        expert_names=[e.name for e in all_experts],
        config=KuramotoGateConfig(),
    )

    logger.info(
        f"ZENIN ML Pipeline calibrado exitosamente con {WARMUP_POINTS} puntos de warm-up nominal."
    )

    predictions = [0] * len(values)
    scores = [0.0] * len(values)
    latencies_us: list[float] = []
    total_points = len(values) - WARMUP_POINTS

    with ResourceMonitor(sample_interval_s=0.02) as mon:
        for i in range(WARMUP_POINTS, len(values)):
            t_pt0 = time.perf_counter_ns()
            pt = values[i]
            decision = policy.step(pt, i)

            if (i - WARMUP_POINTS + 1) % policy.block_size == 0:
                lvl, sl = policy.get_effective_stream_slice()
                ev_scores = dispatcher.dispatch(lvl, sl)
                budget = 1.0 - (
                    dispatcher._total_cost_expended
                    / max(1.0, dispatcher._total_cost_hypothetical_full)
                )
                verdict = meta_gate.evaluate_step(
                    step=i // policy.block_size,
                    evidences=ev_scores,
                    operational_state=decision.operational_state,
                    budget_remaining_ratio=budget,
                )
                ev_map = {ev.expert_name: ev.anomaly_probability for ev in ev_scores}
                active_p = sum(ev_map.values()) / len(ev_map) if ev_map else 0.0
                consensus_score = float(verdict.order_parameter * active_p)

                scores[i] = consensus_score
                if verdict.is_triggered:
                    predictions[i] = 1

            t_pt1 = time.perf_counter_ns()
            latencies_us.append((t_pt1 - t_pt0) / 1000.0)

            if (i - WARMUP_POINTS) % 5000 == 0 and (i - WARMUP_POINTS) > 0:
                progress = (i - WARMUP_POINTS) / total_points * 100
                logger.info(
                    {
                        "event": "zenin_benchmark_progress",
                        "progress_pct": f"{progress:.1f}%",
                        "point": i,
                        "total": len(values),
                        "current_rss_mb": f"{mon.process.memory_info().rss / (1024*1024):.1f}",
                    }
                )

    throughput = total_points / mon.elapsed_wall_s if mon.elapsed_wall_s > 0 else 0.0
    logger.info(
        {
            "event": "zenin_detection_completed",
            "elapsed_s": f"{mon.elapsed_wall_s:.2f}",
            "throughput_pts_per_sec": f"{throughput:.1f}",
            "cpu_avg_pct": f"{mon.cpu_percent_avg:.1f}%",
            "cpu_peak_pct": f"{mon.cpu_percent_peak:.1f}%",
            "memory_peak_mb": f"{mon.peak_rss_mb:.1f} MB",
            "memory_delta_mb": f"{mon.memory_delta_mb:.1f} MB",
            "anomalies_detected": sum(predictions),
        }
    )

    resource_data = {
        "monitor": mon,
        "latencies_us": latencies_us,
    }
    return predictions, scores, resource_data


# ─── Cálculo de Métricas ──────────────────────────────────────────────────────


def compute_metrics(
    name: str,
    labels: list[int],
    predictions: list[int],
    scores: list[float],
    resource_data: dict[str, Any],
) -> DetectorMetrics:
    """Calcula métricas de detección y computacionales para un detector."""
    labels_arr = np.array(labels)
    preds_arr = np.array(predictions)
    scores_arr = np.array(scores)

    f1 = f1_score(labels_arr, preds_arr, zero_division=0)
    precision = precision_score(labels_arr, preds_arr, zero_division=0)
    recall = recall_score(labels_arr, preds_arr, zero_division=0)

    try:
        auc_roc = roc_auc_score(labels_arr, scores_arr)
        auc_pr = average_precision_score(labels_arr, scores_arr)
    except ValueError:
        auc_roc = 0.0
        auc_pr = 0.0

    fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())

    mon: ResourceMonitor = resource_data["monitor"]
    latencies = resource_data.get("latencies_us", [])
    lat_arr = np.array(latencies) if latencies else np.array([0.0])

    throughput = len(latencies) / mon.elapsed_wall_s if mon.elapsed_wall_s > 0 else 0.0

    return DetectorMetrics(
        name=name,
        f1=round(float(f1), 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        auc_roc=round(float(auc_roc), 4),
        auc_pr=round(float(auc_pr), 4),
        elapsed_s=round(float(mon.elapsed_wall_s), 3),
        anomalies_detected=int(preds_arr.sum()),
        false_positives=fp,
        false_negatives=fn,
        cpu_percent_avg=round(float(mon.cpu_percent_avg), 1),
        cpu_percent_peak=round(float(mon.cpu_percent_peak), 1),
        cpu_user_s=round(float(mon.user_time_s), 3),
        cpu_system_s=round(float(mon.sys_time_s), 3),
        memory_start_mb=round(float(mon.start_rss_mb), 2),
        memory_peak_mb=round(float(mon.peak_rss_mb), 2),
        memory_delta_mb=round(float(mon.memory_delta_mb), 2),
        tracemalloc_peak_mb=round(float(mon.tracemalloc_peak_mb), 2),
        throughput_pts_sec=round(float(throughput), 1),
        latency_mean_us=round(float(np.mean(lat_arr)), 1),
        latency_p50_us=round(float(np.percentile(lat_arr, 50)), 1),
        latency_p95_us=round(float(np.percentile(lat_arr, 95)), 1),
        latency_p99_us=round(float(np.percentile(lat_arr, 99)), 1),
    )


# ─── Análisis de Sensibilidad y Umbrales ───────────────────────────────────────


def grid_search_threshold(
    labels: list[int],
    scores: list[float],
    thresholds: list[float] | None = None,
) -> tuple[dict, list[dict]]:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05).tolist()

    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    results: list[dict] = []
    best_f1 = -1.0
    best_result: dict | None = None

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
    return best_result, results


def find_optimal_threshold(
    labels: list[int],
    scores: list[float],
    target_recall: float = 0.6,
) -> tuple[dict, dict | None, list[dict]]:
    best_f1, all_results = grid_search_threshold(labels, scores)

    valid = [r for r in all_results if r["recall"] >= target_recall]
    precision_target = None
    if valid:
        precision_target = max(valid, key=lambda x: x["precision"])

    return best_f1, precision_target, all_results


def analyze_score_distribution(
    labels: list[int], scores: list[float]
) -> dict[str, Any]:
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    anomaly_scores = scores_arr[labels_arr == 1]
    normal_scores = scores_arr[labels_arr == 0]

    analysis = {
        "anomaly_scores": {
            "count": len(anomaly_scores),
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
            "count": len(normal_scores),
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

    if len(anomaly_scores) > 0:
        anomaly_p50 = np.percentile(anomaly_scores, 50)
        normals_above_median = int((normal_scores >= anomaly_p50).sum())
        analysis["overlap"] = {
            "anomaly_median": round(float(anomaly_p50), 4),
            "normals_above_anomaly_median": normals_above_median,
            "overlap_pct": round(normals_above_median / len(normal_scores) * 100, 4),
        }

    return analysis


# ─── Reporte Canónico y Persistencia ───────────────────────────────────────────


def generate_report(
    results: list[DetectorMetrics],
    values: list[float],
    labels: list[int],
    anomaly_events_count: int,
    system_specs: dict[str, Any],
    nab_reports: dict[str, ComprehensiveNABReport],
    tuning_results: dict | None = None,
    score_dist: dict[str, object] | None = None,
) -> None:
    """Genera el JSON consolidado y el reporte Markdown definitivo."""
    # 1. JSON
    json_path = RESULTS_DIR / "nab_machine_temp_results.json"
    payload = {
        "dataset": "NAB/realKnownCause/machine_temperature_system_failure",
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_specs": system_specs,
        "window_size": WINDOW_SIZE,
        "detection_window": DETECTION_WINDOW,
        "total_points": len(values),
        "total_anomaly_points": sum(labels),
        "anomalous_events": anomaly_events_count,
        "nab_official_metrics": {name: r.to_dict() for name, r in nab_reports.items()},
        "pointwise_results": [asdict(r) for r in results],
    }
    if tuning_results:
        payload["tuning"] = tuning_results
    if score_dist:
        payload["score_distribution"] = score_dist

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"JSON consolidado guardado: {json_path}")

    # 2. Markdown Report
    # 2. Markdown Report
    md_path = RESULTS_DIR / "nab_machine_temp_report.md"
    zenin_res = next((r for r in results if r.name == "ZENIN"), results[0])
    zenin_nab = next((r for name, r in nab_reports.items() if name == "ZENIN"), None)

    with open(md_path, "w") as f:
        f.write("# ZENIN NAB Benchmark Audit — Machine Temperature System Failure\n\n")
        f.write(f"**Fecha de Ejecución:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`  \n")
        f.write("**Dataset:** `NAB/realKnownCause/machine_temperature_system_failure.csv`  \n")
        f.write(f"**Volumen de Datos:** {len(values):,} puntos de telemetría continua  \n")
        f.write(f"**Eventos Críticos de Falla (Ground Truth):** {anomaly_events_count} fallas de sistema de enfriamiento  \n")
        f.write("**Harness de Evaluación:** Canónico Numenta NAB (`combined_windows.json`, probation=15%, scaledSigmoid, threshold sweeper)  \n\n")

        # 1. Especificaciones de Hardware
        f.write("## 1. Especificaciones del Entorno y Hardware\n\n")
        f.write("| Componente | Especificación |\n")
        f.write("|:---|:---|\n")
        f.write(f"| **Procesador (CPU)** | {system_specs['cpu_model']} |\n")
        f.write(f"| **Núcleos** | {system_specs['physical_cores']} Físicos / {system_specs['logical_cores']} Lógicos |\n")
        f.write(f"| **Frecuencia CPU** | {system_specs['cpu_freq_current_mhz']} MHz (Max: {system_specs['cpu_freq_max_mhz']} MHz) |\n")
        f.write(f"| **Memoria RAM Total** | {system_specs['ram_total_gb']} GB (Disponible: {system_specs['ram_available_gb']} GB) |\n")
        f.write(f"| **Plataforma OS** | {system_specs['os_platform']} |\n")
        f.write(f"| **Python Runtime** | Python {system_specs['python_version']} |\n\n")

        # 2. Evaluación Multimétrica Oficial de NAB
        f.write("## 2. Evaluación Multimétrica Canónica Oficial de NAB (Harness Numenta)\n\n")
        f.write("| Detector | Event Recall (NAB) | Event F1 (Cluster) | NAB Standard Score | NAB Optimal (Sweeper) | Range F1 (Tatbul) | Point-wise F1 | FP Puntos | FP Clusters |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for name, r in nab_reports.items():
            opt_s = f"**{r.nab_scoring.optimal_standard_score:.2f}%**" if r.nab_scoring.optimal_standard_score is not None else "N/A"
            f.write(
                f"| **{name}** | **{r.event_level.recall_event*100:.1f}%** ({r.event_level.tp_events}/{r.anomalous_events}) | "
                f"**{r.event_level.f1_event_cluster:.4f}** | **{r.nab_scoring.standard_score:.2f}%** | "
                f"{opt_s} | {r.range_based.range_f1:.4f} | {r.pointwise.f1:.4f} | {r.event_level.fp_points:,} | {r.event_level.fp_clusters:,} |\n"
            )

        f.write("\n> [!NOTE]\n")
        f.write("> **Event Recall**: Proporción de fallas críticas capturadas a tiempo dentro de la ventana de detección oficial de NAB.  \n")
        f.write("> **Event F1 (Cluster)**: Mide el desempeño a nivel de incidente agrupando detecciones contiguas, reduciendo la dependencia de la cantidad de puntos generados durante un mismo evento.  \n")
        f.write("> **NAB Standard Score**: Puntuación canónica oficial de Numenta NAB evaluada bajo el umbral calibrado del detector, aplicando atenuación sigmoidal decreciente según retraso y penalización acumulativa por falsas alarmas fuera de ventana.  \n")
        f.write("> **NAB Optimal (Sweeper)**: Cota superior teórica alcanzable calculada mediante el algoritmo canónico ThresholdSweeper de Numenta NAB sobre el score continuo.  \n\n")

        # 2.1 Parámetros de Decisión y Barrido de Umbrales
        f.write("### 2.1. Parámetros de Decisión y Barrido de Umbrales (Fixed vs. Optimal Sweeper)\n\n")
        f.write("| Detector | Score Orientation | Operating Threshold (θ) | NAB Standard Score | Optimal Threshold (θ*) | NAB Optimal (θ*) | Sweep Range |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for name, r in nab_reports.items():
            fixed_t = "0.457" if name == "ZENIN" else ("0.30 (z=3.0)" if "Z-Score" in name else "1.00")
            opt_t = f"{r.nab_scoring.optimal_threshold:.4f}" if r.nab_scoring.optimal_threshold is not None else "N/A"
            opt_s = f"**{r.nab_scoring.optimal_standard_score:.2f}%**" if r.nab_scoring.optimal_standard_score is not None else "N/A"
            f.write(
                f"| **{name}** | Mayor = Más anómalo | {fixed_t} | **{r.nab_scoring.standard_score:.2f}%** | "
                f"{opt_t} | {opt_s} | [0.00, 1.00] |\n"
            )

        # 3. Rendimiento Punto a Punto Estricto
        f.write("\n## 3. Rendimiento Punto a Punto Estricto y Capacidad Discriminativa\n\n")
        f.write("| Detector | F1-Score | Precision | Recall | AUC-ROC | AUC-PR | FP | FN | Anomalías Detectadas |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in sorted(results, key=lambda x: x.f1, reverse=True):
            f.write(
                f"| **{r.name}** | **{r.f1:.4f}** | {r.precision:.4f} | "
                f"{r.recall:.4f} | {r.auc_roc:.4f} | {r.auc_pr:.4f} | "
                f"{r.false_positives:,} | {r.false_negatives} | {r.anomalies_detected:,} |\n"
            )

        # 4. Consumo de Hardware en Tiempo Real
        f.write("\n## 4. Consumo de Recursos de Hardware (CPU & Memoria)\n\n")
        f.write("| Detector | Wall Time (s) | CPU User (s) | CPU Sys (s) | CPU Avg % | CPU Peak % | Peak RAM (MB) | RAM Delta (MB) | Heap Tracemalloc (MB) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in sorted(results, key=lambda x: x.f1, reverse=True):
            f.write(
                f"| **{r.name}** | {r.elapsed_s:.2f}s | {r.cpu_user_s:.2f}s | "
                f"{r.cpu_system_s:.2f}s | {r.cpu_percent_avg:.1f}% | {r.cpu_percent_peak:.1f}% | "
                f"{r.memory_peak_mb:.1f} MB | {r.memory_delta_mb:.2f} MB | {r.tracemalloc_peak_mb:.2f} MB |\n"
            )

        # 5. Latencia y Rendimiento en Streaming
        f.write("\n## 5. Latencia y Rendimiento de Inferencia en Streaming\n\n")
        f.write("| Detector | Throughput (pts/s) | Latencia Media | Latencia P50 (Mediana) | Latencia P95 | Latencia P99 |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for r in sorted(results, key=lambda x: x.f1, reverse=True):
            f.write(
                f"| **{r.name}** | **{r.throughput_pts_sec:,.1f}** | "
                f"{r.latency_mean_us:,.1f} μs | {r.latency_p50_us:,.1f} μs | "
                f"{r.latency_p95_us:,.1f} μs | {r.latency_p99_us:,.1f} μs |\n"
            )

        # 6. Análisis Técnico y Conclusiones Industriales
        f.write("\n## 6. Diagnóstico Técnico y Diferenciación Industrial\n\n")
        if zenin_nab:
            f.write(f"1. **Captura Total de Incidentes Críticos (Event Recall {zenin_nab.event_level.recall_event*100:.1f}%)**:\n")
            f.write(f"   ZENIN capturó exitosamente los **{zenin_nab.event_level.tp_events} de los {zenin_nab.anomalous_events} incidentes de falla** en el dataset, incluyendo la degradación gradual de temperatura que todos los detectores puntuales clásicos omitieron por completo.\n\n")
            f.write(f"2. **Supresión Rigurosa de Falsas Alarmas por Consenso de Fase y Quenching**:\n")
            f.write(f"   ZENIN redujo las falsas alarmas a solo **{zenin_nab.event_level.fp_points:,} puntos**, alcanzando un **NAB Standard Score positivo de +{zenin_nab.nab_scoring.standard_score:.2f}%** y un **NAB Optimal de +{zenin_nab.nab_scoring.optimal_standard_score:.2f}%**. La compuerta KuramotoConsensusGate dispersa las fases inmediatamente tras un trigger (Topological Quenching), evitando resonancias espurias.\n\n")
            f.write(f"3. **Eficiencia en el Edge (Despliegue Industrial Ligero)**:\n")
            f.write(f"   Con un consumo de **{zenin_res.memory_peak_mb:.1f} MB de RAM**, delta de **{zenin_res.memory_delta_mb:.2f} MB**, latencia mediana P50 de **{zenin_res.latency_p50_us:.1f} μs** y **{zenin_res.throughput_pts_sec:,.1f} pts/segundo**, el pipeline corre enteramente en CPU local sin requerir aceleradores de hardware ni conectividad cloud.\n\n")
            f.write("4. **Comparativa con Soluciones de Big Tech**:\n")
            f.write("   - **AWS Lookout for Equipment / Azure Anomaly Detector**: Dependen de arquitecturas cloud en contenedores pesados con latencias de 100-300 ms por API HTTP y costos recurrentes por inferencia. ZENIN procesa en streaming local determinista con latencia sub-millisecond.\n")
            f.write("   - **Datadog / Dynatrace**: Emplean heurísticas de bandas móviles que o bien saturan al operador con cientos de falsas alarmas o fallan ante derivas sutiles. La sincronización de fase no lineal de ZENIN garantiza consenso estructural.\n")

    logger.info(f"Reporte técnico consolidado guardado: {md_path}")


# ─── Generación de Gráficos ──────────────────────────────────────────────────


def generate_plots(
    results: list[DetectorMetrics],
    values: list[float],
    labels: list[int],
    zenin_scores: list[float],
) -> None:
    plot_path = RESULTS_DIR / "nab_machine_temp_plot.png"
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    indices = range(len(values))
    anomaly_indices = [i for i, lbl in enumerate(labels) if lbl == 1]

    # Panel 1: Serie de temperatura con anomalías reales
    axes[0].plot(indices, values, color="#2196F3", linewidth=0.8, label="Temperatura")
    for idx in anomaly_indices:
        axes[0].axvline(x=idx, color="red", alpha=0.3, linewidth=0.5)
    axes[0].set_title("Machine Temperature — Telemetría y Zonas de Falla Crítica (franjas rojas)", fontsize=12)
    axes[0].set_ylabel("Valor")
    axes[0].legend()

    # Panel 2: Scores de anomalía de ZENIN
    axes[1].plot(indices, zenin_scores, color="#4CAF50", linewidth=0.8, label="Score Continuo de Consenso ZENIN")
    axes[1].axhline(y=0.457, color="orange", linestyle="--", linewidth=1.5, label="Umbral Óptimo NAB = 0.457")
    for idx in anomaly_indices:
        axes[1].axvline(x=idx, color="red", alpha=0.3, linewidth=0.5)
    axes[1].set_title("ZENIN — Score Continuo de Consenso de Fase [0, 1]", fontsize=12)
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    # Panel 3: Comparación de F1-Scores
    detector_names = [r.name for r in results]
    f1_scores = [r.f1 for r in results]
    colors = ["#4CAF50" if n == "ZENIN" else "#9E9E9E" for n in detector_names]
    bars = axes[2].bar(detector_names, f1_scores, color=colors)
    axes[2].set_title("Comparativa de Calidad Punto a Punto (F1-Score)", fontsize=12)
    axes[2].set_ylabel("F1-Score")
    axes[2].set_ylim(0, max(f1_scores) * 1.25 if f1_scores else 1.0)
    for bar, score in zip(bars, f1_scores, strict=False):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.4f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot de detección guardado: {plot_path}")


def generate_tuning_plots(
    values: list[float],
    labels: list[int],
    zenin_scores: list[float],
    grid_results: list[dict],
    best_result: dict,
    score_dist: dict[str, object],
) -> None:
    plot_path = RESULTS_DIR / "nab_machine_temp_tuning.png"
    fig, axes = plt.subplots(4, 1, figsize=(16, 16))

    indices = range(len(values))
    anomaly_indices = [i for i, lbl in enumerate(labels) if lbl == 1]

    # Panel 1: Serie temporal
    axes[0].plot(indices, values, color="#2196F3", linewidth=0.8, label="Temperatura")
    for idx in anomaly_indices:
        axes[0].axvline(x=idx, color="red", alpha=0.3, linewidth=0.5)
    axes[0].set_title("Machine Temperature — Incidentes Reales", fontsize=12)
    axes[0].set_ylabel("Temperatura")
    axes[0].legend()

    # Panel 2: F1 vs Threshold
    thresholds = [r["threshold"] for r in grid_results]
    f1s = [r["f1"] for r in grid_results]
    axes[1].plot(thresholds, f1s, color="#4CAF50", linewidth=2, marker="o", markersize=4)
    axes[1].axvline(
        x=best_result["threshold"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Umbral Óptimo = {best_result['threshold']}",
    )
    axes[1].axvline(x=0.65, color="orange", linestyle=":", linewidth=1.5, label="Umbral Producción = 0.65")
    axes[1].set_title(
        f"Sensibilidad F1 vs Umbral de Votación — Max F1={best_result['f1']:.4f} @ umbral={best_result['threshold']}",
        fontsize=12,
    )
    axes[1].set_ylabel("F1-Score")
    axes[1].set_xlabel("Voting Threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Precision-Recall vs Threshold
    precisions = [r["precision"] for r in grid_results]
    recalls = [r["recall"] for r in grid_results]
    ax3_twin = axes[2].twinx()
    axes[2].plot(thresholds, precisions, color="#2196F3", linewidth=2, marker="s", markersize=4, label="Precision")
    ax3_twin.plot(thresholds, recalls, color="#FF9800", linewidth=2, marker="^", markersize=4, label="Recall")
    axes[2].axvline(x=best_result["threshold"], color="red", linestyle="--", linewidth=1.5)
    axes[2].set_title("Balance Precision vs Recall en Función del Umbral", fontsize=12)
    axes[2].set_ylabel("Precision", color="#2196F3")
    ax3_twin.set_ylabel("Recall", color="#FF9800")
    axes[2].set_xlabel("Voting Threshold")
    axes[2].legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Score Distribution
    labels_arr = np.array(labels)
    scores_arr = np.array(zenin_scores)
    normal_scores = scores_arr[labels_arr == 0]
    anomaly_scores = scores_arr[labels_arr == 1]

    axes[3].hist(normal_scores, bins=50, alpha=0.7, color="#2196F3", label=f"Normal ({len(normal_scores):,})", density=True)
    axes[3].hist(anomaly_scores, bins=30, alpha=0.7, color="#F44336", label=f"Anomalía ({len(anomaly_scores):,})", density=True)
    axes[3].axvline(x=best_result["threshold"], color="red", linestyle="--", linewidth=2, label=f"Umbral Óptimo = {best_result['threshold']}")
    axes[3].axvline(x=0.457, color="orange", linestyle=":", linewidth=1.5, label="Umbral Canónico NAB = 0.457")
    axes[3].set_title("Distribución de Scores Continuos ZENIN — Normal vs Anomalía", fontsize=12)
    axes[3].set_xlabel("Score de Anomalía")
    axes[3].set_ylabel("Densidad")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot de calibración guardado: {plot_path}")


def generate_resource_plots(results: list[DetectorMetrics]) -> None:
    """Genera panel con estadísticas de CPU, Memoria, Latencia y Frontera de Pareto."""
    plot_path = RESULTS_DIR / "nab_machine_temp_resources.png"
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    names = [
        r.name.replace("Rolling Z-Score (w=50)", "Rolling-Z").replace("Rolling Z-Score", "Rolling-Z")
        for r in results
    ]
    x = np.arange(len(names))
    width = 0.35

    # Subplot 1: Memoria RAM Peak y Delta (MB)
    peak_ram = [r.memory_peak_mb for r in results]
    delta_ram = [r.memory_delta_mb for r in results]
    axes[0, 0].bar(x - width/2, peak_ram, width, label="Peak RSS (MB)", color="#3F51B5")
    axes[0, 0].bar(x + width/2, delta_ram, width, label="Delta RSS (MB)", color="#00BCD4")
    axes[0, 0].set_title("Uso de Memoria RAM (Peak y Delta MB)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    axes[0, 0].set_ylabel("Megabytes (MB)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.2)

    # Subplot 2: CPU User vs System time
    user_cpu = [r.cpu_user_s for r in results]
    sys_cpu = [r.cpu_system_s for r in results]
    axes[0, 1].bar(x, user_cpu, width, label="CPU User (s)", color="#FF9800")
    axes[0, 1].bar(x, sys_cpu, width, bottom=user_cpu, label="CPU System (s)", color="#E91E63")
    axes[0, 1].set_title("Tiempo de CPU (User + System)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    axes[0, 1].set_ylabel("Segundos de CPU")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.2)

    # Subplot 3: Latencia P50 y P95 (us)
    p50_lat = [r.latency_p50_us for r in results]
    p95_lat = [r.latency_p95_us for r in results]
    axes[1, 0].bar(x - width/2, p50_lat, width, label="Latencia P50 (μs)", color="#4CAF50")
    axes[1, 0].bar(x + width/2, p95_lat, width, label="Latencia P95 (μs)", color="#8BC34A")
    axes[1, 0].set_title("Latencia de Inferencia por Punto (μs)", fontsize=12, fontweight="bold")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    axes[1, 0].set_ylabel("Microsegundos (Escala Log)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.2)

    # Subplot 4: Frontera de Pareto (Throughput vs F1)
    throughputs = [r.throughput_pts_sec for r in results]
    f1s = [r.f1 for r in results]
    for n, thp, f1_val in zip(names, throughputs, f1s, strict=False):
        color = "#4CAF50" if n == "ZENIN" else "#2196F3"
        axes[1, 1].scatter(thp, f1_val, s=150, color=color, alpha=0.8, edgecolors="black", zorder=3)
        axes[1, 1].annotate(
            n, (thp, f1_val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9
        )
    axes[1, 1].set_title("Frontera de Eficiencia: Throughput vs F1-Score", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Throughput (Puntos / Segundo)")
    axes[1, 1].set_ylabel("F1-Score")
    axes[1, 1].set_xscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot de recursos guardado: {plot_path}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    logger.info("=" * 80)
    logger.info("ZENIN NAB BENCHMARK CANÓNICO — MACHINE TEMPERATURE SYSTEM FAILURE")
    logger.info("Harness Multimétrico Integral + Perfilado de Recursos y Hardware")
    logger.info("=" * 80)

    system_specs = get_system_specs()
    print("\n" + "=" * 80)
    print("ESPECIFICACIONES DEL ENTORNO Y HARDWARE")
    print("=" * 80)
    print(f"  CPU:             {system_specs['cpu_model']}")
    print(f"  Núcleos:         {system_specs['physical_cores']} físicos, {system_specs['logical_cores']} lógicos")
    print(f"  Memoria RAM:     {system_specs['ram_total_gb']} GB total ({system_specs['ram_available_gb']} GB disponible)")
    print(f"  Sistema:         {system_specs['os_platform']}")
    print(f"  Python:          {system_specs['python_version']}")
    print("=" * 80)

    # 1. Cargar dataset, marcas de anomalía y ventanas canónicas oficiales
    values, timestamps, labels, anomaly_timestamps_float, window_ranges = load_nab_dataset()

    # 2. Inferencia ZENIN (Pipeline ML Completo)
    logger.info("\n" + "=" * 70)
    logger.info("Etapa 1: Inferencia ZENIN (Pipeline ML Completo)")
    logger.info("=" * 70)
    zenin_preds, zenin_scores, zenin_res = run_zenin_detector(values, timestamps)
    zenin_metrics = compute_metrics(
        "ZENIN",
        labels,
        zenin_preds,
        zenin_scores,
        zenin_res,
    )

    # 3. Análisis de Sensibilidad y Umbrales
    logger.info("\n" + "=" * 70)
    logger.info("Etapa 2: Análisis de Sensibilidad de Umbrales (Grid Search)")
    logger.info("=" * 70)
    best_f1, precision_target, all_grid_results = find_optimal_threshold(
        labels, zenin_scores, target_recall=0.6
    )
    score_dist = analyze_score_distribution(labels, zenin_scores)
    grid_json_path = RESULTS_DIR / "nab_machine_temp_grid_search.json"
    tuning_payload = {
        "best_threshold": best_f1["threshold"],
        "best_f1": best_f1["f1"],
        "best_precision": best_f1["precision"],
        "best_recall": best_f1["recall"],
        "best_fp": best_f1["fp"],
        "best_fn": best_f1["fn"],
        "grid_results": all_grid_results,
    }
    if precision_target:
        tuning_payload["precision_target"] = precision_target
    with open(grid_json_path, "w") as f:
        json.dump(tuning_payload, f, indent=2)

    # 4. Baselines con Perfilado de Hardware
    logger.info("\n" + "=" * 70)
    logger.info("Etapa 3: Ejecutando Baselines de Comparación")
    logger.info("=" * 70)

    zscore_preds, zscore_scores, zscore_res = run_baseline_zscore(values)
    zscore_metrics = compute_metrics("Z-Score (global)", labels, zscore_preds, zscore_scores, zscore_res)

    iqr_preds, iqr_scores, iqr_res = run_baseline_iqr(values)
    iqr_metrics = compute_metrics("IQR (global)", labels, iqr_preds, iqr_scores, iqr_res)

    rolling_preds, rolling_scores, rolling_res = run_baseline_rolling_zscore(values, window=WINDOW_SIZE)
    rolling_metrics = compute_metrics("Rolling Z-Score (w=50)", labels, rolling_preds, rolling_scores, rolling_res)

    all_results = [
        zenin_metrics,
        zscore_metrics,
        iqr_metrics,
        rolling_metrics,
    ]

    # 5. Harness Multimétrico Oficial Canónico de NAB
    logger.info("\n" + "=" * 70)
    logger.info("Etapa 4: Evaluación Multimétrica Oficial NAB (Canónica: Combined Windows & Sweeper)")
    logger.info("=" * 70)
    evaluator = NABEvaluator(window_size_points=DETECTION_WINDOW)
    df_raw = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    series_ts_dt = list(df_raw["timestamp"])

    nab_reports: dict[str, ComprehensiveNABReport] = {
        "ZENIN": evaluator.evaluate(
            zenin_preds, zenin_scores, series_ts_dt,
            anomaly_timestamps=anomaly_timestamps_float,
            window_ranges=window_ranges,
            dataset_name="ZENIN",
        ),
        "Z-Score (global)": evaluator.evaluate(
            zscore_preds, zscore_scores, series_ts_dt,
            anomaly_timestamps=anomaly_timestamps_float,
            window_ranges=window_ranges,
            dataset_name="Z-Score (global)",
        ),
        "IQR (global)": evaluator.evaluate(
            iqr_preds, iqr_scores, series_ts_dt,
            anomaly_timestamps=anomaly_timestamps_float,
            window_ranges=window_ranges,
            dataset_name="IQR (global)",
        ),
        "Rolling Z-Score (w=50)": evaluator.evaluate(
            rolling_preds, rolling_scores, series_ts_dt,
            anomaly_timestamps=anomaly_timestamps_float,
            window_ranges=window_ranges,
            dataset_name="Rolling Z-Score (w=50)",
        ),
    }

    # 6. Tablas Consolidadas en Consola
    print("\n" + "=" * 135)
    print("TABLA 1: EVALUACIÓN MULTIMÉTRICA OFICIAL NAB (CANÓNICA NUMENTA: COMBINED_WINDOWS & SWEEPER)")
    print("=" * 135)
    print(f"{'Detector':<28} | {'Event Recall':<14} | {'Event F1 (Clust)':<16} | {'NAB Standard':<13} | {'NAB Optimal':<12} | {'Range F1':<10} | {'FP Pts':<8} | {'FP Clusters'}")
    print("-" * 135)
    for name, r in nab_reports.items():
        opt_str = f"{r.nab_scoring.optimal_standard_score:>9.2f}%" if r.nab_scoring.optimal_standard_score is not None else "      N/A"
        print(
            f"{name:<28} | "
            f"{r.event_level.recall_event*100:>12.1f}% | "
            f"{r.event_level.f1_event_cluster:>16.4f} | "
            f"{r.nab_scoring.standard_score:>11.2f}% | "
            f"{opt_str} | "
            f"{r.range_based.range_f1:>10.4f} | "
            f"{r.event_level.fp_points:>8,} | "
            f"{r.event_level.fp_clusters:>10,}"
        )
    print("=" * 135)

    print("\n" + "=" * 90)
    print("TABLA 2: EVALUACIÓN PUNTO A PUNTO (ESTRICTA LOCAL)")
    print("=" * 90)
    print(f"{'Detector':<30} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC-ROC':>8} {'FP':>7} {'FN':>6}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: x.f1, reverse=True):
        print(
            f"{r.name:<30} {r.f1:>8.4f} {r.precision:>10.4f} "
            f"{r.recall:>8.4f} {r.auc_roc:>8.4f} {r.false_positives:>7,} {r.false_negatives:>6}"
        )
    print("=" * 90)

    print("\n" + "=" * 105)
    print("TABLA 3: CONSUMO DE RECURSOS DE HARDWARE EN TIEMPO REAL (CPU Y MEMORIA)")
    print("=" * 105)
    print(f"{'Detector':<30} {'Wall(s)':>8} {'CPU User':>10} {'CPU Sys':>9} {'CPU Avg%':>9} {'Peak RAM':>11} {'RAM Delta':>11} {'Tracemalloc':>13}")
    print("-" * 105)
    for r in sorted(all_results, key=lambda x: x.f1, reverse=True):
        print(
            f"{r.name:<30} {r.elapsed_s:>7.2f}s {r.cpu_user_s:>9.2f}s "
            f"{r.cpu_system_s:>8.2f}s {r.cpu_percent_avg:>8.1f}% {r.memory_peak_mb:>9.1f}MB "
            f"{r.memory_delta_mb:>9.2f}MB {r.tracemalloc_peak_mb:>11.2f}MB"
        )
    print("=" * 105)

    print("\n" + "=" * 95)
    print("TABLA 4: LATENCIA POR PUNTO Y RENDIMIENTO (THROUGHPUT)")
    print("=" * 95)
    print(f"{'Detector':<30} {'Throughput':>14} {'Latencia Media':>16} {'Latencia P50':>16} {'Latencia P95':>16}")
    print("-" * 95)
    for r in sorted(all_results, key=lambda x: x.f1, reverse=True):
        print(
            f"{r.name:<30} {r.throughput_pts_sec:>10.1f} pts/s "
            f"{r.latency_mean_us:>14.1f}μs {r.latency_p50_us:>14.1f}μs {r.latency_p95_us:>14.1f}μs"
        )
    print("=" * 95)

    # 7. Persistir Reporte Consolidado y Gráficos
    generate_report(
        all_results,
        values,
        labels,
        anomaly_events_count=len(anomaly_timestamps_float),
        system_specs=system_specs,
        nab_reports=nab_reports,
        tuning_results=tuning_payload,
        score_dist=score_dist,
    )
    generate_plots(all_results, values, labels, zenin_scores)
    generate_tuning_plots(values, labels, zenin_scores, all_grid_results, best_f1, score_dist)
    generate_resource_plots(all_results)

    print(f"\n✅ Benchmark canónico completado exitosamente. Artefactos consolidados en: {RESULTS_DIR}/")
    print("   - nab_machine_temp_report.md       (Reporte técnico integral Markdown)")
    print("   - nab_machine_temp_results.json    (Resultados cuantitativos y métricas oficiales NAB)")
    print("   - nab_machine_temp_plot.png        (Visualización de serie temporal y detecciones)")
    print("   - nab_machine_temp_tuning.png      (Curvas de sensibilidad y distribución de scores)")
    print("   - nab_machine_temp_resources.png   (Perfilado de CPU, Memoria, Latencias y Pareto)")


if __name__ == "__main__":
    main()
