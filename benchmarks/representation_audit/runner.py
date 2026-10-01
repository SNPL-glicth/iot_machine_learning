"""Runner principal para la Auditoría de Pérdida de Representación de ZENIN.

Ejecuta el experimento científico sobre machine_temperature_system_failure.csv
y responde las 5+1 preguntas fundamentales:
1. ¿La anomalía sigue siendo detectable después de transformar la señal?
2. ¿Cuánto cambia su localización temporal (delay/onset)?
3. ¿Cuánto aumenta/disminuye la separabilidad respecto al comportamiento normal?
4. ¿Qué transformaciones destruyen evidencia?
5. ¿Cuánto cuesta computacionalmente cada representación?
6. ¿Qué información barata permite detectar que NO debemos comprimir?
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Configurar path para imports locales
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.representation_audit.event_analysis import (
    CanonicalWindow,
    EventAuditSummary,
    evaluate_events_for_signal,
    evaluate_pre_compression_guard,
)
from benchmarks.representation_audit.metrics import (
    DetectorSeparability,
    StatisticalSeparability,
    compute_detector_separability,
    compute_statistical_separability,
)
from benchmarks.representation_audit.report import (
    format_markdown_report,
    save_audit_json,
)
from benchmarks.representation_audit.resource_measurement import (
    ResourceBenchmarkResult,
    benchmark_transformation,
)
from benchmarks.representation_audit.transformations import (
    TransformedSignal,
    to_downsample,
    to_envelope,
    to_raw,
    to_residual,
)

logging.basicConfig(level=logging.WARNING)

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BENCHMARK_DIR.parent / "data" / "NAB"
DEFAULT_CSV_PATH = (
    DEFAULT_DATA_DIR / "data" / "realKnownCause" / "machine_temperature_system_failure.csv"
)
DEFAULT_WINDOWS_PATH = DEFAULT_DATA_DIR / "labels" / "combined_windows.json"
DEFAULT_OUT_JSON = BENCHMARK_DIR / "results" / "representation_audit.json"


class CanonicalStreamingDetector:
    """Detector streaming robusto idéntico para todas las representaciones.

    Mantiene una línea base nominal calculada sobre el periodo de warmup
    y evalúa desviaciones relativas normalizadas hacia un score continuo in [0, 1].
    Garantiza que la comparación entre representaciones sea estrictamente justa.
    """

    def __init__(self, warmup_points: int = 1000) -> None:
        self.warmup_points = warmup_points
        self.baseline_median: float = 0.0
        self.baseline_mad: float = 1.0

    def fit_warmup(self, values: np.ndarray) -> None:
        n_warm = min(self.warmup_points, len(values))
        warm_vals = values[:n_warm]
        self.baseline_median = float(np.median(warm_vals))
        deviations = np.abs(warm_vals - self.baseline_median)
        mad = float(np.median(deviations))
        # Normalizar MAD hacia desviación estándar equivalente para gaussianas (1.4826)
        self.baseline_mad = max(mad * 1.4826, 1e-4)

    def score_stream(self, values: np.ndarray) -> np.ndarray:
        """Calcula score continuo in [0, 1] usando tanh(0.35 * |z|)."""
        z_scores = np.abs(values - self.baseline_median) / self.baseline_mad
        # tanh(0.35 * z): z=3 -> 0.78, z=4 -> 0.88, z=5 -> 0.94
        scores = np.tanh(0.35 * z_scores)
        return scores


def load_dataset(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carga serie temporal completa de NAB."""
    timestamps_raw: list[str] = []
    timestamps_sec: list[float] = []
    values: list[float] = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row["timestamp"]
            val = float(row["value"])
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            timestamps_raw.append(ts_str)
            timestamps_sec.append(dt.timestamp())
            values.append(val)

    return (
        np.array(values, dtype=np.float64),
        np.array(timestamps_sec, dtype=np.float64),
        timestamps_raw,
    )


def load_canonical_windows(
    windows_path: Path,
    key: str,
    timestamps_raw: list[str],
    timestamps_sec: np.ndarray,
) -> list[CanonicalWindow]:
    """Carga y mapea ventanas canónicas desde el JSON de NAB."""
    with open(windows_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_windows = data.get(key, [])
    canonical: list[CanonicalWindow] = []

    for idx, (ws_str, we_str) in enumerate(raw_windows, 1):
        dt_s = datetime.strptime(ws_str[:19], "%Y-%m-%d %H:%M:%S")
        dt_e = datetime.strptime(we_str[:19], "%Y-%m-%d %H:%M:%S")
        s_sec = dt_s.timestamp()
        e_sec = dt_e.timestamp()

        # Encontrar índices en timestamps_raw
        raw_s_idx = next(i for i, ts in enumerate(timestamps_raw) if ts >= ws_str[:19])
        raw_e_idx = next(
            i for i in range(len(timestamps_raw) - 1, -1, -1) if timestamps_raw[i] <= we_str[:19]
        )

        canonical.append(
            CanonicalWindow(
                event_id=idx,
                start_str=ws_str[:19],
                end_str=we_str[:19],
                start_sec=s_sec,
                end_sec=e_sec,
                raw_start_idx=raw_s_idx,
                raw_end_idx=raw_e_idx,
            )
        )

    return canonical


def run_audit(
    csv_path: Path = DEFAULT_CSV_PATH,
    windows_path: Path = DEFAULT_WINDOWS_PATH,
    out_json: Path = DEFAULT_OUT_JSON,
    detection_threshold: float = 0.65,
) -> None:
    """Ejecuta el protocolo experimental completo."""
    print("=" * 80)
    print("ZENIN EXPERIMENTAL AUDIT: REPRESENTATION LOSS AND INFORMATION PRESERVATION")
    print("=" * 80)
    print(f"Cargando dataset: {csv_path}")

    values, timestamps_sec, timestamps_raw = load_dataset(csv_path)
    n_total = len(values)
    print(f"Total puntos cargados: {n_total:,}")

    rel_key = "realKnownCause/machine_temperature_system_failure.csv"
    canonical_windows = load_canonical_windows(
        windows_path, rel_key, timestamps_raw, timestamps_sec
    )
    print(f"Ventanas canónicas cargadas: {len(canonical_windows)}")
    for w in canonical_windows:
        print(f"  - Evento {w.event_id}: {w.start_str} a {w.end_str} ({w.raw_duration_points} pts)")

    # Definir la suite de representaciones
    transform_configs = [
        ("R0_raw", to_raw, {}),
        ("R1_residual_w50", to_residual, {"window": 50}),
        ("R2_downsample_2x_decimate", to_downsample, {"factor": 2, "mode": "decimate"}),
        ("R2b_downsample_2x_mean", to_downsample, {"factor": 2, "mode": "mean"}),
        ("R3_downsample_5x_decimate", to_downsample, {"factor": 5, "mode": "decimate"}),
        ("R3b_downsample_5x_mean", to_downsample, {"factor": 5, "mode": "mean"}),
        ("R4_downsample_10x_decimate", to_downsample, {"factor": 10, "mode": "decimate"}),
        ("R4b_downsample_10x_mean", to_downsample, {"factor": 10, "mode": "mean"}),
        ("R5_envelope_spread_w10", to_envelope, {"window": 10, "metric": "spread"}),
    ]

    resource_results: list[ResourceBenchmarkResult] = []
    signals: list[TransformedSignal] = []

    print("\nEjecutando y perfilando transformaciones...")
    for name, fn, kwargs in transform_configs:
        sig, res = benchmark_transformation(
            fn, values, timestamps_sec, timestamps_raw, n_runs=3, **kwargs
        )
        signals.append(sig)
        resource_results.append(res)
        print(
            f"  [OK] {res.name:<28}: {res.n_output_points:>6} pts ({res.compression_ratio:>4.1f}x) | {res.elapsed_ms:>6.2f} ms | {res.peak_memory_kb:>6.1f} KB"
        )

    # Identificar máscara de ground truth binario en el espacio crudo
    raw_labels = np.zeros(n_total, dtype=int)
    for w in canonical_windows:
        raw_labels[w.raw_start_idx : w.raw_end_idx + 1] = 1

    # Estructuras de resultados
    stat_results: dict[str, StatisticalSeparability] = {}
    det_results: dict[str, DetectorSeparability] = {}
    event_summaries: dict[str, EventAuditSummary] = {}

    raw_baseline_temporal: dict[int, Any] | None = None
    raw_baseline_peaks: dict[int, float] | None = None

    print("\nEvaluando detectabilidad, separabilidad y preservación de eventos...")
    for sig in signals:
        # 1. Crear etiquetas binarias mapeadas para esta señal transformada
        sig_labels = np.zeros(sig.n_points, dtype=int)
        for w in canonical_windows:
            in_win = (sig.timestamps_sec >= w.start_sec) & (sig.timestamps_sec <= w.end_sec)
            sig_labels[in_win] = 1

        # 2. Separabilidad Estadística pura (distribución de valores)
        norm_vals = sig.values[sig_labels == 0]
        ano_vals = sig.values[sig_labels == 1]
        stat_sep = compute_statistical_separability(norm_vals, ano_vals)
        stat_results[sig.name] = stat_sep

        # 3. Inferencia con el Detector Streaming Canónico
        detector = CanonicalStreamingDetector(warmup_points=min(1000, sig.n_points // 4))
        detector.fit_warmup(sig.values)
        scores = detector.score_stream(sig.values)

        # 4. Separabilidad del Detector (ROC-AUC, PR-AUC, Márgenes)
        det_sep = compute_detector_separability(scores, sig_labels)
        det_results[sig.name] = det_sep

        # 5. Auditoría de Eventos Canónicos y Desplazamiento Temporal
        ev_summary = evaluate_events_for_signal(
            signal=sig,
            scores=scores,
            canonical_windows=canonical_windows,
            threshold=detection_threshold,
            raw_baseline_temporal=raw_baseline_temporal,
            raw_baseline_peaks=raw_baseline_peaks,
        )
        event_summaries[sig.name] = ev_summary

        # Registrar baseline si es R0_raw
        if sig.name == "R0_raw":
            raw_baseline_temporal = ev_summary.temporal_preservation
            raw_baseline_peaks = {
                e_id: ep.peak_score for e_id, ep in ev_summary.event_preservation.items()
            }

    # 6. Evaluación de la Pregunta 6: El Guardián Barato de No-Compresión
    print("\nEvaluando la Pregunta 6 (Pre-Compression Guard)...")
    guard_results = evaluate_pre_compression_guard(
        values=values,
        canonical_windows=canonical_windows,
        block_size=10,
        k_sigma_threshold=2.8,
    )
    for g in guard_results:
        print(
            f"  Guard: {g.trigger_name:<25} | Protección: {g.true_event_protection_rate:.0f}% | Frenados Normal: {g.false_guard_rate_normal:.1f}% | Ahorro: {g.data_savings_achieved_pct:.1f}%"
        )

    # Generar reportes
    report_md = format_markdown_report(
        resource_results=resource_results,
        stat_results=stat_results,
        det_results=det_results,
        event_summaries=event_summaries,
        guard_results=guard_results,
    )

    print("\n" + "=" * 80)
    print(report_md)
    print("=" * 80)

    save_audit_json(
        out_path=out_json,
        resource_results=resource_results,
        stat_results=stat_results,
        det_results=det_results,
        event_summaries=event_summaries,
        guard_results=guard_results,
    )
    print(f"\nReporte JSON estructurado guardado exitosamente en: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ZENIN Representation Loss & Information Preservation Audit"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS_PATH)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    run_audit(
        csv_path=args.csv,
        windows_path=args.windows,
        out_json=args.out_json,
        detection_threshold=args.threshold,
    )
