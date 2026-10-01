"""Análisis detallado por evento canónico, degradación temporal y guardián de no-compresión.

1. Evalúa las 4 ventanas canónicas de falla de NAB en cada representación:
   - Evento 1: Variación térmica / anomalía temprana
   - Evento 2: Caída extrema / shock térmico (2.08°)
   - Evento 3: Desplazamiento persistente de régimen (64.69°)
   - Evento 4: Colapso catastrófico del sistema (25.89°)

2. Evalúa clusters de falsos positivos fuera de ventana.

3. Evalúa la Pregunta 6: El Guardián Barato de No-Compresión (Pre-compression Guard).
   ¿Qué métrica O(1) detecta que una ventana local contiene un shock o transición
   crítica y por ende NO DEBE ser comprimida?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import EventPreservationMetrics, TemporalPreservationMetrics
from .transformations import TransformedSignal


@dataclass(frozen=True)
class CanonicalWindow:
    """Definición de una ventana canónica de falla en NAB."""

    event_id: int
    start_str: str
    end_str: str
    start_sec: float
    end_sec: float
    raw_start_idx: int
    raw_end_idx: int

    @property
    def raw_duration_points(self) -> int:
        return self.raw_end_idx - self.raw_start_idx + 1

    @property
    def duration_seconds(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class EventAuditSummary:
    """Resumen de análisis de eventos para una representación."""

    name: str
    event_preservation: dict[int, EventPreservationMetrics]
    temporal_preservation: dict[int, TemporalPreservationMetrics]
    fp_points_total: int
    fp_clusters_total: int
    max_fp_score: float
    raw_points_processed: int
    transformed_points_processed: int


@dataclass(frozen=True)
class PreCompressionGuardResult:
    """Evaluación del guardián barato de no-compresión (Pregunta 6)."""

    trigger_name: str
    events_protected: dict[int, bool]  # ¿Activó antes o durante cada evento?
    false_guard_rate_normal: float  # % de bloques normales donde frenó la compresión
    true_event_protection_rate: float  # % de eventos canónicos protegidos
    data_savings_achieved_pct: float  # % de bloques que sí pudieron comprimirse


def evaluate_events_for_signal(
    signal: TransformedSignal,
    scores: np.ndarray,
    canonical_windows: list[CanonicalWindow],
    threshold: float = 0.65,
    raw_baseline_temporal: dict[int, TemporalPreservationMetrics] | None = None,
    raw_baseline_peaks: dict[int, float] | None = None,
) -> EventAuditSummary:
    """Evalúa preservación y temporización de los 4 eventos sobre una señal transformada."""
    n_pts = len(scores)
    in_window_ids = np.zeros(n_pts, dtype=int)

    # Identificar qué puntos transformados caen en cada ventana canónica según sus timestamps
    for win in canonical_windows:
        mask = (signal.timestamps_sec >= win.start_sec) & (signal.timestamps_sec <= win.end_sec)
        in_window_ids[mask] = win.event_id

    event_pres: dict[int, EventPreservationMetrics] = {}
    temp_pres: dict[int, TemporalPreservationMetrics] = {}

    for win in canonical_windows:
        win_mask = in_window_ids == win.event_id
        win_scores = scores[win_mask]

        if len(win_scores) == 0:
            # Ventana completamente obliterada por la transformación
            event_pres[win.event_id] = EventPreservationMetrics(
                event_id=win.event_id,
                detected=False,
                window_coverage_pct=0.0,
                peak_score=0.0,
                peak_retention_ratio=0.0,
                mean_window_score=0.0,
                total_window_energy=0.0,
            )
            temp_pres[win.event_id] = TemporalPreservationMetrics(
                event_id=win.event_id,
                detected=False,
                onset_idx_transformed=None,
                onset_idx_raw_equiv=None,
                delay_seconds=None,
                delay_points_equiv=None,
                delta_delay_vs_raw_sec=None,
                window_elapsed_pct=None,
            )
            continue

        peak_sc = float(np.max(win_scores))
        mean_sc = float(np.mean(win_scores))
        total_energy = float(np.sum(win_scores))
        above_thresh = win_scores >= threshold
        detected = bool(np.any(above_thresh))
        coverage_pct = float(np.mean(above_thresh) * 100.0)

        raw_peak = raw_baseline_peaks.get(win.event_id, peak_sc) if raw_baseline_peaks else peak_sc
        peak_retention = (peak_sc / raw_peak) if raw_peak > 0 else 1.0

        event_pres[win.event_id] = EventPreservationMetrics(
            event_id=win.event_id,
            detected=detected,
            window_coverage_pct=round(coverage_pct, 2),
            peak_score=round(peak_sc, 4),
            peak_retention_ratio=round(peak_retention, 4),
            mean_window_score=round(mean_sc, 4),
            total_window_energy=round(total_energy, 2),
        )

        # Análisis temporal
        if detected:
            first_local_idx = int(np.argmax(above_thresh))
            win_indices = np.where(win_mask)[0]
            global_trans_idx = int(win_indices[first_local_idx])
            onset_ts_sec = float(signal.timestamps_sec[global_trans_idx])
            onset_raw_equiv = int(signal.orig_indices[global_trans_idx])

            delay_sec = max(0.0, onset_ts_sec - win.start_sec)
            delay_pts_equiv = int(round(delay_sec / 300.0))  # 300s = 5min por punto NAB
            window_elapsed = (delay_sec / win.duration_seconds) * 100.0 if win.duration_seconds > 0 else 0.0

            delta_vs_raw = None
            if raw_baseline_temporal and win.event_id in raw_baseline_temporal:
                base_delay = raw_baseline_temporal[win.event_id].delay_seconds
                if base_delay is not None:
                    delta_vs_raw = delay_sec - base_delay

            temp_pres[win.event_id] = TemporalPreservationMetrics(
                event_id=win.event_id,
                detected=True,
                onset_idx_transformed=global_trans_idx,
                onset_idx_raw_equiv=onset_raw_equiv,
                delay_seconds=round(delay_sec, 1),
                delay_points_equiv=delay_pts_equiv,
                delta_delay_vs_raw_sec=round(delta_vs_raw, 1) if delta_vs_raw is not None else None,
                window_elapsed_pct=round(window_elapsed, 2),
            )
        else:
            temp_pres[win.event_id] = TemporalPreservationMetrics(
                event_id=win.event_id,
                detected=False,
                onset_idx_transformed=None,
                onset_idx_raw_equiv=None,
                delay_seconds=None,
                delay_points_equiv=None,
                delta_delay_vs_raw_sec=None,
                window_elapsed_pct=None,
            )

    # Análisis de Falsos Positivos fuera de las ventanas
    fp_mask = (in_window_ids == 0) & (scores >= threshold)
    fp_points = int(np.sum(fp_mask))
    normal_scores = scores[in_window_ids == 0]
    max_fp = float(np.max(normal_scores)) if len(normal_scores) > 0 else 0.0

    # Contar clusters de FP
    fp_clusters = 0
    in_cluster = False
    for is_fp in fp_mask:
        if is_fp:
            if not in_cluster:
                fp_clusters += 1
                in_cluster = True
        else:
            in_cluster = False

    return EventAuditSummary(
        name=signal.name,
        event_preservation=event_pres,
        temporal_preservation=temp_pres,
        fp_points_total=fp_points,
        fp_clusters_total=fp_clusters,
        max_fp_score=round(max_fp, 4),
        raw_points_processed=len(signal.orig_indices),
        transformed_points_processed=n_pts,
    )


def evaluate_pre_compression_guard(
    values: np.ndarray,
    canonical_windows: list[CanonicalWindow],
    block_size: int = 10,
    k_sigma_threshold: float = 3.0,
) -> list[PreCompressionGuardResult]:
    """Evalúa la Pregunta 6: ¿Qué disparadores baratos detectan que NO debemos comprimir?

    Prueba tres disparadores ultraligeros O(1) por bloque:
    1. Rango dinámico local (spread): (max - min) / baseline_std > threshold
    2. Velocidad de cambio máxima: max |x[i] - x[i-1]| / baseline_std > threshold
    3. Excursión respecto a la media local: max |x - mean| / baseline_std > threshold
    """
    n = len(values)
    n_blocks = n // block_size

    # Baseline nominal de los primeros 1000 puntos
    warmup_n = min(1000, n)
    baseline_std = float(np.std(values[:warmup_n]))
    if baseline_std < 1e-6:
        baseline_std = 1.0

    # Mapear qué bloques caen dentro de ventanas canónicas
    block_in_window = np.zeros(n_blocks, dtype=int)
    for b_idx in range(n_blocks):
        b_start_raw = b_idx * block_size
        b_end_raw = (b_idx + 1) * block_size
        for win in canonical_windows:
            if not (b_end_raw < win.raw_start_idx or b_start_raw > win.raw_end_idx):
                block_in_window[b_idx] = win.event_id
                break

    reshaped = values[: n_blocks * block_size].reshape(n_blocks, block_size)

    # 1. Spread Trigger
    b_max = np.max(reshaped, axis=1)
    b_min = np.min(reshaped, axis=1)
    spread = (b_max - b_min) / baseline_std
    trigger_spread = spread > k_sigma_threshold

    # 2. Velocity Trigger (max adjacent diff)
    diffs = np.abs(np.diff(reshaped, axis=1))
    max_diff = np.max(diffs, axis=1) / baseline_std
    trigger_velocity = max_diff > (k_sigma_threshold * 0.7)

    # 3. Peak Deviation Trigger
    b_mean = np.mean(reshaped, axis=1, keepdims=True)
    dev = np.max(np.abs(reshaped - b_mean), axis=1) / baseline_std
    trigger_dev = dev > (k_sigma_threshold * 0.8)

    triggers = [
        ("Spread_Excursion_Guard", trigger_spread),
        ("Peak_Velocity_Guard", trigger_velocity),
        ("Max_Deviation_Guard", trigger_dev),
    ]

    results: list[PreCompressionGuardResult] = []

    for name, trig_array in triggers:
        events_prot: dict[int, bool] = {}
        for win in canonical_windows:
            win_blocks = trig_array[block_in_window == win.event_id]
            events_prot[win.event_id] = bool(np.any(win_blocks))

        # Falsos frenados en bloques normales
        normal_blocks = trig_array[block_in_window == 0]
        false_rate = float(np.mean(normal_blocks) * 100.0) if len(normal_blocks) > 0 else 0.0

        n_protected = sum(1 for v in events_prot.values() if v)
        protection_rate = (n_protected / len(canonical_windows)) * 100.0

        # % de bloques normales que sí se pudieron comprimir sin alerta
        savings_pct = float(np.mean(~normal_blocks) * 100.0) if len(normal_blocks) > 0 else 0.0

        results.append(
            PreCompressionGuardResult(
                trigger_name=name,
                events_protected=events_prot,
                false_guard_rate_normal=round(false_rate, 2),
                true_event_protection_rate=round(protection_rate, 2),
                data_savings_achieved_pct=round(savings_pct, 2),
            )
        )

    return results
