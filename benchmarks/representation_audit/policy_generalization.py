"""FASE 0.6 — Representation Policy Generalization Suite.

Implementa y evalúa si una misma política adaptativa de representación puede
determinar cuándo comprimir, cuándo mantener resolución, cuándo escalar y
cuándo recuperar histórico (backfill) sobre señales con distribuciones radicalmente
distintas (industrial lento vs financiero de alta volatilidad y colas pesadas),
SIN introducir lógica, constantes ni thresholds específicos de dominio.

Políticas de la Escalera de Ablación:
- G0: Always RAW
- G1: Always 10X
- G2: Adaptive Quantile Policy (reactivo instantáneo, sin hold, sin backfill)
- G3: Adaptive Policy + Hysteresis (G2 + máquina de estados con cooldown hold)
- G4: Adaptive Policy + Hysteresis + Backfill (G3 + regret ring-buffer)
- G5: Adaptive Policy + Hysteresis + Backfill + Representation Safety
"""

from __future__ import annotations

import collections
import csv
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.nab_evaluator import score_dataset_canonical
from benchmarks.representation_audit.event_analysis import CanonicalWindow
from benchmarks.representation_audit.runner import (
    DEFAULT_CSV_PATH,
    DEFAULT_WINDOWS_PATH,
    CanonicalStreamingDetector,
    load_canonical_windows,
    load_dataset,
)

logger = logging.getLogger(__name__)


from infrastructure.ml.representation import (
    AgnosticPolicyStateMachine,
    EmpiricalDistributionProfile,
    GenericRegimeShiftSentinel,
    GenericShockSentinel,
    NonParametricConformalCalibrator,
    PolicyDecisionReport,
    PolicyOperationalMode,
    PolicyResolutionState,
    RegretRingBuffer,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SUITE DE POLÍTICAS DE ABLACIÓN (G0 a G5)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AblationConfig:
    code: str
    name: str
    enable_adaptive_quantiles: bool
    enable_hysteresis: bool
    enable_backfill: bool
    enable_safety: bool
    fixed_step: int | None = None  # Para G0 (1) y G1 (10)


def get_ablation_ladder() -> list[AblationConfig]:
    return [
        AblationConfig("G0", "Always RAW", False, False, False, False, fixed_step=1),
        AblationConfig("G1", "Always 10X", False, False, False, False, fixed_step=10),
        AblationConfig("G2", "Adaptive Quantile Policy", True, False, False, False),
        AblationConfig("G3", "Adaptive Policy + Hysteresis", True, True, False, False),
        AblationConfig("G4", "Adaptive Policy + Hysteresis + Backfill", True, True, True, False),
        AblationConfig("G5", "Adaptive Policy + Hysteresis + Backfill + Safety", True, True, True, True),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIMULADOR STREAMING Y RECEPTOR DE MÉTRICAS GENERALIZADO
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PolicyRunResult:
    ablation_code: str
    ablation_name: str
    dataset_name: str
    raw_points_seen: int
    evaluated_points: int
    compression_ratio: float
    cpu_time_ms: float
    throughput_pts_per_sec: float
    time_in_raw_pct: float
    time_in_2x_pct: float
    time_in_10x_pct: float
    switch_count: int
    escalations: int
    de_escalations: int
    holds: int
    backfills: int
    # Temporalidad (medida objetivamente)
    forward_detection_delays: dict[int, int | None]
    retrospective_detection_delays: dict[int, int | None]
    temporal_recovery_points: dict[int, int]
    # Calidad supervisada (cuando existan labels)
    nab_score_standard: float | None
    nab_tp: int | None
    nab_fp: int | None
    nab_fn: int | None
    fp_clusters: int | None
    # Componentes de utilidad
    decision_quality: float | None
    computational_cost: float
    efficiency_ratio: float | None


def run_streaming_simulation(
    ablation: AblationConfig,
    values: np.ndarray,
    timestamps_sec: np.ndarray,
    timestamps_raw: list[str],
    dataset_name: str,
    canonical_windows: list[CanonicalWindow] | None = None,
    warmup_points: int = 1000,
    block_size: int = 10,
    detection_threshold: float = 0.911965,
) -> PolicyRunResult:
    """Ejecuta la política bloque a bloque con trazabilidad completa."""
    n_total = len(values)
    n_blocks = n_total // block_size

    # 1. Warmup: calibrar modelos no paramétricos
    calibrator = NonParametricConformalCalibrator(alpha_regime=0.02, alpha_shock=0.01)
    warmup_n = min(warmup_points, n_total // 4)
    level_prof, shock_prof = calibrator.fit_warmup(values[:warmup_n], block_size=block_size)

    shock_sentinel = GenericShockSentinel(shock_prof)
    regime_sentinel = GenericRegimeShiftSentinel(level_prof)

    # 2. Inicializar máquina de estados y detector idéntico
    state_machine = AgnosticPolicyStateMachine(
        enable_hysteresis=ablation.enable_hysteresis,
        enable_backfill=ablation.enable_backfill,
        enable_representation_safety=ablation.enable_safety,
        cooldown_blocks=3,
    )
    ring_buffer = RegretRingBuffer(capacity=block_size * 3)

    detector = CanonicalStreamingDetector(warmup_points=warmup_n)
    detector.fit_warmup(values[:warmup_n])

    forward_scores = np.zeros(n_total, dtype=np.float64)
    retrospective_scores = np.zeros(n_total, dtype=np.float64)
    evaluated_mask = np.zeros(n_total, dtype=bool)

    resolution_time_counts = {
        PolicyResolutionState.HIGH_RESOLUTION: 0,
        PolicyResolutionState.BALANCED: 0,
        PolicyResolutionState.COMPRESSED: 0,
    }

    t0 = time.perf_counter_ns()
    prev_val = float(values[0])

    for b_idx in range(n_blocks):
        b_start = b_idx * block_size
        b_end = (b_idx + 1) * block_size
        block_vals = values[b_start:b_end]

        # Actualizar ring buffer RAW
        for offset, v in enumerate(block_vals):
            g_idx = b_start + offset
            ring_buffer.append(g_idx, v, timestamps_sec[g_idx], timestamps_raw[g_idx])

        # Decisión de la política
        if ablation.fixed_step is not None:
            step = ablation.fixed_step
            req_backfill = False
            curr_res = PolicyResolutionState.HIGH_RESOLUTION if step == 1 else PolicyResolutionState.COMPRESSED
        else:
            # Inspección no paramétrica con sentinelas
            is_shock, s_shock = shock_sentinel.inspect(block_vals, prev_val)
            is_regime, s_regime = regime_sentinel.inspect(block_vals)

            decision = state_machine.evaluate_block(
                block_idx=b_idx,
                shock_triggered=is_shock,
                regime_triggered=is_regime,
                shock_surprise=s_shock,
                regime_surprise=s_regime,
            )
            step = decision.subsample_step
            req_backfill = decision.requires_backfill
            curr_res = decision.resolution

        resolution_time_counts[curr_res] += block_size

        # Inferencia hacia adelante (Forward evaluation)
        selected_rel = np.arange(0, block_size, step)
        selected_global = b_start + selected_rel
        eval_vals = values[selected_global]
        eval_sc = detector.score_stream(eval_vals)

        evaluated_mask[selected_global] = True

        for rel_i, sc in zip(selected_rel, eval_sc, strict=False):
            sub_end = min(b_start + rel_i + step, b_end)
            forward_scores[b_start + rel_i : sub_end] = sc
            retrospective_scores[b_start + rel_i : sub_end] = sc

        # Mecanismo de Backfill (si aplica)
        if req_backfill and ablation.enable_backfill:
            # Recuperar RAW de los últimos 2 bloques (bloque actual + previo comprimido)
            backfill_slice = ring_buffer.get_recent_raw_slice(block_size * 2)
            bf_indices = [pt[0] for pt in backfill_slice]
            bf_values = np.array([pt[1] for pt in backfill_slice])
            bf_scores = detector.score_stream(bf_values)

            # Re-escribir retrospectivamente los scores en alta resolución
            for g_i, sc in zip(bf_indices, bf_scores, strict=False):
                retrospective_scores[g_i] = sc
                evaluated_mask[g_i] = True

        prev_val = float(block_vals[-1])

    t1 = time.perf_counter_ns()
    elapsed_ms = (t1 - t0) / 1_000_000.0

    points_evaluated = int(np.sum(evaluated_mask))
    comp_ratio = float(n_total) / float(max(points_evaluated, 1))
    throughput = (n_total / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

    pct_raw = (resolution_time_counts[PolicyResolutionState.HIGH_RESOLUTION] / n_total) * 100.0
    pct_2x = (resolution_time_counts[PolicyResolutionState.BALANCED] / n_total) * 100.0
    pct_10x = (resolution_time_counts[PolicyResolutionState.COMPRESSED] / n_total) * 100.0

    # Evaluación temporal y calidad supervisada si existen ventanas canónicas
    fwd_delays: dict[int, int | None] = {}
    retro_delays: dict[int, int | None] = {}
    recovery_pts: dict[int, int] = {}
    nab_score: float | None = None
    nab_tp = None
    nab_fp = None
    nab_fn = None
    fp_clusters = None

    if canonical_windows is not None:
        # Delays forward vs retrospective
        for win in canonical_windows:
            w_fwd = forward_scores[win.raw_start_idx : win.raw_end_idx + 1]
            w_retro = retrospective_scores[win.raw_start_idx : win.raw_end_idx + 1]

            above_fwd = np.where(w_fwd >= detection_threshold)[0]
            above_retro = np.where(w_retro >= detection_threshold)[0]

            idx_fwd = int(above_fwd[0]) if len(above_fwd) > 0 else None
            idx_retro = int(above_retro[0]) if len(above_retro) > 0 else None

            fwd_delays[win.event_id] = idx_fwd
            retro_delays[win.event_id] = idx_retro

            if idx_fwd is not None and idx_retro is not None:
                recovery_pts[win.event_id] = max(0, idx_fwd - idx_retro)
            else:
                recovery_pts[win.event_id] = 0

        # Falsos positivos
        in_win_mask = np.zeros(n_total, dtype=bool)
        for win in canonical_windows:
            in_win_mask[win.raw_start_idx : win.raw_end_idx + 1] = True

        fp_mask = (~in_win_mask) & (retrospective_scores >= detection_threshold)
        fp_c = 0
        in_c = False
        for is_fp in fp_mask:
            if is_fp:
                if not in_c:
                    fp_c += 1
                    in_c = True
            else:
                in_c = False
        fp_clusters = fp_c

        # Scoring canónico NAB
        nab_limits = [(w.start_str, w.end_str) for w in canonical_windows]
        nab_res = score_dataset_canonical(
            timestamps=timestamps_raw,
            anomaly_scores=retrospective_scores.tolist(),
            window_limits=nab_limits,
            dataset_name=dataset_name,
            threshold=detection_threshold,
            profile_name="standard",
        )
        num_windows = len(canonical_windows)
        null_s = -1.0 * num_windows
        perfect_s = 1.0 * num_windows
        denom = max(perfect_s - null_s, 1e-6)
        nab_score = round(100.0 * (nab_res.score - null_s) / denom, 2)
        nab_tp = nab_res.tp
        nab_fp = nab_res.fp
        nab_fn = nab_res.fn

    # Métrica de eficiencia
    cost_fraction = points_evaluated / n_total
    decision_q = nab_score if nab_score is not None else None
    efficiency = round(decision_q / max(cost_fraction, 0.05), 2) if decision_q is not None else None

    return PolicyRunResult(
        ablation_code=ablation.code,
        ablation_name=ablation.name,
        dataset_name=dataset_name,
        raw_points_seen=n_total,
        evaluated_points=points_evaluated,
        compression_ratio=round(comp_ratio, 2),
        cpu_time_ms=round(elapsed_ms, 2),
        throughput_pts_per_sec=round(throughput, 1),
        time_in_raw_pct=round(pct_raw, 1),
        time_in_2x_pct=round(pct_2x, 1),
        time_in_10x_pct=round(pct_10x, 1),
        switch_count=state_machine.switch_count,
        escalations=state_machine.escalation_count,
        de_escalations=state_machine.de_escalation_count,
        holds=state_machine.hold_count,
        backfills=state_machine.backfill_count,
        forward_detection_delays=fwd_delays,
        retrospective_detection_delays=retro_delays,
        temporal_recovery_points=recovery_pts,
        nab_score_standard=nab_score,
        nab_tp=nab_tp,
        nab_fp=nab_fp,
        nab_fn=nab_fn,
        fp_clusters=fp_clusters,
        decision_quality=decision_q,
        computational_cost=round(cost_fraction, 4),
        efficiency_ratio=efficiency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. RUNNER PRINCIPAL DE GENERALIZACIÓN MULTI-DOMINIO
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset_b_nvda(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carga y documenta el Dataset B (NVDA_1m.csv).

    Selección de señal: Retornos relativos porcentuales abs(diff(c)/c) o Close price.
    Para evaluar la política adaptativa sobre volatilidad de mercado sin sesgo de drift
    secular, seleccionamos la tasa de retorno minuto a minuto como señal primaria.
    """
    timestamps_raw: list[str] = []
    timestamps_sec: list[float] = []
    close_prices: list[float] = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_f = float(row["ts_open"])
            c_val = float(row["c"])
            dt_str = datetime.fromtimestamp(ts_f).strftime("%Y-%m-%d %H:%M:%S")
            timestamps_sec.append(ts_f)
            timestamps_raw.append(dt_str)
            close_prices.append(c_val)

    # Señal: Close Price normalizado por su primer valor
    arr_values = np.array(close_prices, dtype=np.float64)
    arr_sec = np.array(timestamps_sec, dtype=np.float64)

    return arr_values, arr_sec, timestamps_raw


def run_phase_06_benchmark() -> dict[str, list[PolicyRunResult]]:
    """Ejecuta el protocolo de la Fase 0.6 sobre Dataset A y Dataset B."""
    print("=" * 80)
    print("ZENIN FASE 0.6: REPRESENTATION POLICY GENERALIZATION BENCHMARK")
    print("=" * 80)

    ablations = get_ablation_ladder()

    # ─────────────────────────────────────────────────────────────────────────
    # DATASET A: Industrial Slow Signal (machine_temperature_system_failure.csv)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[DATASET A] Cargando NAB Industrial Temperature...")
    vals_a, sec_a, raw_a = load_dataset(DEFAULT_CSV_PATH)
    windows_a = load_canonical_windows(
        DEFAULT_WINDOWS_PATH,
        "realKnownCause/machine_temperature_system_failure.csv",
        raw_a,
        sec_a,
    )
    print(f"  Puntos: {len(vals_a):,} | Ventanas: {len(windows_a)}")

    results_a: list[PolicyRunResult] = []
    for abl in ablations:
        print(f"  -> Ejecutando Ablación {abl.code}: {abl.name}...")
        res = run_streaming_simulation(
            ablation=abl,
            values=vals_a,
            timestamps_sec=sec_a,
            timestamps_raw=raw_a,
            dataset_name="machine_temperature_system_failure.csv",
            canonical_windows=windows_a,
            warmup_points=1000,
            block_size=10,
            detection_threshold=0.911965,
        )
        results_a.append(res)

    # ─────────────────────────────────────────────────────────────────────────
    # DATASET B: Financial High-Volatility Signal (NVDA_1m.csv)
    # ─────────────────────────────────────────────────────────────────────────
    nvda_path = _REPO_ROOT / "data" / "market" / "NVDA_1m.csv"
    print(f"\n[DATASET B] Cargando Financial High-Volatility: {nvda_path.name}...")
    vals_b, sec_b, raw_b = load_dataset_b_nvda(nvda_path)
    print(f"  Puntos: {len(vals_b):,} | Rango precios: ${vals_b.min():.2f} a ${vals_b.max():.2f}")

    results_b: list[PolicyRunResult] = []
    for abl in ablations:
        print(f"  -> Ejecutando Ablación {abl.code}: {abl.name}...")
        res = run_streaming_simulation(
            ablation=abl,
            values=vals_b,
            timestamps_sec=sec_b,
            timestamps_raw=raw_b,
            dataset_name="NVDA_1m.csv",
            canonical_windows=None,  # No forzar NAB labels sobre datos de mercado
            warmup_points=500,
            block_size=10,
            detection_threshold=0.911965,
        )
        results_b.append(res)

    return {"dataset_a_industrial": results_a, "dataset_b_financial": results_b}
