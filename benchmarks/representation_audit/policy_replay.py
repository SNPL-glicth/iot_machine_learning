"""Fase 0.5: Representation Policy Replay.

Simula y compara retrospectivamente 5 políticas de routing de representación
sobre el stream real de NAB (machine_temperature_system_failure.csv):

- Policy A: Siempre Raw (procesa 100% de puntos a máxima resolución)
- Policy B: Siempre 2x (decimación fija 2x)
- Policy C: Siempre 10x (decimación agresiva 10x)
- Policy D: Guardian Reactivo (ShockSentinel + RegimeShiftSentinel -> Raw/2x/10x)
- Policy E: Guardian Proactivo con Escalamiento e Histéresis (hold time / cooldown)

Mide:
1. Puntos reales evaluados por el detector (carga computacional)
2. Retardo de detección en cada una de las 4 fallas canónicas (onset delay)
3. Falsos positivos (puntos y clusters)
4. Score oficial canónico de NAB (vía score_dataset_canonical)
5. Utilidad neta: NAB Score / Cómputo invertido
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Configurar imports del repo
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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

logging.basicConfig(level=logging.WARNING)

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_JSON = BENCHMARK_DIR / "results" / "policy_replay.json"


# ─────────────────────────────────────────────────────────────────────────────
# SENTINELAS ULTRALIGEROS O(1) (Sin nombres históricos en el código)
# ─────────────────────────────────────────────────────────────────────────────


class ShockSentinel:
    """Detecta choques térmicos, discontinuidades y deltas abruptos (Klein rápido)."""

    def __init__(self, threshold_sigma: float = 2.5) -> None:
        self.threshold_sigma = threshold_sigma

    def inspect_block(
        self, block_values: np.ndarray, prev_value: float, baseline_std: float
    ) -> bool:
        if len(block_values) == 0:
            return False
        diff_with_prev = abs(block_values[0] - prev_value)
        internal_diffs = (
            np.max(np.abs(np.diff(block_values))) if len(block_values) > 1 else 0.0
        )
        max_velocity = max(diff_with_prev, float(internal_diffs))
        return (max_velocity / max(baseline_std, 1e-6)) > self.threshold_sigma


class RegimeShiftSentinel:
    """Detecta que el centroide del bloque abandonó el régimen nominal (Noether macro)."""

    def __init__(self, threshold_sigma: float = 1.8) -> None:
        self.threshold_sigma = threshold_sigma

    def inspect_block(
        self, block_values: np.ndarray, baseline_mean: float, baseline_std: float
    ) -> bool:
        if len(block_values) == 0:
            return False
        block_mean = float(np.mean(block_values))
        displacement = abs(block_mean - baseline_mean) / max(baseline_std, 1e-6)
        return displacement > self.threshold_sigma


# ─────────────────────────────────────────────────────────────────────────────
# POLÍTICAS DE ROUTING
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PolicyDecision:
    representation: str  # "raw", "2x", "10x"
    subsample_step: int  # 1 (raw), 2 (2x), 10 (10x)
    reason: str


class BaseRoutingPolicy:
    def decide_block(
        self,
        block_idx: int,
        block_values: np.ndarray,
        prev_val: float,
        shock_sentinel: ShockSentinel,
        regime_sentinel: RegimeShiftSentinel,
        baseline_mean: float,
        baseline_std: float,
    ) -> PolicyDecision:
        raise NotImplementedError


class PolicyA_AlwaysRaw(BaseRoutingPolicy):
    """Línea base: Siempre procesa la señal completa a máxima resolución."""

    def decide_block(self, *args: Any, **kwargs: Any) -> PolicyDecision:
        return PolicyDecision("raw", 1, "fixed_raw")


class PolicyB_Always2x(BaseRoutingPolicy):
    """Siempre decima 2x."""

    def decide_block(self, *args: Any, **kwargs: Any) -> PolicyDecision:
        return PolicyDecision("2x", 2, "fixed_2x")


class PolicyC_Always10x(BaseRoutingPolicy):
    """Siempre decima 10x."""

    def decide_block(self, *args: Any, **kwargs: Any) -> PolicyDecision:
        return PolicyDecision("10x", 10, "fixed_10x")


class PolicyD_GuardianReactive(BaseRoutingPolicy):
    """Guardián reactivo sin memoria: shock -> raw, régimen -> 2x, reposo -> 10x."""

    def decide_block(
        self,
        block_idx: int,
        block_values: np.ndarray,
        prev_val: float,
        shock_sentinel: ShockSentinel,
        regime_sentinel: RegimeShiftSentinel,
        baseline_mean: float,
        baseline_std: float,
    ) -> PolicyDecision:
        is_shock = shock_sentinel.inspect_block(block_values, prev_val, baseline_std)
        if is_shock:
            return PolicyDecision("raw", 1, "shock_triggered")

        is_regime = regime_sentinel.inspect_block(block_values, baseline_mean, baseline_std)
        if is_regime:
            return PolicyDecision("2x", 2, "regime_shift_triggered")

        return PolicyDecision("10x", 10, "nominal_deep_compression")


class PolicyE_GuardianEscalation(BaseRoutingPolicy):
    """Guardián proactivo con cooldown e histéresis temporal.

    Cuando un centinela se activa, mantiene la alta resolución durante
    un periodo de enfriamiento (cooldown_blocks) para no alternar erráticamente.
    """

    def __init__(self, cooldown_blocks: int = 4) -> None:
        self.cooldown_blocks = cooldown_blocks
        self.active_hold_until_block: int = -1
        self.held_representation: str = "10x"
        self.held_step: int = 10

    def decide_block(
        self,
        block_idx: int,
        block_values: np.ndarray,
        prev_val: float,
        shock_sentinel: ShockSentinel,
        regime_sentinel: RegimeShiftSentinel,
        baseline_mean: float,
        baseline_std: float,
    ) -> PolicyDecision:
        is_shock = shock_sentinel.inspect_block(block_values, prev_val, baseline_std)
        is_regime = regime_sentinel.inspect_block(block_values, baseline_mean, baseline_std)

        if is_shock:
            # Choque máximo: escalar a raw y extender cooldown
            self.held_representation = "raw"
            self.held_step = 1
            self.active_hold_until_block = block_idx + self.cooldown_blocks
            return PolicyDecision("raw", 1, "shock_escalation_hold")

        if is_regime:
            # Desplazamiento de régimen: escalar al menos a 2x
            if self.active_hold_until_block < block_idx or self.held_representation == "10x":
                self.held_representation = "2x"
                self.held_step = 2
            self.active_hold_until_block = max(
                self.active_hold_until_block, block_idx + self.cooldown_blocks
            )
            return PolicyDecision(self.held_representation, self.held_step, "regime_escalation_hold")

        # Sin alertas activas: ¿sigue en periodo de retención?
        if block_idx <= self.active_hold_until_block:
            return PolicyDecision(
                self.held_representation,
                self.held_step,
                f"cooldown_active_{self.held_representation}",
            )

        # Reposo completo: 10x
        self.held_representation = "10x"
        self.held_step = 10
        return PolicyDecision("10x", 10, "nominal_rest_state")


# ─────────────────────────────────────────────────────────────────────────────
# SIMULADOR DE STREAMING Y EVALUADOR NAB
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PolicyReplayMetrics:
    policy_name: str
    points_evaluated: int
    computation_reduction_pct: float
    elapsed_ms: float
    nab_score_standard: float
    nab_tp_count: int
    nab_fp_count: int
    nab_fn_count: int
    event_delays_points: dict[int, int | None]
    raw_fp_clusters: int
    utility_score: float  # nab_score / (points_evaluated / total_points)


def simulate_policy_stream(
    policy: BaseRoutingPolicy,
    policy_name: str,
    values: np.ndarray,
    timestamps_sec: np.ndarray,
    timestamps_raw: list[str],
    canonical_windows: list[CanonicalWindow],
    block_size: int = 10,
    threshold: float = 0.65,
) -> PolicyReplayMetrics:
    """Ejecuta la simulación temporal bloque a bloque evaluando el detector canónico."""
    n_total = len(values)
    n_blocks = n_total // block_size

    # Baseline nominal de warmup
    warmup_n = min(1000, n_total)
    baseline_vals = values[:warmup_n]
    baseline_mean = float(np.mean(baseline_vals))
    baseline_std = float(np.std(baseline_vals))
    if baseline_std < 1e-4:
        baseline_std = 1.0

    shock_sentinel = ShockSentinel(threshold_sigma=2.5)
    regime_sentinel = RegimeShiftSentinel(threshold_sigma=1.8)

    # Inicializar detector canónico con warmup
    detector = CanonicalStreamingDetector(warmup_points=warmup_n)
    detector.fit_warmup(baseline_vals)

    full_resolution_scores = np.zeros(n_total, dtype=np.float64)
    evaluated_mask = np.zeros(n_total, dtype=bool)

    t0 = time.perf_counter_ns()
    prev_val = baseline_mean

    for b_idx in range(n_blocks):
        b_start = b_idx * block_size
        b_end = (b_idx + 1) * block_size
        block_vals = values[b_start:b_end]

        decision = policy.decide_block(
            block_idx=b_idx,
            block_values=block_vals,
            prev_val=prev_val,
            shock_sentinel=shock_sentinel,
            regime_sentinel=regime_sentinel,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
        )

        step = decision.subsample_step

        # Seleccionar qué puntos evaluar en este bloque
        selected_rel_indices = np.arange(0, block_size, step)
        selected_global_indices = b_start + selected_rel_indices

        eval_values = values[selected_global_indices]
        eval_scores = detector.score_stream(eval_values)

        evaluated_mask[selected_global_indices] = True

        # Propagar scores: si se evaluó a 10x (1 punto), el score cubre el bloque
        # si se evaluó a 2x (5 puntos), cada punto cubre 2 posiciones; si raw, 1 a 1.
        for rel_i, sc in zip(selected_rel_indices, eval_scores, strict=False):
            sub_end = min(b_start + rel_i + step, b_end)
            full_resolution_scores[b_start + rel_i : sub_end] = sc

        prev_val = float(block_vals[-1])

    t1 = time.perf_counter_ns()
    elapsed_ms = (t1 - t0) / 1_000_000.0

    points_evaluated = int(np.sum(evaluated_mask))
    comp_reduction = (1.0 - (points_evaluated / n_total)) * 100.0

    # Retardo por evento canónico
    event_delays: dict[int, int | None] = {}
    for win in canonical_windows:
        win_scores = full_resolution_scores[win.raw_start_idx : win.raw_end_idx + 1]
        above = np.where(win_scores >= threshold)[0]
        if len(above) > 0:
            first_idx = int(above[0])
            event_delays[win.event_id] = first_idx
        else:
            event_delays[win.event_id] = None

    # Falsos positivos fuera de ventana
    in_win_mask = np.zeros(n_total, dtype=bool)
    for win in canonical_windows:
        in_win_mask[win.raw_start_idx : win.raw_end_idx + 1] = True

    fp_mask = (~in_win_mask) & (full_resolution_scores >= threshold)

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

    # Evaluación oficial con NAB score_dataset_canonical
    nab_window_limits = [(w.start_str, w.end_str) for w in canonical_windows]
    nab_res = score_dataset_canonical(
        timestamps=timestamps_raw,
        anomaly_scores=full_resolution_scores.tolist(),
        window_limits=nab_window_limits,
        dataset_name="machine_temperature_system_failure.csv",
        threshold=threshold,
        profile_name="standard",
    )

    num_windows = len(canonical_windows)
    null_s = -1.0 * num_windows
    perfect_s = 1.0 * num_windows
    denom = max(perfect_s - null_s, 1e-6)
    normalized_nab = 100.0 * (nab_res.score - null_s) / denom
    nab_score = round(normalized_nab, 2)
    load_ratio = max(points_evaluated / n_total, 0.05)
    utility = round(nab_score / load_ratio, 2)

    return PolicyReplayMetrics(
        policy_name=policy_name,
        points_evaluated=points_evaluated,
        computation_reduction_pct=round(comp_reduction, 1),
        elapsed_ms=round(elapsed_ms, 2),
        nab_score_standard=nab_score,
        nab_tp_count=nab_res.tp,
        nab_fp_count=nab_res.fp,
        nab_fn_count=nab_res.fn,
        event_delays_points=event_delays,
        raw_fp_clusters=fp_clusters,
        utility_score=utility,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER DE POLÍTICAS Y REPORTE
# ─────────────────────────────────────────────────────────────────────────────


def run_policy_replay(
    csv_path: Path = DEFAULT_CSV_PATH,
    windows_path: Path = DEFAULT_WINDOWS_PATH,
    out_json: Path = DEFAULT_OUT_JSON,
) -> None:
    print("=" * 80)
    print("ZENIN FASE 0.5: REPRESENTATION POLICY REPLAY BENCHMARK")
    print("=" * 80)
    print(f"Cargando dataset: {csv_path}")

    values, timestamps_sec, timestamps_raw = load_dataset(csv_path)
    canonical_windows = load_canonical_windows(
        windows_path,
        "realKnownCause/machine_temperature_system_failure.csv",
        timestamps_raw,
        timestamps_sec,
    )

    policies = [
        ("Policy A: Siempre Raw (100% Puntos)", PolicyA_AlwaysRaw()),
        ("Policy B: Siempre 2x (50% Puntos)", PolicyB_Always2x()),
        ("Policy C: Siempre 10x (10% Puntos)", PolicyC_Always10x()),
        ("Policy D: Guardian Reactivo (Shock+Regime)", PolicyD_GuardianReactive()),
        ("Policy E: Guardian Proactivo (Escalation+Hold)", PolicyE_GuardianEscalation(cooldown_blocks=4)),
    ]

    # Evaluación a umbral calibrado Numenta NAB theta* = 0.912
    eval_threshold = 0.911965
    print(f"\nSimulando políticas con umbral óptimo NAB theta* = {eval_threshold:.4f}...\n")
    results: list[PolicyReplayMetrics] = []

    for name, pol in policies:
        print(f"-> Evaluando {name}...")
        res = simulate_policy_stream(
            policy=pol,
            policy_name=name,
            values=values,
            timestamps_sec=timestamps_sec,
            timestamps_raw=timestamps_raw,
            canonical_windows=canonical_windows,
            block_size=10,
            threshold=eval_threshold,
        )
        results.append(res)

    # Imprimir tabla comparativa
    print("\n" + "=" * 95)
    print("RESULTADOS DE LA COMPARATIVA DE POLÍTICAS DE ENRUTAMIENTO TEMPORAL (FASE 0.5)")
    print("=" * 95)
    print(
        f"{'Política':<35} | {'Puntos Eval':<11} | {'Ahorro Cómputo':<14} | {'NAB Score':<9} | {'TP/FP/FN':<10} | {'Utilidad':<8}"
    )
    print("-" * 95)
    for r in results:
        tp_fp_str = f"{r.nab_tp_count}/{r.nab_fp_count}/{r.nab_fn_count}"
        print(
            f"{r.policy_name:<35} | {r.points_evaluated:>11,} | {r.computation_reduction_pct:>13.1f}% | {r.nab_score_standard:>9.2f} | {tp_fp_str:<10} | {r.utility_score:>8.1f}"
        )

    print("\n" + "=" * 95)
    print("DETALLE DE RETARDO TEMPORAL POR EVENTO CANÓNICO (DELAY EN PUNTOS / HORAS)")
    print("=" * 95)
    print(
        f"{'Política':<35} | {'Ev 1 (Variación)':<16} | {'Ev 2 (Shock 2°)':<16} | {'Ev 3 (Régimen 64°)':<18} | {'Ev 4 (Colapso 25°)':<18}"
    )
    print("-" * 95)
    for r in results:

        def _fmt_delay(pts: int | None) -> str:
            if pts is None:
                return "❌ PERDIDO"
            hours = pts * 5 / 60
            return f"+{pts} pts ({hours:.1f}h)"

        d = r.event_delays_points
        print(
            f"{r.policy_name:<35} | {_fmt_delay(d.get(1)):<16} | {_fmt_delay(d.get(2)):<16} | {_fmt_delay(d.get(3)):<18} | {_fmt_delay(d.get(4)):<18}"
        )

    # Guardar a JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nReporte JSON guardado en: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZENIN Phase 0.5: Representation Policy Replay")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS_PATH)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    run_policy_replay(csv_path=args.csv, windows_path=args.windows, out_json=args.out_json)
