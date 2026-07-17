"""A/B Clean — Split temporal estricto 80/20 + MoE 4 vs 5.

Paso A: NeuralExpert entrenado en 80% inicial, evaluado SOLO en 20% final.
Paso B: MoE 4 expertos vs MoE 5 expertos (neural incluido, peso 0.05).
"""

from __future__ import annotations

import json
import math
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PARENT = os.path.dirname(_PROJECT_ROOT)
for _p in (_PROJECT_ROOT, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from infrastructure.ml.moe import (
    ExpertRegistry,
    ContextualRegimeGating,
    DiscrepancyAwareFusion,
    MoEPredictionEngine,
)
from infrastructure.ml.moe.feature_context import FeatureContext
from infrastructure.ml.moe.experts.neural_expert import create_neural_expert, NeuralExpert
from infrastructure.ml.moe.expert_wrappers.engine_adapter import (
    create_baseline_expert,
    create_kalman_expert,
    create_statistical_expert,
    create_taylor_expert,
)
from infrastructure.ml.engines.core.factory import EngineFactory

WINDOW_SIZE = 15
SPLIT_FRAC = 0.80
OUTPUT_PATH = os.path.join(_SCRIPT_DIR, "ab_clean_results.json")


def load_and_pivot(excel_path):
    sheets = pd.read_excel(excel_path, sheet_name=None)
    series_list = []
    for sheet_name, df in sheets.items():
        if all(c in df.columns for c in ("Equipo", "Parámetro", "Valor")):
            df_pivot = df.pivot_table(index="Fecha", columns="Parámetro", values="Valor", aggfunc="first").sort_index()
        else:
            skip = {"FECHA", "HORA", "FECHA Y HORA", "EQUIPO", "FECHA/HORA", "Fecha/Hora",
                    "DATETIME", "timestamp", "fecha", "hora", "fecha_hora", "equipo", "indice", "index"}
            param_cols = [c for c in df.columns if c not in skip and df[c].dtype in ("float64", "int64")]
            df_pivot = df[param_cols]
        for col in df_pivot.columns:
            values = df_pivot[col].dropna().values.tolist()
            if len(values) >= WINDOW_SIZE + 2:
                series_list.append({"equipment": sheet_name, "parameter": str(col).strip(), "values": values, "n_points": len(values)})
    return series_list


def classify_regime(values):
    n = len(values)
    if n < 2:
        return "unknown"
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5 if n >= 2 else 0.0
    x_mean = (n - 1) / 2.0
    num = sum((i - x_mean) * (v - mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if abs(den) > 1e-12 else 0.0
    r_squared = 0.0
    if den > 1e-12 and n >= 3:
        ss_res = sum((v - (mean + slope * (i - x_mean))) ** 2 for i, v in enumerate(values))
        ss_tot = sum((v - mean) ** 2 for v in values)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    noise_ratio = std / (abs(mean) + 1e-6) if abs(mean) > 1e-9 else std / (std + 1e-6)
    if noise_ratio > 0.5 and std > 0.3:
        return "noisy"
    elif r_squared > 0.6 and abs(slope) > 0.005 * (abs(mean) + 1e-6):
        return "trending"
    elif std > 0.8 * (abs(mean) + 1e-6) or std > 2.0:
        return "volatile"
    return "stable"


def build_feature_context(values):
    n = len(values)
    mean = sum(values) / n if n > 0 else 0.0
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5 if n >= 2 else 0.0
    slope, r_squared, curvature, autocorr = 0.0, 0.0, 0.0, 0.0
    if n >= 2:
        x_mean = (n - 1) / 2.0
        num = sum((i - x_mean) * (v - mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if abs(den) > 1e-12 else 0.0
        if den > 1e-12 and n >= 3:
            ss_res = sum((v - (mean + slope * (i - x_mean))) ** 2 for i, v in enumerate(values))
            ss_tot = sum((v - mean) ** 2 for v in values)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    if n >= 3:
        second_diffs = [values[i] - 2 * values[i - 1] + values[i - 2] for i in range(2, n)]
        curvature = sum(second_diffs) / len(second_diffs)
    noise_ratio = std / (abs(mean) + 1e-6) if abs(mean) > 1e-9 else std / (std + 1e-6)
    if n >= 4:
        var = sum((v - mean) ** 2 for v in values) + 1e-12
        autocorr = sum((values[i] - mean) * (values[i - 1] - mean) for i in range(1, n)) / var
    stability = 1.0 / (1.0 + noise_ratio + abs(slope) * 10.0)
    return FeatureContext(
        regime=classify_regime(values),
        mean=mean, std=std, slope=slope, curvature=curvature,
        noise_ratio=noise_ratio, stability=stability,
        hampel_outlier_mask=[], spatial_correlation_score=0.0,
    )


def _try_register(registry, name, create_fn, engine_name):
    try:
        engine = EngineFactory.create(engine_name)
        expert = create_fn(engine.as_port())
        registry.register(name, expert, expert.capabilities)
        return True
    except Exception as exc:
        logger.warning("Failed to register %s: %s", name, exc)
        return False


def create_moe_engine(expert_names=None, sparsity_k=2):
    registry = ExpertRegistry()
    all_experts = [
        ("baseline", create_baseline_expert, "baseline_moving_average"),
        ("statistical", create_statistical_expert, "statistical"),
        ("taylor", create_taylor_expert, "taylor"),
        ("kalman", create_kalman_expert, "kalman"),
    ]
    if expert_names is None:
        expert_names = [e[0] for e in all_experts]
    for name, fn, en in all_experts:
        if name in expert_names:
            _try_register(registry, name, fn, en)
    # Neural expert doesn't use EngineFactory
    if "neural" in expert_names:
        try:
            neural_expert = create_neural_expert()
            registry.register("neural", neural_expert, neural_expert.capabilities)
        except Exception as exc:
            logger.warning("Failed to register neural: %s", exc)
    if len(registry) == 0:
        return None
    gating = ContextualRegimeGating(expert_ids=registry.list_all())
    fusion = DiscrepancyAwareFusion()
    return MoEPredictionEngine(
        registry=registry, gating=gating, fusion=fusion,
        fallback_engine=None, sparsity_k=sparsity_k,
    )


def evaluate_moe_series(engine, values, split_idx, series_id_base="test"):
    n = len(values)
    errors = []
    regimes = {}
    neural_weight_trace = []
    for i in range(max(WINDOW_SIZE, split_idx), n):
        window = values[i - WINDOW_SIZE: i]
        actual = values[i]
        ctx = build_feature_context(window)
        regimes[ctx.regime] = regimes.get(ctx.regime, 0) + 1
        try:
            pred = engine.predict(
                values=window,
                feature_context=ctx,
                series_id=f"{series_id_base}_{i}",
            )
            err = abs(pred.predicted_value - actual)
            if math.isfinite(err):
                errors.append(err)
            # Extract gating probs for neural weight tracking
            if pred.metadata:
                moe_data = pred.metadata.get("moe", {})
                gating_probs = moe_data.get("gating_probs", {})
                if "neural" in gating_probs:
                    neural_weight_trace.append({
                        "regime": ctx.regime,
                        "neural_weight": gating_probs["neural"],
                        "selected_experts": moe_data.get("selected_experts", []),
                        "dominant": moe_data.get("dominant_expert", ""),
                        "neural_selected": "neural" in moe_data.get("selected_experts", []),
                    })
        except Exception:
            continue
    return errors, regimes, neural_weight_trace


def evaluate_neural_series(expert, train_values, full_values, split_idx, series_id):
    n = len(full_values)
    trained = expert.warmup(series_id, train_values)
    if not trained:
        return None, {}, []
    errors = []
    regimes = {}
    for i in range(max(WINDOW_SIZE, split_idx), n):
        window_vals = full_values[i - WINDOW_SIZE: i]
        actual = full_values[i]
        ctx = classify_regime(window_vals)
        regimes[ctx] = regimes.get(ctx, 0) + 1
        try:
            from iot_machine_learning.domain.entities.iot.sensor_reading import Reading
            from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
            readings = [Reading(series_id=series_id, value=v, timestamp=float(t)) for t, v in enumerate(window_vals)]
            sw = SensorWindow(series_id=series_id, readings=readings)
            out = expert.predict(sw)
            err = abs(out.prediction - actual)
            if math.isfinite(err):
                errors.append(err)
        except Exception:
            continue
    return errors, regimes, []


@dataclass
class SeriesResult:
    equipment: str
    parameter: str
    n_points: int
    split_idx: int
    n_predictions: int
    moe4_mae: float = 0.0
    moe4_nmae_pct: float = 0.0
    neural_mae: float = 0.0
    neural_nmae_pct: float = 0.0
    moe5_mae: float = 0.0
    moe5_nmae_pct: float = 0.0
    moe5k3_mae: float = 0.0
    moe5k3_nmae_pct: float = 0.0
    neural_vs_moe4_delta_pct: float = 0.0
    moe5_vs_moe4_delta_pct: float = 0.0
    moe5k3_vs_moe4_delta_pct: float = 0.0
    neural_regime_distribution: Dict[str, int] = field(default_factory=dict)
    neural_weight_by_regime: Dict[str, List[float]] = field(default_factory=dict)
    neural_selected_count_k2: int = 0
    neural_selected_count_k3: int = 0
    regimes: Dict[str, int] = field(default_factory=dict)


def main():
    start = time.time()
    excel_path = os.path.join(_SCRIPT_DIR, "Información Chiller y CA - ZENIN.xlsx")

    print("Loading dataset...")
    series_list = load_and_pivot(excel_path)
    print(f"  {len(series_list)} total series")

    print("Creating MoE engines...")
    engine4 = create_moe_engine(expert_names=["baseline", "statistical", "taylor", "kalman"], sparsity_k=2)
    engine5_k2 = create_moe_engine(expert_names=["baseline", "statistical", "taylor", "kalman", "neural"], sparsity_k=2)
    engine5_k3 = create_moe_engine(expert_names=["baseline", "statistical", "taylor", "kalman", "neural"], sparsity_k=3)
    if engine4 is None or engine5_k2 is None or engine5_k3 is None:
        print("ERROR: engine creation failed")
        return

    results = []
    skipped_short = 0
    skipped_training = 0

    for idx, sd in enumerate(series_list):
        values = sd["values"]
        n = len(values)
        split_idx = int(n * SPLIT_FRAC)

        min_train = 30
        min_test = WINDOW_SIZE + 2
        if split_idx < min_train or (n - split_idx) < min_test:
            skipped_short += 1
            continue

        train_vals = values[:split_idx]
        series_id = f"{sd['equipment']}_{sd['parameter']}_split80"

        # --- Paso A: Neural standalone ---
        expert = create_neural_expert()
        neural_errs, neural_regimes, _ = evaluate_neural_series(expert, train_vals, values, split_idx, series_id)

        if neural_errs is None or len(neural_errs) < 2:
            skipped_training += 1
            continue

        neural_mae = float(np.mean(neural_errs))
        neural_series_mean = float(np.mean(values[split_idx:]))
        neural_nmae = (neural_mae / (abs(neural_series_mean) + 1e-10)) * 100

        # --- Paso B: MoE 4 on test portion ---
        moe4_errs, moe4_regimes, _ = evaluate_moe_series(engine4, values, split_idx, series_id)
        if len(moe4_errs) < 2:
            continue
        moe4_mae = float(np.mean(moe4_errs))

        # --- MoE 5 (k=2) on test portion ---
        moe5k2_errs, _, weight_trace_k2 = evaluate_moe_series(engine5_k2, values, split_idx, series_id)
        if len(moe5k2_errs) < 2:
            continue
        moe5k2_mae = float(np.mean(moe5k2_errs))

        # --- MoE 5 (k=3) on test portion ---
        moe5k3_errs, _, weight_trace_k3 = evaluate_moe_series(engine5_k3, values, split_idx, series_id)
        if len(moe5k3_errs) < 2:
            continue
        moe5k3_mae = float(np.mean(moe5k3_errs))

        test_mean = abs(neural_series_mean) + 1e-10
        moe4_nmae = (moe4_mae / test_mean) * 100
        moe5k2_nmae = (moe5k2_mae / test_mean) * 100
        moe5k3_nmae = (moe5k3_mae / test_mean) * 100

        delta_neural_vs_moe4 = ((neural_mae - moe4_mae) / (moe4_mae + 1e-10)) * 100
        delta_moe5k2_vs_moe4 = ((moe5k2_mae - moe4_mae) / (moe4_mae + 1e-10)) * 100
        delta_moe5k3_vs_moe4 = ((moe5k3_mae - moe4_mae) / (moe4_mae + 1e-10)) * 100

        def aggregate_weights(trace):
            nw = {}
            sel = 0
            for entry in trace:
                r = entry["regime"]
                if r not in nw:
                    nw[r] = []
                nw[r].append(entry["neural_weight"])
                if entry["neural_selected"]:
                    sel += 1
            return nw, sel

        nw_k2, sel_k2 = aggregate_weights(weight_trace_k2)
        nw_k3, sel_k3 = aggregate_weights(weight_trace_k3)

        results.append(SeriesResult(
            equipment=sd["equipment"],
            parameter=sd["parameter"],
            n_points=n,
            split_idx=split_idx,
            n_predictions=len(moe4_errs),
            moe4_mae=round(moe4_mae, 4),
            moe4_nmae_pct=round(moe4_nmae, 2),
            neural_mae=round(neural_mae, 4),
            neural_nmae_pct=round(neural_nmae, 2),
            moe5_mae=round(moe5k2_mae, 4),
            moe5_nmae_pct=round(moe5k2_nmae, 2),
            moe5k3_mae=round(moe5k3_mae, 4),
            moe5k3_nmae_pct=round(moe5k3_nmae, 2),
            neural_vs_moe4_delta_pct=round(delta_neural_vs_moe4, 2),
            moe5_vs_moe4_delta_pct=round(delta_moe5k2_vs_moe4, 2),
            moe5k3_vs_moe4_delta_pct=round(delta_moe5k3_vs_moe4, 2),
            neural_regime_distribution=neural_regimes,
            neural_weight_by_regime={r: [round(w, 4) for w in ws] for r, ws in nw_k2.items()},
            neural_selected_count_k2=sel_k2,
            neural_selected_count_k3=sel_k3,
            regimes=moe4_regimes,
        ))

        if (idx + 1) % 10 == 0:
            print(f"  ... {idx + 1}/{len(series_list)} series (kept={len(results)}, skipped_short={skipped_short}, skipped_train={skipped_training})")

    # ── Global Metrics ─────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("ERROR: no valid results")
        return

    moe4_maes = np.array([r.moe4_mae for r in results])
    moe4_nmaes = np.array([r.moe4_nmae_pct for r in results])
    neural_maes = np.array([r.neural_mae for r in results])
    neural_nmaes = np.array([r.neural_nmae_pct for r in results])
    moe5_maes = np.array([r.moe5_mae for r in results])
    moe5_nmaes = np.array([r.moe5_nmae_pct for r in results])
    moe5k3_maes = np.array([r.moe5k3_mae for r in results])
    moe5k3_nmaes = np.array([r.moe5k3_nmae_pct for r in results])

    global_metrics = {
        "n_series": n,
        "total_predictions": sum(r.n_predictions for r in results),
        "split_frac": SPLIT_FRAC,
        "moe4_baseline": {
            "median_mae": round(float(np.median(moe4_maes)), 4),
            "median_nmae_pct": round(float(np.median(moe4_nmaes)), 2),
            "avg_mae": round(float(np.mean(moe4_maes)), 4),
            "avg_nmae_pct": round(float(np.mean(moe4_nmaes)), 2),
        },
        "neural_standalone": {
            "median_mae": round(float(np.median(neural_maes)), 4),
            "median_nmae_pct": round(float(np.median(neural_nmaes)), 2),
            "avg_mae": round(float(np.mean(neural_maes)), 4),
            "avg_nmae_pct": round(float(np.mean(neural_nmaes)), 2),
        },
        "moe5_k2": {
            "median_mae": round(float(np.median(moe5_maes)), 4),
            "median_nmae_pct": round(float(np.median(moe5_nmaes)), 2),
            "avg_mae": round(float(np.mean(moe5_maes)), 4),
            "avg_nmae_pct": round(float(np.mean(moe5_nmaes)), 2),
        },
        "moe5_k3": {
            "median_mae": round(float(np.median(moe5k3_maes)), 4),
            "median_nmae_pct": round(float(np.median(moe5k3_nmaes)), 2),
            "avg_mae": round(float(np.mean(moe5k3_maes)), 4),
            "avg_nmae_pct": round(float(np.mean(moe5k3_nmaes)), 2),
        },
    }

    # Verdicts
    improved_neural = sum(1 for r in results if r.neural_mae < r.moe4_mae - 1e-10)
    worsened_neural = sum(1 for r in results if r.neural_mae > r.moe4_mae + 1e-10)
    neutral_neural = n - improved_neural - worsened_neural

    improved_moe5k2 = sum(1 for r in results if r.moe5_mae < r.moe4_mae - 1e-10)
    worsened_moe5k2 = sum(1 for r in results if r.moe5_mae > r.moe4_mae + 1e-10)
    neutral_moe5k2 = n - improved_moe5k2 - worsened_moe5k2

    improved_moe5k3 = sum(1 for r in results if r.moe5k3_mae < r.moe4_mae - 1e-10)
    worsened_moe5k3 = sum(1 for r in results if r.moe5k3_mae > r.moe4_mae + 1e-10)
    neutral_moe5k3 = n - improved_moe5k3 - worsened_moe5k3

    med4 = global_metrics["moe4_baseline"]["median_nmae_pct"]
    med_nn = global_metrics["neural_standalone"]["median_nmae_pct"]
    med5k2 = global_metrics["moe5_k2"]["median_nmae_pct"]
    med5k3 = global_metrics["moe5_k3"]["median_nmae_pct"]

    def verdict(val, base, label):
        if val < base:
            return f"✅ {label} MEJORA mediana nMAE: {base}% → {val}% (Δ {val - base:+.2f}pp)"
        elif val == base:
            return f"⚖️ {label} NEUTRO: ambos {base}%"
        else:
            return f"❌ {label} EMPEORA mediana nMAE: {base}% → {val}% (Δ {val - base:+.2f}pp)"

    verdict_a = verdict(med_nn, med4, "NeuralExpert standalone")
    verdict_b_k2 = verdict(med5k2, med4, "MoE 5 (k=2)")
    verdict_b_k3 = verdict(med5k3, med4, "MoE 5 (k=3)")

    # ── Neural weight evolution summaries (from k=2, since k=2 gating_probs reflect base weights) ──
    all_weight_traces = []
    for r in results:
        for regime, weights in r.neural_weight_by_regime.items():
            if weights:
                all_weight_traces.append({
                    "series": f"{r.equipment}/{r.parameter}",
                    "regime": regime,
                    "min_weight": round(float(min(weights)), 4),
                    "max_weight": round(float(max(weights)), 4),
                    "mean_weight": round(float(np.mean(weights)), 4),
                    "n_obs": len(weights),
                })

    # ── Print report ──
    print("\n" + "=" * 70)
    print(" VEREDICTO A/B — Split temporal 80/20, evaluación sobre 20% final")
    print("=" * 70)
    print(f"  Series totales:            {len(series_list)}")
    print(f"  Series evaluadas:          {n}")
    print(f"  Descartadas (cortas):      {skipped_short}")
    print(f"  Descartadas (train fail):  {skipped_training}")
    print(f"  Predicciones totales:      {global_metrics['total_predictions']}")
    print(f"  Split:                     {SPLIT_FRAC*100:.0f}% train / {(1-SPLIT_FRAC)*100:.0f}% test cronológico")
    print()

    print(f"{'Métrica':<35} {'MoE 4':>14} {'Neural':>14} {'MoE5k2':>14} {'MoE5k3':>14}")
    print("-" * 91)
    for metric in ["median_mae", "median_nmae_pct", "avg_mae", "avg_nmae_pct"]:
        b4 = global_metrics["moe4_baseline"][metric]
        nn = global_metrics["neural_standalone"][metric]
        b5k2 = global_metrics["moe5_k2"][metric]
        b5k3 = global_metrics["moe5_k3"][metric]
        print(f"{metric:<35} {b4:>14.4f} {nn:>14.4f} {b5k2:>14.4f} {b5k3:>14.4f}")
    print("-" * 91)

    print(f"\n── PASO A: NeuralExpert standalone vs MoE 4 ──")
    print(f"  Mejora:  {improved_neural}/{n}   Empeora: {worsened_neural}/{n}   Neutral: {neutral_neural}/{n}")
    print(f"  {verdict_a}")

    print(f"\n── PASO B: MoE 5 expertos vs MoE 4 ──")
    print(f"  [k=2] Mejora: {improved_moe5k2}/{n}   Empeora: {worsened_moe5k2}/{n}   Neutral: {neutral_moe5k2}/{n}")
    print(f"  [k=2] {verdict_b_k2}")
    print(f"  [k=3] Mejora: {improved_moe5k3}/{n}   Empeora: {worsened_moe5k3}/{n}   Neutral: {neutral_moe5k3}/{n}")
    print(f"  [k=3] {verdict_b_k3}")

    # Top 5 deltas
    sorted_nn = sorted(results, key=lambda r: r.neural_vs_moe4_delta_pct)
    sorted_5k2 = sorted(results, key=lambda r: r.moe5_vs_moe4_delta_pct)
    sorted_5k3 = sorted(results, key=lambda r: r.moe5k3_vs_moe4_delta_pct)

    print(f"\n  Top 5 Neural standalone mejora vs MoE 4 (MAE Δ%):")
    for r in sorted_nn[:5]:
        print(f"    {r.equipment:12s} / {r.parameter[:30]:30s}  "
              f"MAE: {r.moe4_mae:>10.4f} → {r.neural_mae:>10.4f}  ({r.neural_vs_moe4_delta_pct:+.2f}%)")

    print(f"\n  Top 5 Neural standalone empeora vs MoE 4 (MAE Δ%):")
    for r in sorted_nn[-5:]:
        print(f"    {r.equipment:12s} / {r.parameter[:30]:30s}  "
              f"MAE: {r.moe4_mae:>10.4f} → {r.neural_mae:>10.4f}  ({r.neural_vs_moe4_delta_pct:+.2f}%)")

    print(f"\n  Top 5 MoE 5 k=3 mejora vs MoE 4 (MAE Δ%):")
    for r in sorted_5k3[:5]:
        if abs(r.moe5k3_vs_moe4_delta_pct) > 0.01:
            print(f"    {r.equipment:12s} / {r.parameter[:30]:30s}  "
                  f"MAE: {r.moe4_mae:>10.4f} → {r.moe5k3_mae:>10.4f}  ({r.moe5k3_vs_moe4_delta_pct:+.2f}%)")

    print(f"\n  Top 5 MoE 5 k=3 empeora vs MoE 4 (MAE Δ%):")
    for r in sorted_5k3[-5:]:
        if abs(r.moe5k3_vs_moe4_delta_pct) > 0.01:
            print(f"    {r.equipment:12s} / {r.parameter[:30]:30s}  "
                  f"MAE: {r.moe4_mae:>10.4f} → {r.moe5k3_mae:>10.4f}  ({r.moe5k3_vs_moe4_delta_pct:+.2f}%)")

    # Neural weight summary
    total_preds = sum(r.n_predictions for r in results)
    total_selected_k2 = sum(r.neural_selected_count_k2 for r in results)
    total_selected_k3 = sum(r.neural_selected_count_k3 for r in results)

    if all_weight_traces:
        print(f"\n── Evolución de peso de 'neural' en MoE 5 (gating_probs, antes de top-k) ──")
        by_regime: Dict[str, List[float]] = {}
        for t in all_weight_traces:
            r = t["regime"]
            if r not in by_regime:
                by_regime[r] = []
            by_regime[r].append(t["mean_weight"])
        for regime, means in sorted(by_regime.items()):
            print(f"  {regime:12s}: mean_weight={np.mean(means):.4f}  "
                  f"min={np.min(means):.4f}  max={np.max(means):.4f}  "
                  f"sobre {len(means)} series")

        pct_k2 = (total_selected_k2 / total_preds * 100) if total_preds > 0 else 0.0
        pct_k3 = (total_selected_k3 / total_preds * 100) if total_preds > 0 else 0.0
        if total_selected_k2 == 0:
            print(f"\n  Neural NUNCA seleccionado en top-2 (k=2): 0/{total_preds}")
        else:
            print(f"\n  Neural en top-2 (k=2): {total_selected_k2}/{total_preds} ({pct_k2:.1f}%)")
        print(f"  Neural en top-3 (k=3): {total_selected_k3}/{total_preds} ({pct_k3:.1f}%)")
    else:
        print(f"\n  No se registraron pesos para 'neural'")

    print("=" * 70)

    # ── Save ──
    elapsed = time.time() - start
    output = {
        "metadata": {
            "test": "A/B Clean — split temporal 80/20, MoE 4 vs Neural standalone vs MoE 5",
            "window_size": WINDOW_SIZE,
            "split_frac": SPLIT_FRAC,
            "neural_config": {"input_size": 10, "hidden_size": 6, "lr": 0.001, "epochs": 300},
            "generated_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
        },
        "global": global_metrics,
        "verdicts": {
            "paso_a_neural_standalone": verdict_a,
            "paso_b_moe5_k2": verdict_b_k2,
            "paso_b_moe5_k3": verdict_b_k3,
        },
        "neural_weight_summary": {
            "by_regime": {
                regime: {
                    "mean": round(float(np.mean(means)), 4),
                    "min": round(float(np.min(means)), 4),
                    "max": round(float(np.max(means)), 4),
                    "n_series": len(means),
                }
                for regime, means in by_regime.items()
            },
            "selected_top2": total_selected_k2,
            "selected_top3": total_selected_k3,
            "total_predictions": total_preds,
            "selection_rate_k2_pct": round(total_selected_k2 / total_preds * 100, 1) if total_preds > 0 else 0,
            "selection_rate_k3_pct": round(total_selected_k3 / total_preds * 100, 1) if total_preds > 0 else 0,
        } if all_weight_traces else {"note": "neural_never_selected_in_top2"},
        "per_series": [asdict(r) for r in sorted(results, key=lambda x: (x.equipment, x.parameter))],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    main()
