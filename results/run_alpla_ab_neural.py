"""A/B test — NeuralExpert standalone vs fused MoE (baseline).

Loads the baseline JSON, runs sliding-window evaluation comparing:
A) Fused MoE (4 experts) — already in baseline JSON
B) NeuralExpert standalone — trained per-series, predicting alone

Compares per-series MAE/nMAE% and produces a verdict.
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

from infrastructure.ml.moe.experts.neural_expert import create_neural_expert, NeuralExpert

WINDOW_SIZE = 15

BASELINE_PATH = os.path.join(_SCRIPT_DIR, "baseline_moe_4experts.json")
OUTPUT_PATH = os.path.join(_SCRIPT_DIR, "ab_neural_vs_baseline.json")


def load_baseline() -> Dict[str, Any]:
    with open(BASELINE_PATH) as f:
        return json.load(f)


def build_baseline_lookup(baseline: Dict[str, Any]) -> Dict[str, Any]:
    lookup = {}
    for s in baseline["per_series"]:
        key = f"{s['equipment']}///{s['parameter']}"
        lookup[key] = s
    return lookup


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


def load_and_pivot(excel_path):
    sheets = pd.read_excel(excel_path, sheet_name=None)
    series_list = []
    for sheet_name, df in sheets.items():
        if all(c in df.columns for c in ("Equipo", "Parámetro", "Valor")):
            df_pivot = df.pivot_table(index="Fecha", columns="Parámetro", values="Valor", aggfunc="first").sort_index()
        else:
            skip = {"FECHA", "HORA", "FECHA Y HORA", "EQUIPO", "FECHA/HORA", "Fecha/Hora", "DATETIME", "timestamp", "fecha", "hora", "fecha_hora", "equipo", "indice", "index"}
            param_cols = [c for c in df.columns if c not in skip and df[c].dtype in ("float64", "int64")]
            df_pivot = df[param_cols]
        for col in df_pivot.columns:
            values = df_pivot[col].dropna().values.tolist()
            if len(values) >= WINDOW_SIZE + 2:
                series_list.append({"equipment": sheet_name, "parameter": str(col).strip(), "values": values, "n_points": len(values)})
    return series_list


@dataclass
class SeriesComparison:
    equipment: str
    parameter: str
    n_points: int
    n_predictions: int
    # MoE baseline
    moe_mae: float = 0.0
    moe_nmae_pct: float = 0.0
    # Neural standalone
    neural_mae: float = 0.0
    neural_nmae_pct: float = 0.0
    # Delta
    mae_delta_pct: float = 0.0  # (neural - moe) / moe * 100  = negative means neural is better
    regime_distribution: Dict[str, int] = field(default_factory=dict)


def main():
    start = time.time()

    # 1. Load baseline
    print("Loading baseline (4 experts)...")
    baseline = load_baseline()
    baseline_lookup = build_baseline_lookup(baseline)
    print(f"  Baseline: {baseline['global']['n_series']} series, median_nmae={baseline['global']['median_nmae_pct']}%")

    # 2. Load dataset
    excel_path = os.path.join(_SCRIPT_DIR, "Información Chiller y CA - ZENIN.xlsx")
    series_list = load_and_pivot(excel_path)
    print(f"  → {len(series_list)} series")

    # 3. Evaluate each series
    results: List[SeriesComparison] = []

    for idx, sd in enumerate(series_list):
        values = sd["values"]
        n = len(values)
        key = f"{sd['equipment']}///{sd['parameter']}"
        bl = baseline_lookup.get(key, {})

        # Create a fresh NeuralExpert for this series
        expert = create_neural_expert()
        series_id = f"{sd['equipment']}_{sd['parameter']}"

        # Pre-train on full data
        trained = expert.warmup(series_id, values)
        if not trained:
            continue

        # Sliding window evaluation
        neural_errs = []
        regimes = {}

        for i in range(WINDOW_SIZE, n):
            window_vals = values[i - WINDOW_SIZE: i]
            actual = values[i]
            ctx_regime = classify_regime(window_vals)
            regimes[ctx_regime] = regimes.get(ctx_regime, 0) + 1

            try:
                from iot_machine_learning.domain.entities.iot.sensor_reading import Reading
                readings = [Reading(series_id=series_id, value=v, timestamp=float(t)) for t, v in enumerate(window_vals)]
                from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
                sw = SensorWindow(series_id=series_id, readings=readings)
                out = expert.predict(sw)
                err = abs(out.prediction - actual)
                if math.isfinite(err):
                    neural_errs.append(err)
            except Exception:
                continue

        if len(neural_errs) < 2:
            continue

        # Compute neural metrics
        neural_mae = float(np.mean(neural_errs))
        neural_rmse = float(np.sqrt(np.mean(np.array(neural_errs) ** 2)))
        series_mean = float(np.mean(values[WINDOW_SIZE:])) if len(values) > WINDOW_SIZE else 1.0
        neural_nmae = (neural_mae / (abs(series_mean) + 1e-10)) * 100

        # Baseline metrics from JSON
        bl_mae = bl.get("mae", neural_mae)
        bl_nmae = bl.get("nmae_pct", neural_nmae)
        mae_delta = ((neural_mae - bl_mae) / (bl_mae + 1e-10)) * 100

        results.append(SeriesComparison(
            equipment=sd["equipment"],
            parameter=sd["parameter"],
            n_points=n,
            n_predictions=len(neural_errs),
            moe_mae=round(bl_mae, 4),
            moe_nmae_pct=round(bl_nmae, 2),
            neural_mae=round(neural_mae, 4),
            neural_nmae_pct=round(neural_nmae, 2),
            mae_delta_pct=round(mae_delta, 2),
            regime_distribution=regimes,
        ))

        if (idx + 1) % 10 == 0:
            print(f"  ... {idx + 1}/{len(series_list)} done")

    # 4. Compute global metrics
    n = len(results)
    moe_maes = np.array([r.moe_mae for r in results])
    moe_nmaes = np.array([r.moe_nmae_pct for r in results])
    neural_maes = np.array([r.neural_mae for r in results])
    neural_nmaes = np.array([r.neural_nmae_pct for r in results])

    global_metrics = {
        "n_series": n,
        "total_predictions": sum(r.n_predictions for r in results),
        "moe_baseline": {
            "median_mae": round(float(np.median(moe_maes)), 4),
            "median_nmae_pct": round(float(np.median(moe_nmaes)), 2),
            "avg_mae": round(float(np.mean(moe_maes)), 4),
            "avg_nmae_pct": round(float(np.mean(moe_nmaes)), 2),
        },
        "neural_standalone": {
            "median_mae": round(float(np.median(neural_maes)), 4),
            "median_nmae_pct": round(float(np.median(neural_nmaes)), 2),
            "avg_mae": round(float(np.mean(neural_maes)), 4),
            "avg_nmae_pct": round(float(np.mean(neural_nmaes)), 2),
        },
    }

    # 5. Verdict
    improved = sum(1 for r in results if r.neural_mae < r.moe_mae - 1e-10)
    worsened = sum(1 for r in results if r.neural_mae > r.moe_mae + 1e-10)
    neutral = n - improved - worsened

    med_moe = global_metrics["moe_baseline"]["median_nmae_pct"]
    med_nn = global_metrics["neural_standalone"]["median_nmae_pct"]

    if med_nn < med_moe:
        verdict = f"✅ NeuralExpert MEJORA mediana nMAE: {med_moe}% → {med_nn}% (Δ {med_nn - med_moe:+.2f}pp)"
    elif med_nn == med_moe:
        verdict = f"⚖️ NeuralExpert NEUTRO en mediana nMAE: ambos {med_moe}%"
    else:
        verdict = f"❌ NeuralExpert EMPEORA mediana nMAE: {med_moe}% → {med_nn}% (Δ {med_nn - med_moe:+.2f}pp)"

    # 6. Print
    print("\n" + "=" * 70)
    print(" VEREDICTO A/B — NeuralExpert standalone vs MoE fused baseline")
    print("=" * 70)
    print(f"{'Métrica':<35} {'MoE baseline':>14} {'Neural':>14} {'Δ%':>10}")
    print("-" * 70)
    for metric in ["median_mae", "median_nmae_pct", "avg_mae", "avg_nmae_pct"]:
        bv = global_metrics["moe_baseline"][metric]
        nv = global_metrics["neural_standalone"][metric]
        delta = ((nv - bv) / (bv + 1e-10)) * 100
        sign = "+" if delta > 0 else ""
        print(f"{metric:<35} {bv:>14.4f} {nv:>14.4f} {sign}{delta:>8.2f}%")
    print("-" * 70)
    print(f"Series donde Neural mejora:   {improved}/{n}")
    print(f"Series donde Neural empeora: {worsened}/{n}")
    print(f"Series neutral:               {neutral}/{n}")
    print("=" * 70)
    print(verdict)
    print("=" * 70)

    # Top improvements
    sorted_by_delta = sorted(results, key=lambda r: r.mae_delta_pct)
    print("\n  Top 5 mejoras (Neural reduce MAE% más):")
    for r in sorted_by_delta[:5]:
        print(f"    {r.equipment:12s} / {r.parameter[:30]:30s}  "
              f"MAE: {r.moe_mae:>10.4f} → {r.neural_mae:>10.4f}  ({r.mae_delta_pct:+.2f}%)")

    print("\n  Top 5 regresiones (Neural aumenta MAE% más):")
    for r in sorted_by_delta[-5:]:
        print(f"    {r.equipment:12s} / {r.parameter[:30]:30s}  "
              f"MAE: {r.moe_mae:>10.4f} → {r.neural_mae:>10.4f}  ({r.mae_delta_pct:+.2f}%)")

    # 7. Save
    elapsed = time.time() - start
    output = {
        "metadata": {
            "test": "A/B — NeuralExpert standalone vs MoE fused (4 experts)",
            "window_size": WINDOW_SIZE,
            "dataset_md5": baseline["metadata"]["dataset"]["md5_prefix"],
            "neural_config": {"input_size": 10, "hidden_size": 6, "lr": 0.001, "epochs": 300},
            "generated_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
        },
        "global": global_metrics,
        "verdict": verdict,
        "per_series": [asdict(r) for r in sorted(results, key=lambda x: (x.equipment, x.parameter))],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_PATH}")

    return output


if __name__ == "__main__":
    main()
