"""A/B baseline: sliding-window evaluation of MoE (4 experts) over ALPLA dataset.

Computes per-series and global MAE/RMSE by predicting each point via
a sliding window and comparing against the actual ground-truth value.

Output: results/baseline_moe_4experts.json (versioned, with dataset hash)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
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
from infrastructure.ml.engines.core.factory import EngineFactory
from infrastructure.ml.moe.expert_wrappers.engine_adapter import (
    create_baseline_expert,
    create_kalman_expert,
    create_statistical_expert,
    create_taylor_expert,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SIZE: int = 15  # points per sliding window
SPARSITY_K: int = 2     # top-k experts to execute


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_and_pivot(excel_path: str) -> List[Dict[str, Any]]:
    """Load ALPLA Excel, pivot each sheet to wide format, return per-parameter series.

    Returns list of dicts: {equipment, parameter, values, n_points}
    """
    sheets = pd.read_excel(excel_path, sheet_name=None)
    series_list: List[Dict[str, Any]] = []

    for sheet_name, df in sheets.items():
        # Detect columns for long-format pivot
        equipo_col = "Equipo"
        parametro_col = "Parámetro"
        valor_col = "Valor"

        if all(c in df.columns for c in (equipo_col, parametro_col, valor_col)):
            df_pivot = df.pivot_table(
                index="Fecha",
                columns=parametro_col,
                values=valor_col,
                aggfunc="first",
            ).sort_index()
        else:
            skip = {"FECHA", "HORA", "FECHA Y HORA", "EQUIPO", "FECHA/HORA",
                    "Fecha/Hora", "DATETIME", "timestamp", "fecha", "hora",
                    "fecha_hora", "equipo", "indice", "index"}
            param_cols = [c for c in df.columns if c not in skip and df[c].dtype in ("float64", "int64")]
            df_pivot = df[param_cols]

        for col in df_pivot.columns:
            values = df_pivot[col].dropna().values.tolist()
            if len(values) < WINDOW_SIZE + 2:
                continue
            series_list.append({
                "equipment": sheet_name,
                "parameter": str(col).strip(),
                "values": values,
                "n_points": len(values),
            })

    return series_list


# ---------------------------------------------------------------------------
# Feature context (reused from pipeline)
# ---------------------------------------------------------------------------

def classify_regime(values: List[float]) -> str:
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


def build_feature_context(values: List[float]) -> FeatureContext:
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


# ---------------------------------------------------------------------------
# MoE engine creation
# ---------------------------------------------------------------------------

def _try_register(registry: ExpertRegistry, name: str, create_fn, engine_name: str) -> bool:
    try:
        engine = EngineFactory.create(engine_name)
        expert = create_fn(engine.as_port())
        registry.register(name, expert, expert.capabilities)
        return True
    except Exception as exc:
        logger.warning("Failed to register %s: %s", name, exc)
        return False


def create_moe_engine() -> Optional[MoEPredictionEngine]:
    registry = ExpertRegistry()
    names = [
        ("baseline", create_baseline_expert, "baseline_moving_average"),
        ("statistical", create_statistical_expert, "statistical"),
        ("taylor", create_taylor_expert, "taylor"),
        ("kalman", create_kalman_expert, "kalman"),
    ]
    for n, fn, en in names:
        _try_register(registry, n, fn, en)
    if len(registry) == 0:
        return None
    gating = ContextualRegimeGating(expert_ids=registry.list_all())
    fusion = DiscrepancyAwareFusion()
    return MoEPredictionEngine(
        registry=registry,
        gating=gating,
        fusion=fusion,
        fallback_engine=None,
        sparsity_k=SPARSITY_K,
    )


# ---------------------------------------------------------------------------
# Sliding-window evaluation
# ---------------------------------------------------------------------------

@dataclass
class SeriesResult:
    equipment: str
    parameter: str
    n_points: int
    n_predictions: int
    mae: float
    rmse: float
    mape_pct: float
    nmae_pct: float  # normalized MAE = MAE / mean(values) * 100
    mean_value: float
    regime_distribution: Dict[str, int] = field(default_factory=dict)


def evaluate_series(series: Dict[str, Any], engine: MoEPredictionEngine) -> Optional[SeriesResult]:
    values = series["values"]
    n = len(values)
    errors: List[float] = []
    regimes: Dict[str, int] = {}
    predicted_vals: List[float] = []
    actual_vals: List[float] = []

    for i in range(WINDOW_SIZE, n):
        window = values[i - WINDOW_SIZE: i]
        actual = values[i]

        ctx = build_feature_context(window)
        regimes[ctx.regime] = regimes.get(ctx.regime, 0) + 1

        try:
            pred_result = engine.predict(
                values=window,
                feature_context=ctx,
                series_id=f"{series['equipment']}_{series['parameter']}_{i}",
            )
            error = abs(pred_result.predicted_value - actual)
            if math.isfinite(error):
                errors.append(error)
                predicted_vals.append(pred_result.predicted_value)
                actual_vals.append(actual)
        except Exception:
            continue

    if len(errors) < 2:
        return None

    arr = np.array(errors)
    mae = float(np.mean(arr))
    rmse = float(np.sqrt(np.mean(arr ** 2)))

    # MAPE
    actual_arr = np.array(actual_vals)
    pred_arr = np.array(predicted_vals)
    denom = np.abs(actual_arr) + 1e-10
    mape = float(np.mean(np.abs(actual_arr - pred_arr) / denom)) * 100
    mape = min(mape, 1000.0)

    # Normalized MAE (nMAE%): MAE / mean_of_series * 100
    series_mean = np.mean(actual_arr) if len(actual_arr) > 0 else 1.0
    nmae = (mae / (abs(series_mean) + 1e-10)) * 100

    return SeriesResult(
        equipment=series["equipment"],
        parameter=series["parameter"],
        n_points=n,
        n_predictions=len(errors),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        mape_pct=round(mape, 2),
        nmae_pct=round(nmae, 2),
        mean_value=round(float(series_mean), 4),
        regime_distribution=regimes,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    results: List[SeriesResult],
    excel_path: str,
    elapsed: float,
) -> Dict[str, Any]:
    total_preds = sum(r.n_predictions for r in results)

    mae_vals = np.array([r.mae for r in results])
    rmse_vals = np.array([r.rmse for r in results])
    mape_vals = np.array([r.mape_pct for r in results])
    nmae_vals = np.array([r.nmae_pct for r in results])

    global_mae = round(float(np.mean(mae_vals)), 4) if len(mae_vals) > 0 else None
    median_mae = round(float(np.median(mae_vals)), 4) if len(mae_vals) > 0 else None
    global_rmse = round(float(np.sqrt(np.mean(rmse_vals ** 2))), 4) if len(rmse_vals) > 0 else None
    global_mape = round(float(np.mean(mape_vals)), 2) if len(mape_vals) > 0 else None
    global_nmae = round(float(np.mean(nmae_vals)), 2) if len(nmae_vals) > 0 else None
    median_nmae = round(float(np.median(nmae_vals)), 2) if len(nmae_vals) > 0 else None

    total_abs_err = sum(r.mae * r.n_predictions for r in results)
    weighted_mae = round(total_abs_err / total_preds, 4) if total_preds else None

    per_series = []
    for r in sorted(results, key=lambda x: (x.equipment, x.parameter)):
        per_series.append(asdict(r))

    return {
        "metadata": {
            "test": "A/B Baseline — MoE 4 experts (sliding window)",
            "window_size": WINDOW_SIZE,
            "sparsity_k": SPARSITY_K,
            "experts": ["baseline", "statistical", "taylor", "kalman"],
            "dataset": {
                "file": os.path.basename(excel_path),
                "md5_prefix": _file_hash(excel_path),
            },
            "generated_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
        },
        "global": {
            "n_series": len(results),
            "total_predictions": total_preds,
            "avg_mae_per_series": global_mae,
            "median_mae": median_mae,
            "weighted_mae": weighted_mae,
            "global_rmse": global_rmse,
            "avg_mape_pct": global_mape,
            "avg_nmae_pct": global_nmae,
            "median_nmae_pct": median_nmae,
        },
        "per_series": per_series,
    }


def print_summary(report: Dict[str, Any]) -> None:
    g = report["global"]
    m = report["metadata"]
    print("=" * 60)
    print(" A/B BASELINE — MoE 4 experts")
    print("=" * 60)
    print(f"  Dataset:         {m['dataset']['file']}  (md5:{m['dataset']['md5_prefix']})")
    print(f"  Window size:     {m['window_size']}")
    print(f"  Sparsity k:      {m['sparsity_k']}")
    print(f"  Series:          {g['n_series']}")
    print(f"  Predictions:     {g['total_predictions']}")
    print(f"  Avg MAE:         {g['avg_mae_per_series']}")
    print(f"  Median MAE:      {g['median_mae']}   ← robusto a outliers")
    print(f"  Weighted MAE:    {g['weighted_mae']}")
    print(f"  Global RMSE:     {g['global_rmse']}")
    print(f"  Avg MAPE:        {g['avg_mape_pct']}%")
    print(f"  Avg nMAE:        {g['avg_nmae_pct']}%   ← MAE/mean*100")
    print(f"  Median nMAE:     {g['median_nmae_pct']}%")
    print("=" * 60)

    sorted_by_mae = sorted(report["per_series"], key=lambda x: x["mae"], reverse=True)
    print("\n  Worst 5 (highest MAE / highest nMAE%):")
    for s in sorted_by_mae[:5]:
        print(f"    {s['equipment']:12s} / {s['parameter'][:30]:30s}  "
              f"MAE={s['mae']:>10.4f}  nMAE={s['nmae_pct']:>6.2f}%  n={s['n_predictions']}")
    print("\n  Best 5 (lowest MAE) — typically constant series:")
    for s in sorted_by_mae[-5:]:
        print(f"    {s['equipment']:12s} / {s['parameter'][:30]:30s}  "
              f"MAE={s['mae']:>10.4f}  nMAE={s['nmae_pct']:>6.2f}%  n={s['n_predictions']}")

    # Worst by nMAE (normalized) — different picture
    sorted_by_nmae = sorted(report["per_series"], key=lambda x: x["nmae_pct"], reverse=True)
    print("\n  Worst 5 by nMAE% (normalized — highest relative error):")
    for s in sorted_by_nmae[:5]:
        print(f"    {s['equipment']:12s} / {s['parameter'][:30]:30s}  "
              f"nMAE={s['nmae_pct']:>6.2f}%  MAE={s['mae']:>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = time.time()
    excel_path = os.path.join(_SCRIPT_DIR, "Información Chiller y CA - ZENIN.xlsx")

    if not os.path.exists(excel_path):
        print(f"ERROR: dataset not found at {excel_path}")
        sys.exit(1)

    print(f"Loading dataset: {excel_path}")
    series_list = load_and_pivot(excel_path)
    print(f"  → {len(series_list)} parameter-series found (≥{WINDOW_SIZE + 1} points each)")

    print("Creating MoE engine (4 experts)...")
    engine = create_moe_engine()
    if engine is None:
        print("ERROR: MoE engine creation failed")
        sys.exit(1)
    print(f"  Registered: {engine._registry.list_all()}")

    print(f"Running sliding-window evaluation (window={WINDOW_SIZE})...")
    results: List[SeriesResult] = []
    for idx, sd in enumerate(series_list):
        label = f"{sd['equipment']} / {sd['parameter']}"
        result = evaluate_series(sd, engine)
        if result is not None:
            results.append(result)
        if (idx + 1) % 10 == 0:
            print(f"  ... {idx + 1}/{len(series_list)} series done")

    elapsed = time.time() - start
    report = build_report(results, excel_path, elapsed)

    output_path = os.path.join(_SCRIPT_DIR, "baseline_moe_4experts.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")

    print_summary(report)
    return report


if __name__ == "__main__":
    report = main()
