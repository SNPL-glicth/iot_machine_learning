#!/usr/bin/env python3
"""Forensic ensemble data collector for ZENIN ML anomaly detection.

Collects per-detector votes, scores, and ground-truth alignment
for comprehensive offline analysis. Does NOT modify any detector logic.
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, "/home/nicolas/Documentos/Iot_System")
sys.path.insert(0, "/home/nicolas/Documentos/Iot_System/iot_machine_learning")

from iot_machine_learning.domain.entities.iot.sensor_reading import (
    Reading, TimeSeriesWindow,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.detector import (
    VotingAnomalyDetector,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.config import (
    AnomalyDetectorConfig,
)
from iot_machine_learning.infrastructure.ml.anomaly.voting.strategy import VotingStrategy
from iot_machine_learning.infrastructure.ml.anomaly.detectors.rolling_z_detector import (
    RollingZScoreDetector,
)

NAB_ROOT = Path("/tmp/NAB")
DATASET_PATH = NAB_ROOT / "data/realKnownCause/machine_temperature_system_failure.csv"
LABELS_PATH = NAB_ROOT / "labels/combined_labels.json"
OUT_DIR = Path("benchmarks/results/forensic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE, DETECTION_WINDOW, SERIES_ID = 50, 10, "NAB-machine-temp-001"

BASE_WEIGHTS = {
    "isolation_forest": 0.2222,
    "z_score": 0.1778,
    "local_outlier_factor": 0.1333,
    "iqr": 0.0444,
    "velocity_z": 0.1333,
    "acceleration_z": 0.0889,
}

def load_data():
    df = (
        pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    values = df["value"].tolist()
    timestamps = [ts.timestamp() for ts in df["timestamp"]]
    with open(LABELS_PATH) as f:
        labels_map = json.load(f)
    anomaly_dts = pd.to_datetime(
        labels_map.get("realKnownCause/machine_temperature_system_failure.csv", [])
    )
    labels = [0] * len(df)
    for adt in anomaly_dts:
        idx = (df["timestamp"] - adt).abs().idxmin()
        start = max(0, idx - DETECTION_WINDOW)
        end = min(len(df), idx + DETECTION_WINDOW + 1)
        for i in range(start, end):
            labels[i] = 1
    return values, timestamps, labels


def build_custom_weights(rz_weight):
    other_sum = sum(BASE_WEIGHTS.values())
    scale = (1.0 - rz_weight) / other_sum
    weights = {k: v * scale for k, v in BASE_WEIGHTS.items()}
    weights["rolling_z"] = rz_weight
    return weights


def collect_forensic_data():
    """Run ensemble and collect per-detector votes + scores for every point."""
    values, timestamps, labels = load_data()
    labels_arr = np.array(labels[WINDOW_SIZE:])

    cfg = AnomalyDetectorConfig(voting_threshold=0.75)
    detector = VotingAnomalyDetector(
        config=cfg, series_id=SERIES_ID, enable_adaptive_weights=False
    )
    detector.train(values[:WINDOW_SIZE], timestamps=timestamps[:WINDOW_SIZE])

    # Collect fixed votes from default ensemble (without rolling_z)
    fixed_votes = []
    for i in range(WINDOW_SIZE, len(values)):
        slice_v = values[i - WINDOW_SIZE + 1 : i + 1]
        slice_t = timestamps[i - WINDOW_SIZE + 1 : i + 1]
        readings = [
            Reading(series_id=SERIES_ID, value=v, timestamp=t)
            for v, t in zip(slice_v, slice_t)
        ]
        window = TimeSeriesWindow(series_id=SERIES_ID, readings=readings)
        result = detector.detect(window)
        votes = dict(result.method_votes)
        votes.pop("rolling_z", None)
        fixed_votes.append(votes)

    scaler = detector._scaler
    if scaler is not None:
        values_scaled = scaler.transform(np.array(values).reshape(-1, 1)).flatten().tolist()
    else:
        values_scaled = values

    # Production config
    rz = RollingZScoreDetector(
        short_window=10, long_window=400, lower=3.5, upper=3.5, hysteresis=7
    )
    rz.train(values_scaled[:WINDOW_SIZE])

    custom_weights = build_custom_weights(0.20)
    strategy = VotingStrategy(weights=custom_weights, threshold=0.75, default_weight=0.1)

    # Collect per-point data
    records = []
    all_detector_names = set()
    for fv in fixed_votes:
        all_detector_names.update(fv.keys())
    all_detector_names.add("rolling_z")
    all_detector_names = sorted(all_detector_names)

    for i, fv in enumerate(fixed_votes):
        idx = WINDOW_SIZE + i
        scaled_value = values_scaled[idx]
        rz_vote = rz.vote(scaled_value)
        if rz_vote is None:
            rz_vote = 0.0

        votes = dict(fv)
        votes["rolling_z"] = rz_vote

        score = strategy.combine(votes)
        is_anomaly = strategy.is_anomaly(score)
        label = labels_arr[i]

        record = {
            "index": idx,
            "value": values[idx],
            "scaled_value": scaled_value,
            "score": float(score),
            "is_anomaly_pred": int(is_anomaly),
            "label": int(label),
            "rz_z_score": float(rz._audit_z_scores[i]) if i < len(rz._audit_z_scores) else None,
        }
        for dn in all_detector_names:
            record[f"vote_{dn}"] = float(votes.get(dn, 0.0))
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "forensic_per_point.csv", index=False)
    print(f"Saved {len(df)} records to {OUT_DIR / 'forensic_per_point.csv'}")
    return df, all_detector_names


def analyze_weight_interactions(df, detector_names):
    """Mathematical analysis of how weights interact."""
    custom_weights = build_custom_weights(0.20)

    # Marginal contribution: what each detector adds on average
    marginal = {}
    for dn in detector_names:
        col = f"vote_{dn}"
        w = custom_weights.get(dn, 0.1)
        marginal[dn] = {
            "weight": w,
            "avg_vote": float(df[col].mean()),
            "max_vote": float(df[col].max()),
            "min_vote": float(df[col].min()),
            "std_vote": float(df[col].std()),
            "marginal_contribution": float(df[col].mean() * w),
            "pct_nonzero": float((df[col] > 0).mean() * 100),
            "pct_one": float((df[col] >= 1.0).mean() * 100),
        }

    # Sensitivity: how much does score change if detector flips 0->1?
    sensitivity = {}
    for dn in detector_names:
        w = custom_weights.get(dn, 0.1)
        # If detector votes 0 vs 1, what's the score difference?
        sensitivity[dn] = {
            "score_delta_0_to_1": float(w / sum(custom_weights.values())),
            "score_delta_0_to_05": float(0.5 * w / sum(custom_weights.values())),
        }

    # Saturation analysis
    max_possible = sum(custom_weights.values()) / sum(custom_weights.values())  # = 1.0
    # But can we actually reach threshold with fewer than all detectors?
    threshold = 0.75
    coalition_analysis = {}
    for dn in detector_names:
        w = custom_weights.get(dn, 0.1)
        alone_score = w / sum(custom_weights.values())
        coalition_analysis[dn] = {
            "can_reach_threshold_alone": alone_score > threshold,
            "max_possible_with_only_this": alone_score,
            "votes_needed_if_all_others_1": max(0, (threshold * sum(custom_weights.values()) - (sum(custom_weights.values()) - w)) / w) if w > 0 else float('inf'),
        }

    result = {
        "marginal": marginal,
        "sensitivity": sensitivity,
        "coalition": coalition_analysis,
        "total_weight": sum(custom_weights.values()),
        "threshold": threshold,
    }
    with open(OUT_DIR / "weight_interactions.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved weight interaction analysis to {OUT_DIR / 'weight_interactions.json'}")
    return result


def analyze_score_distributions(df):
    """Forensic analysis of score distributions by class."""
    normal_df = df[df["label"] == 0]
    anomaly_df = df[df["label"] == 1]

    pred = df["is_anomaly_pred"].values
    label = df["label"].values
    tp_mask = (pred == 1) & (label == 1)
    fp_mask = (pred == 1) & (label == 0)
    fn_mask = (pred == 0) & (label == 1)
    tn_mask = (pred == 0) & (label == 0)

    def dist_stats(subdf):
        if len(subdf) == 0:
            return {}
        scores = subdf["score"].values
        return {
            "count": int(len(scores)),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "median": float(np.median(scores)),
            "p25": float(np.percentile(scores, 25)),
            "p75": float(np.percentile(scores, 75)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
            "range_used": float(scores.max() - scores.min()),
            "pct_above_0_5": float((scores > 0.5).mean() * 100),
            "pct_above_0_75": float((scores > 0.75).mean() * 100),
        }

    dists = {
        "normal": dist_stats(normal_df),
        "anomaly": dist_stats(anomaly_df),
        "tp": dist_stats(df[tp_mask]),
        "fp": dist_stats(df[fp_mask]),
        "fn": dist_stats(df[fn_mask]),
        "tn": dist_stats(df[tn_mask]),
    }

    # Overlap analysis
    if len(anomaly_df) > 0 and len(normal_df) > 0:
        a_scores = anomaly_df["score"].values
        n_scores = normal_df["score"].values
        greater = np.sum(a_scores[:, None] > n_scores[None, :])
        less = np.sum(a_scores[:, None] < n_scores[None, :])
        cliff = (greater - less) / (len(a_scores) * len(n_scores))
        dists["cliff_delta"] = float(cliff)
    else:
        dists["cliff_delta"] = 0.0

    # Compression analysis
    all_scores = df["score"].values
    hist, bins = np.histogram(all_scores, bins=20, range=(0, 1))
    dists["histogram"] = {
        "bins": bins.tolist(),
        "counts": hist.tolist(),
    }

    with open(OUT_DIR / "score_distributions.json", "w") as f:
        json.dump(dists, f, indent=2, default=str)
    print(f"Saved score distributions to {OUT_DIR / 'score_distributions.json'}")
    return dists


def analyze_detector_coordination(df, detector_names):
    """Analyze temporal detector coordination and redundancy."""
    corr_matrix = {}
    for d1 in detector_names:
        for d2 in detector_names:
            c1 = f"vote_{d1}"
            c2 = f"vote_{d2}"
            corr = df[c1].corr(df[c2])
            corr_matrix.setdefault(d1, {})[d2] = float(corr) if not pd.isna(corr) else 0.0

    # Redundancy: which detectors vote together most often?
    redundancy = {}
    for d1 in detector_names:
        for d2 in detector_names:
            if d1 >= d2:
                continue
            c1 = f"vote_{d1}"
            c2 = f"vote_{d2}"
            both_nonzero = ((df[c1] > 0) & (df[c2] > 0)).mean()
            both_one = ((df[c1] >= 1.0) & (df[c2] >= 1.0)).mean()
            redundancy[f"{d1}+{d2}"] = {
                "both_nonzero_pct": float(both_nonzero * 100),
                "both_one_pct": float(both_one * 100),
            }

    with open(OUT_DIR / "detector_coordination.json", "w") as f:
        json.dump({"correlation": corr_matrix, "redundancy": redundancy}, f, indent=2)
    print(f"Saved detector coordination to {OUT_DIR / 'detector_coordination.json'}")
    return corr_matrix, redundancy


def analyze_fp_fn_root_cause(df, detector_names):
    """Classify FP and FN by detector contributions."""
    pred = df["is_anomaly_pred"].values
    label = df["label"].values
    fp_mask = (pred == 1) & (label == 0)
    fn_mask = (pred == 0) & (label == 1)

    fp_df = df[fp_mask]
    fn_df = df[fn_mask]

    def classify_contributors(subdf):
        contributors = defaultdict(list)
        for dn in detector_names:
            col = f"vote_{dn}"
            contributors[dn] = {
                "avg_vote": float(subdf[col].mean()),
                "max_vote": float(subdf[col].max()),
                "voted_nonzero_pct": float((subdf[col] > 0).mean() * 100),
                "voted_one_pct": float((subdf[col] >= 1.0).mean() * 100),
            }
        return dict(contributors)

    result = {
        "fp_count": int(fp_mask.sum()),
        "fn_count": int(fn_mask.sum()),
        "fp_contributors": classify_contributors(fp_df) if len(fp_df) > 0 else {},
        "fn_contributors": classify_contributors(fn_df) if len(fn_df) > 0 else {},
    }

    with open(OUT_DIR / "fp_fn_root_cause.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved FP/FN root cause to {OUT_DIR / 'fp_fn_root_cause.json'}")
    return result


def analyze_threshold_feasibility(df, detector_names):
    """Analyze what detector coalitions can reach threshold."""
    custom_weights = build_custom_weights(0.20)
    threshold = 0.75
    total_w = sum(custom_weights.values())
    local_strategy = VotingStrategy(weights=custom_weights, threshold=0.75, default_weight=0.1)

    # Minimum votes needed from each detector to reach threshold
    # Assuming all other detectors vote 1.0
    min_needed = {}
    for dn in detector_names:
        w = custom_weights.get(dn, 0.1)
        others_max = sum(custom_weights.get(d, 1.0) for d in detector_names if d != dn)
        # w * v + others_max >= threshold * total_w
        # v >= (threshold * total_w - others_max) / w
        needed = (threshold * total_w - others_max) / w
        min_needed[dn] = {
            "needed_vote": float(needed),
            "is_possible": needed <= 1.0 and needed >= 0.0,
            "impossible_reason": "need_others" if needed < 0 else ("need_full" if needed > 1.0 else None),
        }

    # Actual coalition analysis on anomaly points
    anomaly_df = df[df["label"] == 1]
    coalition_scores = []
    for _, row in anomaly_df.iterrows():
        votes = {dn: row[f"vote_{dn}"] for dn in detector_names}
        score = local_strategy.combine(votes)
        # Which detectors actually contributed to pushing score above threshold?
        contributors = [dn for dn in detector_names if votes[dn] > 0.5]
        coalition_scores.append({
            "score": float(score),
            "above_threshold": score > threshold,
            "n_contributors": len(contributors),
            "contributors": contributors,
        })

    result = {
        "min_needed_per_detector": min_needed,
        "actual_anomaly_coalitions": coalition_scores,
        "threshold": threshold,
        "total_weight": total_w,
    }
    with open(OUT_DIR / "threshold_feasibility.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved threshold feasibility to {OUT_DIR / 'threshold_feasibility.json'}")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("FORENSIC ENSEMBLE DATA COLLECTION")
    print("=" * 70)

    df, detector_names = collect_forensic_data()
    print(f"\nDetectors: {detector_names}")
    print(f"Total points: {len(df)}")
    print(f"Anomaly points: {df['label'].sum()}")
    print(f"Predicted anomalies: {df['is_anomaly_pred'].sum()}")

    print("\n--- Weight Interaction Analysis ---")
    analyze_weight_interactions(df, detector_names)

    print("\n--- Score Distribution Analysis ---")
    analyze_score_distributions(df)

    print("\n--- Detector Coordination Analysis ---")
    analyze_detector_coordination(df, detector_names)

    print("\n--- FP/FN Root Cause Analysis ---")
    analyze_fp_fn_root_cause(df, detector_names)

    print("\n--- Threshold Feasibility Analysis ---")
    analyze_threshold_feasibility(df, detector_names)

    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print(f"Results in: {OUT_DIR}")
    print("=" * 70)
