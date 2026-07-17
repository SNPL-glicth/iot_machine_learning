"""ZENIN on all 47 ALPLA parameters."""
import json, os, sys, time, warnings, logging
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _p in (_PARENT, os.path.dirname(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from iot_machine_learning.domain.entities.iot.sensor_reading import Reading, TimeSeriesWindow
from iot_machine_learning.infrastructure.ml.anomaly.core.detector import VotingAnomalyDetector
from iot_machine_learning.infrastructure.ml.anomaly.core.config import AnomalyDetectorConfig

W = 50
TH = 0.5
SID = "alpla"

def run_one(vals, ts):
    n = len(vals)
    if n <= W:
        return [], []
    cfg = AnomalyDetectorConfig(voting_threshold=TH, contamination=0.005, min_training_points=W-2)
    det = VotingAnomalyDetector(config=cfg, series_id=SID, enable_adaptive_weights=False)
    det.train(vals[:W], ts[:W])
    preds, scores = [0]*n, [0.0]*n
    for i in range(W, n):
        slv = vals[i-W+1:i+1]
        slt = ts[i-W+1:i+1]
        rs = [Reading(series_id=SID, value=v, timestamp=t) for v,t in zip(slv, slt)]
        r = det.detect(TimeSeriesWindow(series_id=SID, readings=rs))
        preds[i] = int(r.is_anomaly)
        scores[i] = float(r.score)
    return preds, scores

def main():
    t0 = time.time()
    all_res = []
    for sheet, fn in [("Chiller","chiller_with_anomalies.csv"),("CA","ca_with_anomalies.csv")]:
        path = os.path.join(_SCRIPT_DIR, fn)
        df = pd.read_csv(path, parse_dates=["Fecha"])
        ts = (df["Fecha"].astype("int64").values // 10**9).tolist()
        cols = [c for c in df.columns if c not in ("Fecha","iso_anomaly")]
        print(f"  {sheet}: {len(cols)} params, {len(df)} rows")
        for ci, col in enumerate(cols):
            vals = df[col].dropna().values.tolist()
            if len(vals) < W + 2:
                continue
            local_ts = ts[:len(vals)]
            preds, scores = run_one(vals, local_ts)
            if not scores:
                continue
            arr = np.array(scores)
            det05 = int(sum(preds))
            n_det = sum(1 for s in scores if s > 0)
            r = {
                "equipment": sheet,
                "parameter": col.strip(),
                "n_points": len(scores),
                "n_unique_values": int(len(np.unique(vals))),
                "score_mean": round(float(np.mean(arr)), 4),
                "score_std": round(float(np.std(arr)), 4),
                "score_p50": round(float(np.median(arr)), 4),
                "score_p25": round(float(np.percentile(arr, 25)), 4),
                "score_p75": round(float(np.percentile(arr, 75)), 4),
                "score_p90": round(float(np.percentile(arr, 90)), 4),
                "score_p95": round(float(np.percentile(arr, 95)), 4),
                "score_p99": round(float(np.percentile(arr, 99)), 4),
                "score_max": round(float(np.max(arr)), 4),
                "score_min": round(float(np.min(arr)), 4),
                "det_fixed_0.5": det05,
                "n_scores_nonzero": n_det,
                "signal_mean": round(float(np.mean(vals)), 2),
                "signal_std": round(float(np.std(vals)), 2),
            }
            all_res.append(r)
            if (ci + 1) % 10 == 0 or ci == len(cols) - 1:
                print(f"    [{ci+1}/{len(cols)}] {col.strip()[:30]:30s} p50={r['score_p50']:.3f} p95={r['score_p95']:.3f} det05={det05}")
    elapsed = time.time() - t0
    print(f"\nTotal: {len(all_res)} params in {elapsed:.1f}s")

    for eq in ["Chiller","CA"]:
        pp = [r for r in all_res if r["equipment"] == eq]
        dets = [r["det_fixed_0.5"] for r in pp]
        print(f"  {eq}: sum(det@0.5)={sum(dets)}, per-param mean={np.mean(dets):.1f}")
        const = [r for r in pp if r["n_unique_values"] <= 3]
        print(f"    near-constant params: {len(const)} (n_unique<=3)")

    out = {
        "metadata": {
            "pipeline": "ZENIN ALPLA",
            "voting_threshold": TH,
            "window_size": W,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_parameters": len(all_res),
        },
        "parameters": all_res,
        "warnings": [
            "No ground truth labels for ALPLA. F1/precision/recall cannot be computed.",
            "Detection counts are unvalidated. Higher detections do not imply better detection.",
        ],
    }
    path = os.path.join(_SCRIPT_DIR, "zenin_alpla_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved to {path}")

if __name__ == "__main__":
    main()
