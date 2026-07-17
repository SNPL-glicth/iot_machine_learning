# UCI AI4I 2020 Benchmark — Predictive Maintenance Dataset

**Dataset:** `UCI AI4I 2020 Predictive Maintenance Dataset`
**Total points:** 10,000
**Anomaly points:** 339 (3.39%)

## Results

| Detector | F1 | Precision | Recall | AUC-ROC | AUC-PR | FP | FN | Speed (pts/s) |
|----------|-----|-----------|--------|---------|------------|----|--------------|
| ZENIN VotingEnsemble 🏆 | 0.0611 | 0.0420 | 0.1121 | 0.5812 | 0.0448 | 867 | 301 | 81 |
| Rolling Z-score | 0.0243 | 0.0390 | 0.0177 | 0.5247 | 0.0362 | 148 | 333 | 29,264 |
| Z-score (global) | 0.0000 | 0.0000 | 0.0000 | 0.5945 | 0.0430 | 0 | 339 | 0 |
| IQR (global) | 0.0000 | 0.0000 | 0.0000 | 0.5852 | 0.0413 | 0 | 339 | 0 |

## Interpretation

✅ **ZENIN wins** with F1=0.0611 vs best baseline F1=0.0243

## Dataset Description

The UCI AI4I 2020 Predictive Maintenance dataset contains sensor data from machines with various failure types including tool wear, heat dissipation failure, power failure, overstrain failure, and random failures. It includes air temperature, process temperature, rotational speed, torque, and tool wear measurements.
