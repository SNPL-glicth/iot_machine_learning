# SKAB Benchmark — Skoltech Anomaly Benchmark

**Dataset:** `SKAB (Skoltech Anomaly Benchmark)`
**Total points:** 5,937
**Anomaly points:** 2,039 (34.34%)

## Results

| Detector | F1 | Precision | Recall | AUC-ROC | AUC-PR | FP | FN | Speed (pts/s) |
|----------|-----|-----------|--------|---------|------------|----|--------------|
| ZENIN VotingEnsemble 🏆 | 0.1136 | 0.3538 | 0.0677 | 0.8187 | 0.6743 | 252 | 1901 | 67 |
| Rolling Z-score | 0.0162 | 0.3091 | 0.0083 | 0.6348 | 0.4068 | 38 | 2022 | 21,803 |
| Z-score (global) | 0.0000 | 0.0000 | 0.0000 | 0.7145 | 0.4495 | 0 | 2039 | 0 |
| IQR (global) | 0.0000 | 0.0000 | 0.0000 | 0.7016 | 0.4433 | 0 | 2039 | 0 |

## Interpretation

✅ **ZENIN wins** with F1=0.1136 vs best baseline F1=0.0162

## Dataset Description

The SKAB (Skoltech Anomaly Benchmark) dataset contains real-world industrial sensor data with labeled anomalies from various equipment including valves and other industrial components. It includes different types of anomalies such as leaks, blockages, and sensor failures.
