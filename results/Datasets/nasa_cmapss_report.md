# NASA C-MAPSS Benchmark — Turbofan Engine Degradation

**Dataset:** `NASA C-MAPSS FD001`
**Total points:** 42,273
**Anomaly points:** 100 (0.24%)
**Detection threshold (RUL):** 0.7 cycles

## Results

| Detector | F1 | Precision | Recall | AUC-ROC | AUC-PR | FP | FN | Speed (pts/s) |
|----------|-----|-----------|--------|---------|------------|----|--------------|
| ZENIN VotingEnsemble 🏆 | 0.0050 | 0.0025 | 0.9800 | 0.6543 | 0.0041 | 38827 | 2 | 106 |
| Rolling Z-score | 0.0046 | 0.0025 | 0.0300 | 0.4939 | 0.0023 | 1207 | 97 | 28,351 |
| Z-score (global) | 0.0000 | 0.0000 | 0.0000 | 0.8829 | 0.0092 | 0 | 100 | 0 |
| IQR (global) | 0.0000 | 0.0000 | 0.0000 | 0.8391 | 0.0067 | 0 | 100 | 4,224,300 |

## Interpretation

✅ **ZENIN wins** with F1=0.0050 vs best baseline F1=0.0046

## Dataset Description

The NASA C-MAPSS dataset contains sensor data from turbofan engines simulating different degradation profiles. This benchmark uses FD001 subset with sensor 2 (total temperature at fan inlet) as the main signal.
Anomalies are defined as points where Remaining Useful Life (RUL) falls below 0.7 cycles, indicating imminent failure.
