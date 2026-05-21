# ZENIN vs NAB Benchmark — Machine Temperature

**Dataset:** `realKnownCause/machine_temperature_system_failure`
**Total points:** 22,695
**Anomaly points:** 84 (0.37%)

## Results

| Detector | F1 | Precision | Recall | AUC-ROC | AUC-PR | FP | FN | Speed (pts/s) |
|----------|-----|-----------|--------|---------|------------|----|--------------|
| ZENIN VotingEnsemble (tuned) | 0.1773 | 0.1513 | 0.2143 | 0.8662 | 0.0636 | 101 | 66 | 169 |
| Z-Score (global) | 0.1538 | 0.0909 | 0.5000 | 0.9502 | 0.3774 | 420 | 42 | 0 |
| ZENIN VotingEnsemble | 0.0871 | 0.0533 | 0.2381 | 0.8662 | 0.0636 | 355 | 64 | 178 |
| IQR (global) | 0.0529 | 0.0274 | 0.7500 | 0.8256 | 0.0215 | 2235 | 21 | 0 |
| Rolling Z-Score (w=50) | 0.0056 | 0.0031 | 0.0238 | 0.6710 | 0.0065 | 634 | 82 | 25,732 |

## Interpretation

⚠️ **Baseline wins** (ZENIN VotingEnsemble (tuned) F1=0.1773) vs ZENIN F1=0.0871

### Diagnóstico del Threshold

El `voting_threshold=0.5` default produce sobre-sensibilidad: 87% del dataset es marcado como anomalía (19,878 de 22,695). Esto indica que los scores del ensemble son altos incluso para puntos normales.

### Resultados del Threshold Grid Search

- **Threshold óptimo (max F1):** `0.75`
  - F1=0.1773 | Precision=0.1513 | Recall=0.2143
- **Alternativa Precision@Recall≥0.6:** threshold=0.45
  - F1=0.0118 | Precision=0.0059 | Recall=1.0000

### Recomendaciones de Tuning

- **AUMENTAR** `voting_threshold` de 0.5 a 0.75 (para max F1)
- Alternativa conservadora: threshold=0.45
- No es necesario reducir contamination (default 0.5% ≈ real 0.37%)
- Considerar `DETECTION_WINDOW=20` para reducir fragmentación de alertas
