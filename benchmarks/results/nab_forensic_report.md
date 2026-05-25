# ZENIN NAB Forensic Analysis

**Dataset:** `realKnownCause/machine_temperature_system_failure`
**Threshold:** 0.75
**Window size:** 50

## 1. Métricas Globales (post warm-up)

| Métrica | Valor |
|---------|-------|
| F1 | 0.1875 |
| Precision | 0.2727 |
| Recall | 0.1429 |
| TP | 12 |
| FP | 32 |
| FN | 72 |
| TN | 22529 |

## 2. Análisis de Falsos Negativos (FN)

**Total FN:** 72

### 2.1 Distribución por tipo

| Tipo | Count | % del total FN |
|------|-------|----------------|
| gradual | 72 | 100.0% |

### 2.2 Detalle de cada FN

| Idx | Valor | Score | Tipo | Votes |
|-----|-------|-------|------|-------|
| 2399 | 100.9526 | 0.3903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=1.00 |
| 2400 | 100.9087 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2401 | 99.8584 | 0.4403 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=1.00, acceleration_z=0.00 |
| 2402 | 101.3228 | 0.5403 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=1.00, acceleration_z=1.00 |
| 2403 | 101.8511 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2404 | 101.4601 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2405 | 101.9073 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2406 | 102.1787 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2407 | 101.3611 | 0.3657 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.50, acceleration_z=0.00 |
| 2408 | 101.2631 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2409 | 102.7362 | 0.5035 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=1.00, acceleration_z=0.63 |
| 2410 | 101.9778 | 0.4412 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.34, acceleration_z=1.00 |
| 2411 | 101.4234 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2412 | 102.0425 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2413 | 102.0594 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2414 | 102.1514 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2415 | 101.7576 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2416 | 101.6939 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2417 | 101.681 | 0.2903 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 2418 | 100.0538 | 0.5086 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=1.00, acceleration_z=0.68 |
| 2419 | 101.5506 | 0.5403 | gradual | z_score=0.00, iqr=0.00, isolation_forest=0.47, local_outlier_factor=1.00, velocity_z=1.00, acceleration_z=1.00 |
| 3977 | 18.592 | 0.7197 | gradual | z_score=1.00, iqr=1.00, isolation_forest=0.51, local_outlier_factor=1.00, velocity_z=0.14, acceleration_z=0.97 |
| 3982 | 10.002 | 0.7021 | gradual | z_score=1.00, iqr=1.00, isolation_forest=0.51, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=1.00 |
| 3984 | 6.4402 | 0.7021 | gradual | z_score=1.00, iqr=1.00, isolation_forest=0.51, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=1.00 |
| 3987 | 12.1204 | 0.6521 | gradual | z_score=0.00, iqr=1.00, isolation_forest=0.51, local_outlier_factor=1.00, velocity_z=1.00, acceleration_z=1.00 |
| 3990 | 41.1624 | 0.7021 | gradual | z_score=1.00, iqr=1.00, isolation_forest=0.51, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=1.00 |
| 3991 | 41.1408 | 0.6021 | gradual | z_score=1.00, iqr=1.00, isolation_forest=0.51, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 3992 | 40.7822 | 0.6021 | gradual | z_score=1.00, iqr=1.00, isolation_forest=0.51, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 3995 | 41.812 | 0.5229 | gradual | z_score=0.60, iqr=1.00, isolation_forest=0.51, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |
| 3996 | 42.5513 | 0.4924 | gradual | z_score=0.45, iqr=1.00, isolation_forest=0.51, local_outlier_factor=1.00, velocity_z=0.00, acceleration_z=0.00 |

*(42 FN adicionales en JSON)*

## 3. Contribution Analysis por Detector

| Detector | TP | FP | FN | TN | Precision | Recall | F1 | Vote Rate | Impact |
|----------|----|----|----|----|-----------|--------|----|-----------|--------|
| isolation_forest | 63 | 3244 | 21 | 19317 | 0.0191 | 0.7500 | 0.0372 | 0.1460 | neutral (+0.0018) |
| z_score | 19 | 333 | 65 | 22228 | 0.0540 | 0.2262 | 0.0872 | 0.0155 | helpful (-0.1760) |
| iqr | 63 | 8588 | 21 | 13973 | 0.0073 | 0.7500 | 0.0144 | 0.3820 | helpful (-0.0804) |
| local_outlier_factor | 84 | 10571 | 0 | 11990 | 0.0079 | 1.0000 | 0.0156 | 0.4705 | helpful (-0.0867) |
| velocity_z | 34 | 9131 | 50 | 13430 | 0.0037 | 0.4048 | 0.0074 | 0.4047 | neutral (+0.0038) |
| acceleration_z | 29 | 9329 | 55 | 13232 | 0.0031 | 0.3452 | 0.0061 | 0.4132 | helpful (-0.0221) |

## 4. Ablation Study

**Baseline F1:** 0.1875

| Excluido | F1 | Delta F1 | Delta P | Delta R | Impacto |
|----------|----|----------|---------|---------|---------|
| z_score ✅ | 0.0115 | -0.1760 | -0.2667 | -0.0119 | helpful |
| local_outlier_factor ✅ | 0.1008 | -0.0867 | -0.1013 | -0.0714 | helpful |
| iqr ✅ | 0.1071 | -0.0804 | -0.0584 | -0.0714 | helpful |
| acceleration_z ✅ | 0.1654 | -0.0221 | -0.0482 | -0.0119 | helpful |
| isolation_forest | 0.1893 | +0.0018 | -0.0845 | +0.0476 | neutral |
| velocity_z | 0.1913 | +0.0038 | +0.0821 | -0.0119 | neutral |

## 5. Interpretación y Recomendaciones

### Detectores tóxicos (quitarlos mejora F1)
- Ninguno. Todos los detectores aportan (o son neutrales).

### Detectores críticos (quitarlos empeora F1)
- **z_score**: quitarlo empeora F1 en -0.1760
- **iqr**: quitarlo empeora F1 en -0.0804
- **local_outlier_factor**: quitarlo empeora F1 en -0.0867
- **acceleration_z**: quitarlo empeora F1 en -0.0221

### Tipo de FN dominante: `gradual`
72 de 72 FN son de tipo `gradual`.
→ Anomalías graduales sin spike claro. Considerar IF con ventanas más cortas o LOF con menor contamination.
