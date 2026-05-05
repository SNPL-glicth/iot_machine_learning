# Drift Detection y Adaptación del Sistema

**Última actualización:** 2026-05-04
**Archivo fuente:** `infrastructure/ml/cognitive/orchestration/phases/drift_detection_phase.py`

---

## Page-Hinkley

### Parámetros (desde `CognitiveConfig`)

| Parámetro | Default | Qué controla |
|-----------|---------|-------------|
| `ML_DRIFT_DELTA` | 0.005 | Magnitud mínima de cambio a detectar. Menor = más sensible. |
| `ML_DRIFT_LAMBDA` | 50.0 | Umbral acumulador. Menor = más estricto (detecta antes, más falsos positivos). |
| `ML_DRIFT_ALPHA` | 0.9999 | Factor de olvido. 1.0 = sin olvido (detecta drift histórico); 0.9999 = olvido gradual. |
| `ML_DRIFT_COOLDOWN_SECONDS` | 300.0 | Mínimo segundos entre resets por serie. Evita oscilación. |

### Cómo detecta

Page-Hinkley acumula desviaciones de la media de referencia:

```
PH_t = max(0, PH_{t-1} + (x_t - μ_ref) - δ)
```

Si `PH_t > λ`, drift detectado.

**Señal usada:** `noise_ratio + stability` como proxy de cambio de concepto. No usa el valor crudo del sensor para evitar detectar drift por cambios de setpoint legítimos.

### Cuándo dispara reset

1. Drift confirmado (no en cooldown).
2. `BayesianWeightTracker.reset_regime(regime, series_id, drift_severity)`.
3. Pesos bayesianos del régimen afectado vuelven a priors uniformes.
4. El sistema re-aprende desde cero para ese régimen.

---

## ADWIN

### Parámetros

| Parámetro | Default | Qué controla |
|-----------|---------|-------------|
| `ML_ENABLE_ADWIN` | `false` | Master switch. Cuando `true`, reemplaza Page-Hinkley. |
| `ML_ADWIN_DELTA` | 0.002 | Confianza estadística. Menor = más estricto. |
| `ML_ADWIN_MAX_WINDOW` | 1000 | Tamaño máximo de ventana. Mayor = más memoria, más precisión. |

### Diferencia con Page-Hinkley

- **Page-Hinkley:** Paramétrico. Requiere definir δ y λ. Funciona bien con señales de varianza conocida.
- **ADWIN:** No paramétrico. Divide la ventana en subventanas y compara medias con test estadístico. Mejor para señales de varianza desconocida o cambiante.

**Recomendación:** Usar Page-Hinkley por defecto. Activar ADWIN solo si la señal tiene varianza altamente heterogénea (ej. proceso químico con múltiples fases de operación).

---

## Qué Pasa Cuando Se Detecta Drift

### Secuencia de acciones

```mermaid
sequenceDiagram
    participant DD as DriftDetectionPhase
    participant BT as BayesianWeightTracker
    participant AP as AuditPort
    participant CE as ComplianceExporter

    DD->>DD: PageHinkley.update(noise_ratio + stability)
    DD->>DD: cooldown_active? → no
    DD->>BT: reset_regime(regime, series_id, severity)
    BT->>BT: del accuracy[regime]; del priors[regime]
    DD->>AP: log_config_change("regime_reset", old, new, "drift_detector")
    DD->>CE: build_record(series_id, result_with_drift_metadata)
```

### Indicador ISO 13374

Se emite condición indicador `DRIFT_MAGNITUDE` con valor `drift_score` en el contexto. Esto cumple el bloque funcional "Condition Monitoring" de ISO 13374.

---

## Limitación Documentada

> **El detector actual usa `noise_ratio + stability` como proxy de drift. No detecta concept drift de P(y|X) cuando los valores son estables pero incorrectos (ej: sensor descalibrado que sigue reportando 50.0°C sin variación cuando debería ser 55.0°C).**
>
> **Para ese caso, activar `ML_ENABLE_ERROR_DRIFT_DETECTOR=True` (a verificar — flag no confirmado en `cognitive_config.py`).**
>
> **Impacto:** Una planta con sensores descalibrados pero estables no verá drift hasta que la señal empiece a variar. Recomendación: calibración periódica + drift basado en error (cuando `abs(predicted - actual)` crece sistemáticamente).
