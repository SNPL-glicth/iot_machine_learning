# Fase 1: Asymmetric Mixture of Experts (MoE) Benchmark Report

## 1. Resumen Ejecutivo de la Fase 1

La Fase 1 implementa la arquitectura de inferencia asimétrica basada en contratos puros del Dominio y despacho por afinidad de escala:
$$\text{Stream} \longrightarrow \text{Representation Policy} \longrightarrow \text{Asymmetric Dispatcher} \longrightarrow \text{Evidence Accumulator} \longrightarrow \text{Integrated Evidence}$$

### Verificación de Invariantes Científicos
* **Ahorro Computacional**: **76.05%** (Target $\ge 50%$) -> **APROBADO**
* **Retardo en Evento 3 (Deriva Sutil)**: **+2 puntos** (Línea base RAW $\le +139$, naive 10X $= +273$) -> **APROBADO**
* **Desacoplamiento Estricto de Puertos**: **100% verificado por Architecture Gate**

---

## 2. Comparativa Cuantitativa: Simétrico vs Asimétrico

| Métrica | Simétrico RAW (Todos los expertos) | Asimétrico MoE (Fase 1) | Impacto / Delta |
| :--- | :---: | :---: | :---: |
| **Puntos Evaluados** | 21,695 | 10,649 | **50.9% menos puntos** |
| **Costo Computacional Normalizado** | 27,118.8 | 649.4 | **76.05% de ahorro** |
| **Tiempo de Inferencia (ms)** | 2462.5 ms | 299.0 ms | **8.2x más rápido** |
| **Tiempo en 10X (Reposo/Envolvente)** | 0.0% | 23.2% | Operación eficiente |
| **Tiempo en 2X (Deriva/Régimen)** | 0.0% | 60.0% | Cobertura de deriva |
| **Tiempo en RAW (Alta Frecuencia)** | 100.0% | 16.8% | Activación selectiva |

---

## 3. Localización Temporal de Anomalías (Event Detection Delays)

| Evento Canónico | Descripción | Delay Simétrico RAW | Delay Asimétrico MoE | Estatus de Preservación |
| :---: | :---: | :---: | :---: | :---: |
| **Evento 1** | Choque térmico inicial | +0 pts | +3 pts | Preservado |
| **Evento 2** | Anomalía oscilatoria | +0 pts | +6 pts | Preservado |
| **Evento 3** | **Deriva lenta persistente** | **+0 pts** | **+2 pts** | **Límite $+139$ Blindado** |
| **Evento 4** | Falla catastrófica final | +0 pts | +7 pts | Preservado |

---

## 4. Desglose de Telemetría del Despachador

```json
{
  "dispatches_by_level": {
    "10X": 504,
    "2X": 1301,
    "RAW": 364
  },
  "evaluations_by_expert": {
    "resting_invariants_10x": 504,
    "regime_shift_2x": 1301,
    "high_frequency_raw": 364
  },
  "total_cost_expended": 649.4,
  "total_cost_hypothetical_full": 2711.25,
  "compute_savings_ratio": 0.7605
}
```
