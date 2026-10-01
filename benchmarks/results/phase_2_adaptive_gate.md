# Fase 2: Adaptive Meta-Gating & Conformal Risk Management Report

## 1. Resumen Ejecutivo de la Fase 2

La Fase 2 dota al Mixture of Experts Asimétrico de garantías matemáticas de cota de error, calibración online y compuertas de decisión sensibles al presupuesto computacional:
$$\text{Expertos Asimétricos} \xrightarrow{E_{i, t}} \text{Online Conformal Calibrator (Hedge)} \xrightarrow{w_{i, t}} \text{Ville Martingale Gate (}\tau_t\text{)} \longrightarrow \text{Certified Alarm}$$

### Verificación de Criterios de Aceptación
* **No-Regresión en Ahorro Computacional**: **76.05%** (Target >= 70%) -> **APROBADO**
* **Retardo en Evento 3 (Deriva Sutil)**: **+2 puntos** (Target <= +5 pts) -> **APROBADO**
* **Supresión de Falsas Alarmas en Colas Pesadas (NVDA)**: **48.6% de reducción** (Target >= 40%) -> **APROBADO**
* **Pureza Arquitectónica y Architecture Gate**: **100% verificado sin dependencias en domain/**

---

## 2. Dataset A (Industrial NAB): Comparativa Fase 1 vs Fase 2

| Parámetro | Fase 1 (Heurística SPRT) | Fase 2 (Ville Martingale + Online Hedge) | Estado |
| :--- | :---: | :---: | :---: |
| **Delay Evento 1** | +3 pts | +3 pts | Preservado |
| **Delay Evento 2** | +6 pts | +6 pts | Preservado |
| **Delay Evento 3 (Deriva)** | +2 pts | **+2 pts** | **Blindado (<= +5 pts)** |
| **Delay Evento 4** | +7 pts | +7 pts | Preservado |
| **Ahorro de Cómputo** | 76.05% | **76.05%** | Objetivo >= 70% cumplido |
| **Pesos Finales Expertos (Hedge)** | Estáticos | `{"resting_invariants_10x": 3.1481314341596344e-37, "regime_shift_2x": 0.9999790683809396, "high_frequency_raw": 2.0931619060373456e-05}` | Adaptación contextual |

---

## 3. Dataset B (Financiero NVDA): Control de Riesgo Conforme (Curtosis > 100)

En series financieras con saltos abruptos y colas pesadas, el umbral estático sufre por falsas alarmas persistentes. El proceso de martingala acotado por Ville y la atenuación Hedge eliminan el ruido espurio:

| Métrica | Fase 1 (Umbral Estático) | Fase 2 (Meta-Gate Certificado) | Impacto |
| :--- | :---: | :---: | :---: |
| **Falsas Alarmas en Reposo** | 74 | 38 | **-48.6% de reducción** |
| **Garantía Teórica de Stopping Time** | Ninguna (heurística) | $\mathbb{P}(\exists t : M_t \ge 1/\alpha) \le \alpha$ | **Certificada libre de distribución** |
| **Ponderación Adaptativa Final** | Uniforme | `{"resting_invariants_10x": 0.0023133336173769923, "regime_shift_2x": 0.9750935889863784, "high_frequency_raw": 0.022593077396244554}` | Calibración sin reentrenamiento |
