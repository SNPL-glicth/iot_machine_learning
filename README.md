# ZENIN — Unified Machine Learning Engine

Motor de Machine Learning unificado para inferencia en series temporales, detección de anomalías y control estocástico. Emplea una arquitectura **Mixture of Experts (MoE)** gobernada por **Rosa Roja** como motor cabeza y árbitro de consenso.

---

## 🏛️ Arquitectura

```
                        ┌─────────────────────────────────────────┐
                        │      ROSA ROJA (MoE Master Head)        │
                        │  • M1: Filtro Mahalanobis (Outliers)    │
                        │  • M2: Generador de Trayectorias Ritmo  │
                        │  • M3: Gating Multiplicativo & Veto     │
                        └────────────────────┬────────────────────┘
                                             │ Consenso & Veto
                 ┌───────────────────────────┼───────────────────────────┐
                 ▼                           ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
        │     KALMAN      │         │     TAYLOR      │         │   STATISTICAL   │
        │ 2D CV Kinematic │         │ High-Order Poly │         │ Exp Smoothing & │
        │ Noise Filtering │         │ Fast Local Extr │         │ Drift Analysis  │
        └─────────────────┘         └─────────────────┘         └─────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     PredictionResult      │
                               │      / ExecutionPlan      │
                               └─────────────┬─────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
       ┌──────────────────────────────┐              ┌──────────────────────────────┐
       │     SERIES TEMPORALES / IoT  │              │    TELEMETRÍA & MONITOREO   │
       │ • Ingesta de sensores        │              │ • Streaming WS (Port 8765)   │
       │ • Detección de drift         │              │ • Zenin-TUI Dashboard        │
       └──────────────┬───────────────┘              └──────────────────────────────┘
                      │
                      ▼
       ┌────────────────────────────────────────────────────────────┐
       │             BUCLE METACONGNITIVO & META-APRENDIZAJE         │
       │  • PostMortemEvaluator: Auditoría y Credit Assignment      │
       │  • FailureTaxonomy: Diagnóstico Causal de Fallos           │
       │  • MetacognitiveTracker: Cuantificación de Auto-Ceguera Ω_t│
       │  • RegimePlasticityManager: Pesos dinámicos y Breakers     │
       └────────────────────────────────────────────────────────────┘
```

---

## ⚡ Ecuación Maestra y Metacognición

Rosa Roja modula la confianza y la toma de decisiones mediante inferencia activa bayesiana:

$$\Phi_{\text{MoE}} = \Phi_{\text{base}} \cdot \Big(1 - \lambda_t \cdot (1 - \Phi_{\text{ritmo}})\Big)$$

* **$\Phi_{\text{base}}$**: Consenso ponderado del jurado interno (Kalman, Taylor, Estadístico).
* **$\Phi_{\text{ritmo}}$**: Coherencia dinámica calculada sobre trayectorias temporales (11 a 15 pasos).
* **$\lambda_t \in [0, 1]$**: Factor bayesiano de incertidumbre y exploración. Modulado dinámicamente por el **Índice de Meta-Competencia ($\Omega_t$)**:
  $$\lambda_t \leftarrow \max(\lambda_t, 1 - \Omega_t)$$
  Si el sistema detecta **ceguera sistémica** (todos los expertos fallando correlacionadamente en el régimen actual), $\Omega_t \to 0$ forzando $\lambda_t \to 1$ y abstención inmediata (**`HOLD`**).
* **Veto Estocástico**: Si un experto crítico discrepa o Mahalanobis detecta cambio de régimen, Rosa Roja suspende la acción.
* **Plasticidad por Régimen**: Los pesos de los expertos se ajustan continuamente según su pérdida exponencial en cada contexto: $w_i \propto \exp(-\eta \cdot \text{Loss}_i)$.

---

## 📁 Estructura del Proyecto

* **`domain/`**: Entidades puras y modelos de dominio (series temporales, ventanas, anomalías, cognición). Aislado de infraestructura.
  * **`domain/entities/cognitive/`**: Taxonomía de fallos (`failure_taxonomy.py`), evaluador post-mortem (`post_mortem_evaluator.py`) y tracker de meta-competencia (`metacognitive_tracker.py`).
* **`infrastructure/ml/engines/`**:
  * **`rosa_roja/`**: Motor cabeza MoE (`RosaRojaMoEEngine`) y suite algorítmica (`algorithms/`).
  * **`kalman/`**, **`taylor/`**, **`statistical/`**: Motores matemáticos especializados del jurado.
* **`infrastructure/ml/cognitive/`**:
  * **`plasticity/`**: Gestor de plasticidad y circuit-breakers por régimen (`regime_plasticity_manager.py`).
  * **`metacognitive_coordinator.py`**: Orquestador central del bucle de retroalimentación metacognitivo.
* **`infrastructure/ml/moe/`**: Despacho y registro de Mixture of Experts (`RosaRojaExpert`, adaptadores).
* **`infrastructure/adapters/`**: Adaptadores de entrada y salida (sensores, Binance Futures WebSocket, telemetría).
* **`zenin-tui/`**: Interfaz de terminal interactiva (Ink/TypeScript) para monitoreo en tiempo real vía WebSocket.

---

## 🚀 Inicio Rápido

### Instalación
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ejecución de Tests
```bash
# Tests unitarios del núcleo MoE y Rosa Roja
PYTHONPATH=. pytest tests/unit/rosa_roja/ tests/unit/moe/ tests/integration/test_moe_multi_expert_consensus.py

# Tests del sistema metacognitivo y meta-aprendizaje
PYTHONPATH=. pytest tests/unit/cognitive/test_metacognitive_system.py

# Tests de adaptadores de series temporales
PYTHONPATH=. pytest tests/test_iot_adapter.py
```

---

## 🛡️ Reglas Arquitectónicas

1. **Aislamiento de Dominio**: `domain/` no puede importar nada de `infrastructure/`.
2. **Modularidad**: Límite estricto de **< 180 líneas de código** por archivo.
3. **Persistencia e Independencia**: Cada dominio mantiene su almacenamiento y esquemas desacoplados.

