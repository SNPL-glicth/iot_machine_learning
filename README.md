# Arquitectura ZENIN: Motor de Decisión, Ecuación Maestra y Metacognición

> **Módulo Central de Machine Learning e Inferencia de Sistemas Dinámicos**  
> Paquete: `iot_machine_learning` | Versión: `0.1.0` | Python: `>=3.10`

---

## 1. Jerarquía Arquitectónica del Sistema

En la arquitectura de ZENIN, existen dos niveles claramente diferenciados con roles ontológicos distintos:

```
┌────────────────────────────────────────────────────────────────────────┐
│               1. ECUACIÓN MAESTRA (ENTIDAD PRINCIPAL)                  │
│       MasterEquationOrchestrator  &  master_equation.py                │
│                                                                        │
│  • Soberanía en la decisión final: Plan de Ejecución y Veto            │
│  • Resonancia de Ondas Complejas y Acoplamiento de Kuramoto: r(t)      │
│  • Adaptador de Riesgo Estocástico Continuo: I_CVaR                    │
│  • Adaptador Cronométrico Fractal: Lambda(t)                           │
│  • Veto de Momentum Cinético Continuo: V_mom                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Consulta / Integra
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            2. MOTOR COGNITIVO ROSA ROJA (COMPONENTE BASE)              │
│               RosaRojaEngine  &  algorithms/modules/                   │
│                                                                        │
│  • Ingesta y Filtro Anti-Contaminación de Mahalanobis: Módulo 1        │
│  • Generador de Trayectorias y Ritmos en Espacio de Fases: Módulo 2   │
│  • Jurado MoE (Taylor, Kalman, Estadístico) y Varianza: Módulo 3       │
│  • Entrega la Confianza Epistémica Base: Phi_MoE_base                  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Ecuación Maestra (Entidad Soberana y Principal):** Gobierna el orquestador general ([`MasterEquationOrchestrator`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/master_engine/orchestrator.py)). Integra las dimensiones ortogonales de riesgo, tiempo, inercia y física de osciladores para determinar si el sistema actúa o veta la operación.
2. **Motor Rosa Roja (Componente Complementario de Soporte):** Es el motor de trayectorias ([`RosaRojaEngine`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/engines/rosa_roja/algorithms/engine.py)). Alimenta a la Ecuación Maestra con la proyección del espacio de estados y la evaluación epistémica del jurado de expertos Mixture-of-Experts ($\Phi_{\text{MoE,base}}$).

---

## 2. La Ecuación Maestra de ZENIN (Entidad Principal)

### 2.1. Ecuación de Control y Acción de Ejecución
Determina la magnitud y el estado de ejecución en la realidad:

$$\mathcal{O}_{\text{Master}}(t) = \mu_{\text{obj}}(t) \cdot \Theta\Big( \Phi_{\text{certeza}}(t) - \gamma_{\text{exec}} \Big) \cdot \mathcal{V}_{\text{mom}}(t)$$

* **$\mu_{\text{obj}}(t)$ (Magnitud Objetivo):** Magnitud en unidades de estado calculada por consenso ponderado de los expertos:
  $$\mu_{\text{obj}}(t) = \frac{\sum_{i=1}^{N} w_i \cdot y_i(t)}{\sum_{i=1}^{N} w_i}$$
* **$\Theta\big(\Phi_{\text{certeza}}(t) - \gamma_{\text{exec}}\big)$ (Gating de Umbral):** Función escalón de Heaviside que valida si la certeza resonante supera el umbral de ejecución $\gamma_{\text{exec}}$.
* **$\mathcal{V}_{\text{mom}}(t)$ (Filtro de Momentum Cinético):** Validación de inercia instantánea para evitar operar contra la aceleración del entorno:
  $$\mathcal{V}_{\text{mom}}(t) = \text{clip}\left( \frac{k_{\text{flux}} - \delta_{\text{deadband}}}{\delta_{\text{deadband}}}, \, 0.0, \, 1.0 \right)$$
  Donde $k_{\text{flux}} = \overline{(ds/dt)}_{\text{EMA}} \cdot \mu_{\text{obj}}(t)$ y $\delta_{\text{deadband}} = \tau_{\text{mom}} \cdot \max(\sigma_{\text{mom}}, \sigma_{\text{market}})$.

---

### 2.2. Ecuación Maestra Compuesta (Composite Master Equation)
Implementada en [`master_equation.py`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/master_engine/master_equation.py):

$$\Phi_{\text{Master}}(t) = I_{\text{CVaR}} \cdot \Lambda(t) \cdot \Phi_{\text{MoE,base}} \cdot \Big( r(t) \cdot \alpha_{\text{align}} \Big)$$

Términos auditables conforme a la norma ISO 22989:
* **$I_{\text{CVaR}} \in [0.0, 1.0]$:** Factor continuo de solvencia y veto derivado del Conditional Value at Risk ($i_{\text{cvar}}$). Si $I_{\text{CVaR}} \le 0.0$, la certeza colapsa inmediatamente a $0.0$.
* **$\Lambda(t) \in [0.0, 1.0]$:** Sincronía cronométrica y tasa de cambio temporal:
  $$\Lambda(t) = \exp\left( -\min\left(10.0, \, \left| \frac{\big| \partial S / \partial t \big|}{\max\big(|\partial R / \partial t|, \, \epsilon \cdot \sigma_{\partial R / \partial t}\big)} - 1.0 \right| \right) \right)$$
* **$\Phi_{\text{MoE,base}} \in [0.0, 1.0]$:** Confianza epistémica base entregada por el motor complementario Rosa Roja.
* **$r(t) \in [0.0, 1.0]$:** Parámetro de Orden de Kuramoto (coherencia de fase inter-oscilador).
* **$\alpha_{\text{align}} \in [0.0, 1.0]$:** Grado de alineación de fase instantánea (`phase_alignment`).

---

### 2.3. Interferencia Resonante de Ondas Complejas y Parámetro de Kuramoto
Implementada en [`compute_certeza`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/master_engine/master_equation.py#L31-L85):

$$\Phi_{\text{certeza}}(t) = A_1 \cdot A_2 \cdot A_3 \cdot \psi\big(r(t)\big)$$

#### Amplitudes de Onda:
* $A_1 = a_{\text{risk}} = \text{clip}(I_{\text{CVaR}}, 0, 1)$
* $A_2 = \Lambda(t) = \exp(-\min(10, |\text{ratio} - 1|))$
* $A_3 = \Phi_{\text{epist}} = \text{clip}(\Phi_{\text{MoE,base}}, 0, 1)$

#### Espacio de Fases Local:
$$\theta_k = \text{atan2}\left( v_k, \, \Delta x_k \right)$$
* $\theta_1 = 0.0$ (Fase de referencia / amortiguación de riesgo).
* $\theta_2 = \text{atan2}(\Delta v, \, \text{denominador})$, donde $\Delta v = v_s - v_r$.
* $\theta_3 = \text{atan2}\left(v_s, \, \max\big(10^{-4}, |\Phi_{\text{epist}} - 0.5|\big)\right)$.

#### Parámetro de Orden de Kuramoto:
$$z(t) = \frac{1}{N}\sum_{k=1}^N e^{i \theta_k} \implies r(t) = |z(t)| = \sqrt{ \left(\frac{1}{N}\sum_{k=1}^N \cos\theta_k\right)^2 + \left(\frac{1}{N}\sum_{k=1}^N \sin\theta_k\right)^2 }$$

#### Modulación Cuadrática y Protección Estacionaria:
$$\psi(r) = r(t)^2$$
> **Invarianza Estacionaria:** Cuando el sistema opera en régimen laminar o estacionario ($|\Delta v| < 10^{-7}$ o $|v_s| < 10^{-9}$), $r(t) = 1.0$, garantizando el producto marginal exacto $\mathbb{E}[\Phi] = A_1 \cdot A_2 \cdot A_3$ y previniendo desvanecimiento de certeza (*vanishing certainty*).

---

## 3. El Motor Cognitivo Rosa Roja (Componente Complementario)

Rosa Roja provee el colector de trayectorias candidatas y evalúa el consenso interno de los expertos que alimentan la variable $\Phi_{\text{MoE,base}}$ de la Ecuación Maestra.

### 3.1. Gating MoE y Penalización por Desacuerdo (Módulo 3)
Implementado en [`module3_moe_gating.py`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/engines/rosa_roja/algorithms/modules/module3_moe_gating.py):

$$\Phi_{\text{MoE}}(T) = \left[ \prod_{k \in \text{Criticos}} \mathbb{I}\big(\Psi_k(T) \ge \tau_k\big) \right] \cdot \frac{\frac{\sum_{e=1}^{M} w_e \Psi_e(T)}{\sum_{e=1}^{M} w_e}}{1 + \gamma \cdot \text{Var}\big(\{\Psi_e(T)\}\big)}$$

* **Hard-Gating Veto:** Si cualquier experto crítico falla su umbral individual ($\Psi_k < \tau_k$), la indicatriz $\mathbb{I}$ anula la trayectoria.
* **Penalización por Varianza:** El divisor $1 + \gamma \cdot \text{Var}(\{\Psi_e\})$ castiga la confianza si los expertos (Kalman, Taylor, Modelos Estadísticos) emiten señales contradictorias.

---

### 3.2. Ingesta y Filtro Mahalanobis Anti-Contaminación (Módulo 1)
Implementado en [`module1_ingestion.py`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/engines/rosa_roja/algorithms/modules/module1_ingestion.py):

$$\mathcal{D}_t = \mathcal{D}_{t-1} \cup \left\{ (\Delta s_t, \Delta t) \cdot \mathbb{I}\big(d_{\text{Mahalanobis}}(\Delta s_t) \le \tau_{\text{noise}}\big) \right\}$$

$$d_M^2(\Delta s_t) = (\Delta s_t - \mu_n)^T \Sigma_n^{-1} (\Delta s_t - \mu_n)$$

* **Actualización Online $O(1)$ (Welford):**
  $$\mu_n = \mu_{n-1} + \frac{\Delta s_t - \mu_{n-1}}{n}, \quad M_{2,n} = M_{2,n-1} + (\Delta s_t - \mu_{n-1}) \otimes (\Delta s_t - \mu_n)$$
  $$\Sigma_n = \frac{M_{2,n}}{n - 1} + 10^{-6} \cdot I$$

---

## 4. Estructura del Código en `iot_machine_learning`

```
iot_machine_learning/
├── core/
│   └── parameters/
│       └── numerical_constants.py      # Constantes numéricas, tolerancias y umbrales épsilon
├── domain/
│   ├── entities/                       # Entidades de dominio, calibración y evaluación
│   └── ports/                          # Puertos/contratos para jurado, sensores y estado
├── infrastructure/
│   ├── adapters/                       # Adaptadores de telemetría, Weaviate y puente de ejecución
│   └── ml/
│       ├── adapters/                   # Adaptadores de expertos (Kalman, Taylor, Statistical)
│       ├── master_engine/              # [ENTIDAD SOBERANA]
│       │   ├── master_equation.py      # Ecuación Maestra, Interferencia de Ondas y Kuramoto
│       │   ├── orchestrator.py         # MasterEquationOrchestrator (coordinador general)
│       │   ├── plan_builder.py         # Constructor de ExecutionPlan y ActionEnvelope
│       │   └── telemetry.py            # Generación de trazas auditables ISO 22989
│       └── engines/
│           ├── kalman/                 # Motor de predicción por Filtro de Kalman
│           ├── taylor/                 # Motor de predicción por expansión polinomial
│           ├── statistical/            # Motor de predicción por modelos autorregresivos
│           └── rosa_roja/              # [COMPONENTE COMPLEMENTARIO]
│               └── algorithms/
│                   ├── domain/         # Máquina de estados, trayectorias y persistencia
│                   └── modules/        # Ingesta Mahalanobis, Random Walk, Gating MoE
└── tests/
    ├── unit/                           # Tests unitarios matemáticos y de invarianzas
    └── integration/                    # Tests de certificación institucional E2E
```

---

## 5. Ejecución y Verificación de Tests

```bash
# Validar invarianzas de la Ecuación Maestra y Resonancia de Kuramoto
pytest -v iot_machine_learning/tests/unit/market/test_zenin_v22_master_equation.py \
          iot_machine_learning/tests/unit/market/test_kuramoto_resonance.py \
          iot_machine_learning/tests/unit/market/test_master_orchestrator_invariance.py

# Ejecutar la suite unitaria completa (698+ tests)
pytest -v iot_machine_learning/tests/unit/market/
```
