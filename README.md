# Arquitectura ZENIN: Motor de Decisión, Resonancia de Ondas y Metacognición

> **Módulo Central de Machine Learning e Inferencia de Sistemas Dinámicos**  
> Paquete: `iot_machine_learning` | Versión: `0.1.0` | Python: `>=3.10`

---

## 1. Visión y Fundamentos del Sistema

**ZENIN** implementa un paradigma de *metacognición computacional* y física de sistemas dinámicos para la toma de decisiones en entornos estocásticos con alta presencia de ruido y no-estacionariedad.

A diferencia de los modelos de inferencia convencionales que fusionan magnitud y probabilidad en una única salida determinista, ZENIN desacopla ortogonalmente tres dimensiones fundamentales:
1. **Magnitud Objetivo ($\mu$):** Consenso ponderado sobre la acción óptima en unidades de estado.
2. **Certeza Resonante ($\Phi$):** Grado de coherencia analítica y de fase entre los subsistemas de inferencia, acotado en $[0.0, 1.0]$.
3. **Filtro de Momentum Cinético ($\mathcal{V}_{\text{mom}}$):** Validación de inercia y dinámica temporal instantánea mediante deadband adaptativo.

---

## 2. Marco Matemático y Ecuaciones Activas

### 2.1. Ecuación Maestra Compuesta (Composite Master Equation)
Implementada en [`master_equation.py`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/master_engine/master_equation.py):

$$\Phi_{\text{RedRose}}(t) = I_{\text{CVaR}} \cdot \Lambda(t) \cdot \Phi_{\text{MoE\_base}} \cdot \Big( r(t) \cdot \alpha_{\text{align}} \Big)$$

Donde cada componente auditable cumple con la norma ISO 22989:
* **$I_{\text{CVaR}} \in [0.0, 1.0]$:** Factor de solvencia y veto continuo derivado del Conditional Value at Risk ($i_{\text{cvar}}$). Si $I_{\text{CVaR}} \le 0.0$, la certeza colapsa a $0.0$.
* **$\Lambda(t) \in [0.0, 1.0]$:** Coherencia y sincronía cronométrica de tiempo:
  $$\Lambda(t) = \exp\left( -\min\left(10.0, \, \left| \frac{\big| \partial S / \partial t \big|}{\max\big(|\partial R / \partial t|, \, \epsilon \cdot \sigma_{\partial R / \partial t}\big)} - 1.0 \right| \right) \right)$$
* **$\Phi_{\text{MoE\_base}} \in [0.0, 1.0]$:** Confianza epistémica base entregada por el jurado de expertos Mixture-of-Experts.
* **$r(t) \in [0.0, 1.0]$:** Parámetro de Orden de Kuramoto para sincronización de fase entre osciladores del sistema.
* **$\alpha_{\text{align}} \in [0.0, 1.0]$:** Grado de alineación de fase inter-componente (`phase_alignment`).

---

### 2.2. Interferencia Resonante de Ondas Complejas y Parámetro de Kuramoto
Implementada en [`compute_certeza`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/master_engine/master_equation.py#L31-L85):

$$\Phi_{\text{certeza}}(t) = A_1 \cdot A_2 \cdot A_3 \cdot \psi\big(r(t)\big)$$

#### 1. Amplitudes de Onda:
* $A_1 = a_{\text{risk}} = \text{clip}(I_{\text{CVaR}}, 0, 1)$
* $A_2 = \Lambda(t) = \exp(-\min(10, |\text{ratio} - 1|))$
* $A_3 = \Phi_{\text{epist}} = \text{clip}(\text{certeza\_epistemica}, 0, 1)$

#### 2. Mapeo a Espacio de Fases Local:
$$\theta_k = \text{atan2}\left( \text{velocidad}_k, \, \text{desplazamiento}_k \right)$$
* $\theta_1 = 0.0$ (Fase de referencia / amortiguación de riesgo).
* $\theta_2 = \text{atan2}(\Delta v, \, \text{denominador})$, donde $\Delta v = v_s - v_r$.
* $\theta_3 = \text{atan2}\left(v_s, \, \max\big(10^{-4}, |\Phi_{\text{epist}} - 0.5|\big)\right)$.

#### 3. Parámetro de Orden de Kuramoto:
$$z(t) = \frac{1}{N}\sum_{k=1}^N e^{i \theta_k} \implies r(t) = |z(t)| = \sqrt{ \left(\frac{1}{N}\sum_{k=1}^N \cos\theta_k\right)^2 + \left(\frac{1}{N}\sum_{k=1}^N \sin\theta_k\right)^2 }$$

#### 4. Modulación Resonante y Protección Estacionaria:
$$\psi(r) = r(t)^2$$
> **Invarianza Estacionaria:** Si la dinámica no presenta fluctuación diferencial ($|\Delta v| < 10^{-7}$ o $|v_s| < 10^{-9}$), se fija $r(t) = 1.0$, preservando el producto marginal clásico $\mathbb{E}[\Phi] = A_1 \cdot A_2 \cdot A_3$ sin desvanecimiento de certeza (*vanishing certainty*).

---

### 2.3. Consenso de Magnitud Objetivo
Implementada en [`compute_magnitud_objetivo`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/master_engine/master_equation.py#L87-L109):

$$\mu_{\text{obj}}(t) = \frac{\sum_{i=1}^{N} w_i \cdot y_i(t)}{\sum_{i=1}^{N} w_i}$$

Calcula el promedio ponderado de las proyecciones $\{y_i\}$ de los expertos (ej. Taylor, Kalman, Modelos Estadísticos) según sus pesos dinámicos $\{w_i\}$.

---

### 2.4. Veto de Momentum Continuo (Smooth Deadband Filter)
Implementada en [`compute_momentum_veto`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/master_engine/master_equation.py#L111-L133):

$$\mathcal{V}_{\text{mom}}(t) = \text{clip}\left( \frac{k_{\text{flux}} - \delta_{\text{deadband}}}{\delta_{\text{deadband}}}, \, 0.0, \, 1.0 \right)$$

Donde:
* **Flujo cinético:** $k_{\text{flux}} = \overline{\left(\frac{ds}{dt}\right)}_{\text{EMA}} \cdot \mu_{\text{obj}}(t)$
* **Banda muerta adaptativa:** $\delta_{\text{deadband}} = \tau_{\text{mom}} \cdot \max(\sigma_{\text{mom}}, \sigma_{\text{market}})$
* Si $k_{\text{flux}} \le \delta_{\text{deadband}}$, $\mathcal{V}_{\text{mom}} = 0.0$ (bloqueo por ruido o contra-inercia).

---

### 2.5. Gating MoE Multiplicativo y Penalización por Desacuerdo
Implementada en [`module3_moe_gating.py`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/engines/rosa_roja/algorithms/modules/module3_moe_gating.py):

$$\Phi_{\text{MoE}}(T) = \left[ \prod_{k \in \text{Críticos}} \mathbb{I}\big(\Psi_k(T) \ge \tau_k\big) \right] \cdot \frac{\frac{\sum_{e=1}^{M} w_e \Psi_e(T)}{\sum_{e=1}^{M} w_e}}{1 + \gamma \cdot \text{Var}\big(\{\Psi_e(T)\}\big)}$$

* **Hard-Gating:** Si un único experto calificado como *crítico* evalúa la trayectoria $T$ por debajo de su umbral $\tau_k$, la indicatriz $\mathbb{I}$ se anula y la propuesta queda vetada de inmediato.
* **Penalización por Varianza $\gamma$:** Amortigua la confianza si los expertos presentan opiniones divergentes sobre la trayectoria candidata.

---

### 2.6. Ingesta y Filtro Mahalanobis Anti-Contaminación
Implementado en [`module1_ingestion.py`](file:///home/nicolas/Documentos/Proyectos/ST/iot_machine_learning/infrastructure/ml/engines/rosa_roja/algorithms/modules/module1_ingestion.py):

$$\mathcal{D}_t = \mathcal{D}_{t-1} \cup \left\{ (\Delta s_t, \Delta t) \cdot \mathbb{I}\big(d_{\text{Mahalanobis}}(\Delta s_t) \le \tau_{\text{noise}}\big) \right\}$$

$$d_M^2(\Delta s_t) = (\Delta s_t - \mu_n)^T \Sigma_n^{-1} (\Delta s_t - \mu_n)$$

* **Actualización en Tiempo Real $O(1)$ (Welford):**
  $$\mu_n = \mu_{n-1} + \frac{\Delta s_t - \mu_{n-1}}{n}, \quad M_{2,n} = M_{2,n-1} + (\Delta s_t - \mu_{n-1}) \otimes (\Delta s_t - \mu_n)$$
  $$\Sigma_n = \frac{M_{2,n}}{n - 1} + 10^{-6} \cdot I$$

---

## 3. Estructura del Código en `iot_machine_learning`

```
iot_machine_learning/
├── core/
│   └── parameters/
│       └── numerical_constants.py      # Umbrales épsilon, tolerancias y constantes numéricas
├── domain/
│   ├── entities/                       # Entidades de mercado, calibración y evaluación
│   └── ports/                          # Interfaces/contratos de expertos, sensores y stores
├── infrastructure/
│   ├── adapters/                       # Adaptadores de telemetría, Weaviate y puente Zephyr
│   └── ml/
│       ├── adapters/                   # Adaptadores de jurado experto (Kalman, Taylor, Statistical)
│       ├── engines/
│       │   ├── kalman/                 # Motor de predicción por Filtro de Kalman
│       │   ├── taylor/                 # Motor de predicción por expansión polinomial de Taylor
│       │   ├── statistical/            # Motor de predicción por modelos autorregresivos/estadísticos
│       │   └── rosa_roja/              # Motor cognitivo metacognitivo Rosa Roja
│       │       └── algorithms/
│       │           ├── domain/         # Máquina de estados, trayectorias y persistencia
│       │           └── modules/        # Ingesta Mahalanobis, Random Walk, Gating MoE
│       └── master_engine/
│           ├── master_equation.py      # Ecuación Maestra, Interferencia de Ondas y Kuramoto
│           ├── orchestrator.py         # MasterEquationOrchestrator (coordinador general)
│           ├── plan_builder.py         # Constructor de ExecutionPlan y ActionEnvelope
│           └── telemetry.py            # Generación de trazas de decisión ISO 22989
└── tests/
    ├── unit/                           # Tests unitarios matemáticos y de invarianzas
    └── integration/                    # Tests de certificación E2E institucional
```

---

## 4. Ejecución de Tests y Verificación

Para validar la suite matemática completa y las invarianzas de la Ecuación Maestra:

```bash
# Tests unitarios del motor maestro e interferencia de Kuramoto
pytest -v iot_machine_learning/tests/unit/market/test_zenin_v22_master_equation.py \
          iot_machine_learning/tests/unit/market/test_kuramoto_resonance.py \
          iot_machine_learning/tests/unit/market/test_master_orchestrator_invariance.py

# Suite unitaria completa de mercado (698+ tests)
pytest -v iot_machine_learning/tests/unit/market/
```
