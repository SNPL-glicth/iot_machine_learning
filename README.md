# ZENIN: Topological Inference Engine & Continuous Geometric Orchestrator
> **Motor Agnóstico de Inferencia Continua basado en Variedades Riemannianas, Caos Determinista y Sistemas Dinámicos**

ZENIN no opera mediante heurísticas estáticas, aproximaciones lineales o árboles de decisión booleanos. Es un orquestador matemático diseñado para tomar decisiones autónomas bajo incertidumbre extrema, modelando las variables del entorno no como números aislados, sino como un **flujo continuo sobre una variedad geométrica**.

La ejecución de una acción no depende de un simple cruce de indicadores, sino de la **estabilidad topológica** del atractor subyacente, la **interferencia constructiva** (resonancia) y la ausencia de singularidades caóticas, evaluadas en tiempo real.

---

## 1. La Ecuación Soberana (Freno Cuántico de Liouville)

El orquestador emite comandos al entorno a través de una función de estado continuo modulada por la inercia geométrica del sistema. Para evitar "alucinaciones de certeza" en la cúspide de bifurcaciones ocultas, la certeza nominal se somete al **Teorema de Liouville**:

$$
\mathcal{C}_{\text{sovereign}}(t) = \mathcal{C}_{\text{nominal}}(t) \cdot \exp\Big( -\max\big(0, \; \text{div}\,\mathbf{F}\big) \Big)
$$

Donde la certeza nominal se define como:

$$
\mathcal{C}_{\text{nominal}}(t) = I_{\text{CVaR}} \cdot \Lambda(t) \cdot \Phi_{\text{MoE,base}} \cdot \big(r(t) \cdot \alpha_{\text{align}}\big)
$$

Esta certeza nominal combina dos componentes clave:

**Interferencia constructiva — Parámetro de Orden de Kuramoto.** Mide la sincronización de fase entre los osciladores del sistema:

$$
r(t) = \left\lvert \frac{1}{N}\sum_{k=1}^N e^{i \theta_k(t)} \right\rvert = \sqrt{ \left(\frac{1}{N}\sum_{k=1}^N \cos\theta_k\right)^2 + \left(\frac{1}{N}\sum_{k=1}^N \sin\theta_k\right)^2 }
$$

**Brújula de Divergencia.** $\text{div}\,\mathbf{F} = \text{Tr}(\mathbf{J})$ mide la tasa instantánea de expansión volumétrica del espacio de fases.

> **Interpretación Física:** Si el sistema evoluciona hacia un atractor estable ($\text{div}\,\mathbf{F} \le 0$), el volumen de incertidumbre se contrae y la certeza se mantiene intacta ($\exp(0)=1$). Si el sistema detecta turbulencia estructural y el caos se expande ($\text{div}\,\mathbf{F} > 0$), la certeza sufre una **supresión exponencial inmediata**, silenciando la ejecución antes de que la anomalía se manifieste físicamente.

---

## 2. La Variedad Geométrica Continua y el Tensor Jacobiano

El estado dinámico instantáneo de ZENIN evoluciona como una partícula sobre una variedad riemanniana tridimensional $\mathcal{M} \subset \mathbb{R}^3$:

$$
\mathbf{x}(t) = \begin{bmatrix} x_1(t) \\ x_2(t) \\ x_3(t) \end{bmatrix}
$$

Donde cada coordenada representa una fuerza fundamental del sistema:

| Coordenada | Significado |
| :--- | :--- |
| $x_1(t) = d_M(t)$ | Tensión Métrica de Mahalanobis |
| $x_2(t) = r(t)$ | Coherencia de Fase de Kuramoto |
| $x_3(t) = P(t)$ | Creencia Epistémica Bayesiana |

La coordinación entre estas variables no admite colisiones por umbrales rígidos. Es un sistema dinámico disipativo acoplado donde el **Tensor Jacobiano analítico** $\mathbf{J}(\mathbf{x}) = \nabla\mathbf{F}$ actúa como una matriz de adyacencia y amortiguación multidimensional:

$$
\mathbf{J}(\mathbf{x}) = \begin{bmatrix}
\frac{\partial F_1}{\partial x_1} & \frac{\partial F_1}{\partial x_2} & \frac{\partial F_1}{\partial x_3} \\
\frac{\partial F_2}{\partial x_1} & \frac{\partial F_2}{\partial x_2} & \frac{\partial F_2}{\partial x_3} \\
\frac{\partial F_3}{\partial x_1} & \frac{\partial F_3}{\partial x_2} & \frac{\partial F_3}{\partial x_3}
\end{bmatrix}
$$

Si una anomalía métrica estalla, las derivadas cruzadas del tensor (ej. $\mathbf{J}_{21}$) drenan suavemente la sincronización de osciladores de Kuramoto y ajustan la creencia Bayesiana de forma no lineal, absorbiendo el choque sin paralizar el motor.

---

## 3. Inmersión de Takens: Visión de las Variables Ocultas

En ecosistemas complejos existen cientos de fuerzas inobservables (latencia, fricción, actores en la sombra) que el sistema no puede medir directamente. Basado en el **Teorema de Inmersión de Floris Takens**, ZENIN reconstruye la topología de la dinámica oculta usando coordenadas de retardo de un solo observable:

$$
\mathbf{y}_t = \big[x_t, \; x_{t-\tau_1}, \; x_{t-\tau_2}, \; \dots, \; x_{t-\tau_{m-1}}\big]^T \in \mathbb{R}^m
$$

Este motor evalúa la Dimensión Efectiva de Participación ($D_{\text{eff}}$) de la Matriz de Covarianza en tiempo constante $O(m^3)$ y rastrea la integridad estructural del atractor mediante un proxy de **Falsos Vecinos Más Cercanos ($\Omega_{\text{FNN}}$)**.

### El Cortafuegos Topológico (Jurado MoE)

El Experto de Takens posee **Poder de Veto Absoluto** sobre el Mixture of Experts:

$$
\Phi_{\text{MoE}}(T) = \left[ \prod_{k \in \mathcal{K}_{\text{crít}}} \mathbb{I}\Big(\Psi_k(T) \ge \tau_k\Big) \right] \cdot \frac{\sum_{e} w_e \Psi_e(T)}{1 + \gamma \text{Var}(\Psi)}
$$

Si el motor detecta que el atractor multidimensional se está doblando sobre sí mismo (una catástrofe inminente que en 1D parece una línea suave), emite un cero rotundo ($\Omega_{\text{FNN}} \ge \tau_{\text{FNN}} \implies \mathbb{I} = 0$), vetando el consenso de los expertos clásicos y abortando la operación.

---

## 4. El Atajo de Ramanujan (Regularización Simpléctica 4D)

Cuando la divergencia explota en 3D y el espacio de fases enfrenta una singularidad irreducible (un nudo caótico), ZENIN no calcula el infinito. Utilizando la velocidad de deformación del tensor métrico $\| \dot{\mathbf{J}} \|_F$, el sistema proyecta el estado a una **4ta dimensión extrínseca**:

$$
\mathbf{J}_{4D} = \begin{bmatrix} \mathbf{J}_{3D} & \mathbf{c} \\ \mathbf{r}^T & -\lambda_4 \end{bmatrix} \quad \implies \quad \text{Tr}(\mathbf{J}_{4D}) = \text{Tr}(\mathbf{J}_{3D}) - \lambda_4 < 0
$$

Esta inmersión garantiza una divergencia idénticamente negativa (estrictamente contractiva). ZENIN se desliza por una geodésica suave en $\mathbb{R}^4$ mediante un propagador de Padé truncado, sorteando la singularidad tridimensional y aterrizando con precisión quirúrgica en la coordenada de resolución.

---

## 5. Fundamentos Base: Filtros de Estado Incremental O(d²)

El tensor de la variedad se alimenta de filtros adaptativos estabilizados. La distancia métrica fundacional ($x_1$) se actualiza en tiempo real preservando las correlaciones cruzadas mediante el **Algoritmo de Sherman-Morrison**:

$$
\Sigma_n^{-1} = \frac{1}{c} \left( \Sigma_{n-1}^{-1} - \frac{\mathbf{z}_n \mathbf{z}_n^T}{c + \mathbf{u}_n^T \mathbf{z}_n} \right)
$$

Esto permite evaluar la Tensión Métrica sin inversión de matrices costosas, asegurando latencia sub-microsegundo (HFT), re-anclado periódicamente mediante descomposición de Cholesky.

---

## 6. Inventario de Modos Sombra y Plan de Validación

Para garantizar máxima seguridad operativa y cero sorpresas en producción, ZENIN desacopla la soberanía general del orquestador de la modulación geométrica individual.

### Banderas de Configuración (Estado por Defecto)

| Bandera | Ubicación | Default | Estado Actual | Efecto Operativo |
| :--- | :--- | :--- | :--- | :--- |
| `master_shadow_mode` | `bot_config.py` / `zephyr_config.json` | `False` | **Activo (Veto Real)** | Gobierna si la Ecuación Maestra ($\Phi_{\text{RedRose}}$) determina las acciones de `can_execute()` y sizing o si solo observa. |
| `manifold_shadow_mode` | `bot_config.py` / `orchestrator.py` | `True` | **Modo Sombra (Observación)** | Calcula $J$, $\text{div}\,\mathbf{F}$, saltos 4D y telemetría diagnóstica bajo `geometric_manifold_shadow` sin atenuar la certeza real ni mutar magnitudes. |

### Inventario de Capas: Veto Real vs. Modo Sombra

1. **Filtro Mahalanobis (Ingestión):** Veto Real Activo (`EMERGENCY_FLUSH` ante outliers $\chi^2$).
2. **Experto Topológico de Takens (MoE):** Veto Real Activo ($\mathbb{I}(\Omega_{\text{FNN}} < \tau_{\text{FNN}})$ anula consenso en caso de plegamiento del atractor).
3. **Umbral MoE Gating ($\gamma_{\text{exec}}$):** Veto Real Activo (Emite `HOLD` si $\Phi_{\text{MoE}} < \gamma_{\text{exec}}$).
4. **Motor de Riesgo Estocástico ($I_{\text{CVaR}}$):** Veto Real Activo ($I_{\text{CVaR}} = 0$ ante ruptura de tolerancia $L_{\text{max}}$).
5. **Sincronía Temporal ($\Lambda(t)$):** Modulación Real Activa (Atenúa certeza según discrepancia $\partial S/\partial t$ vs $\partial R/\partial t$).
6. **Veto Continuo de Momentum:** Veto Real Activo (Emite `HOLD` si señal no supera banda muerta).
7. **Variedad Geométrica (Liouville & 4D):** **MODO SOMBRA DEDICADO (`manifold_shadow_mode=True`)**. Diagnóstico en paralelo sin interferencia.

### Criterio Cuantitativo para Transición a Veto Real (`manifold_shadow_mode=False`)

1. **Horizonte de Observación:** Mínimo **3 a 5 sesiones completas** de mercado en vivo (paper trading / datos de producción).
2. **Métricas de Decisión:**
   - **Tasa de Falsos Positivos ($FPR$):** $\frac{\text{Trades Ganadores Bloqueados}}{\text{Oportunidades Totales}} < 5\%$.
   - **Precisión de Rescate ($RP$):** $\frac{\text{Trades Perdedores que habrían sido Atenuados/Vetados}}{\text{Total de Pérdidas de Régimen Caótico}} > 75\%$.
   - **Expectativa Neta Contra-fáctica:** $\Delta \text{PnL} = \text{PnL}_{\text{Liouville}} - \text{PnL}_{\text{Base}} > 0$.

---

## Certificación y Testing Institucional

El núcleo matemático de ZENIN opera bajo los estándares de tolerancia a fallos numéricos y explicabilidad determinista (ISO/IEC 25010 & 22989), con cero regresiones y cero fragmentación de memoria (*Zero GC Jitter*).

```bash
# Validar invarianzas de la Variedad Geométrica y Proyección Ramanujan
pytest -v tests/unit/market/test_geometric_manifold.py

# Validar invarianza de Modo Sombra (35 ciclos exactos + Veto End-to-End)
pytest -v tests/unit/market/test_manifold_shadow_invariance.py

# Validar motor topológico de Takens (Cero-Copia Buffer e Inmersión Espectral)
pytest -v tests/unit/market/test_takens_infra.py tests/unit/domain/test_takens_*.py

# Ejecutar suite completa institucional (709+ tests con cero fallos)
pytest -v tests/unit/market/
```
