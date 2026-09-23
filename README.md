# ZENIN: Stochastic Decision Engine & Wave Resonance Orchestrator
> **Motor Agnóstico de Inferencia Continua basado en Caos Determinista y Sistemas Dinámicos**

ZENIN no opera mediante heurísticas estáticas o árboles de decisión booleanos. Es un orquestador matemático diseñado para tomar decisiones bajo incertidumbre extrema modelando las variables del entorno como **ondas en un espacio de fases**. 

La ejecución de una acción no depende de un cruce de indicadores, sino de la **interferencia constructiva** (resonancia) entre el riesgo, la inercia temporal y la confianza epistémica.

---

## 1. La Ecuación de Acción (Soberanía de Ejecución)
El orquestador emite comandos al mundo real a través de una función de estado continuo modulada por la inercia del sistema:

$$
\mathcal{O}_{\text{Master}}(t) = \mu_{\text{obj}}(t) \cdot \Theta\Big( \Phi_{\text{certeza}}(t) - \gamma_{\text{exec}} \Big) \cdot \mathcal{V}_{\text{mom}}(t)
$$

* $\mu_{\text{obj}}(t)$: **Magnitud del vector objetivo calculada por consenso bayesiano.**
* $\Theta(\cdot)$: **Función de activación (Gating) que evalúa si la certeza resonante supera el umbral crítico** $\gamma_{\text{exec}}$.
* $\mathcal{V}_{\text{mom}}(t)$: **Filtro de Momentum Cinético que bloquea la ejecución si la orden va en contra de la aceleración del entorno (banda muerta estocástica).**

---

## 2. Interferencia de Ondas y Parámetro de Kuramoto
El corazón de ZENIN es el cálculo de $\Phi_{\text{certeza}}(t)$. En lugar de colapsar probabilidades prematuramente, el sistema proyecta el Riesgo Estocástico ($I_{\text{CVaR}}$), el Ritmo Cronométrico ($\Lambda(t)$) y la Confianza de los Expertos ($\Phi_{\text{MoE,base}}$) como osciladores acoplados:

$$
\Phi_{\text{Master}}(t) = I_{\text{CVaR}} \cdot \Lambda(t) \cdot \Phi_{\text{MoE,base}} \cdot \Big( r(t) \cdot \alpha_{\text{align}} \Big)
$$

Para evitar el desvanecimiento de certeza (*Vanishing Certainty*), la alineación del sistema se evalúa midiendo la coherencia de fase a través del **Parámetro de Orden de Kuramoto** $r(t)$:

$$
r(t) = \left\lvert \frac{1}{N}\sum_{k=1}^N e^{i \theta_k(t)} \right\rvert = \sqrt{ \left(\frac{1}{N}\sum_{k=1}^N \cos\theta_k\right)^2 + \left(\frac{1}{N}\sum_{k=1}^N \sin\theta_k\right)^2 }
$$

> **Interpretación Física:** Si el riesgo se mitiga, el tiempo es óptimo y los expertos coinciden, los osciladores se sincronizan ($\theta_k \approx \theta_j$) y la interferencia es constructiva ($r(t) \to 1$). Si hay ruido blanco o perturbaciones caóticas, la interferencia es destructiva ($r(t) \to 0$) y el sistema se protege asumiendo un estado de `HOLD` inquebrantable.

---

## 3. Filtrado de Estado y Variedad Topológica (Módulo Rosa Roja)

Para preservar la estabilidad del atractor frente a perturbaciones estocásticas o singularidades exógenas en el espacio de fases, el módulo opera tres mecanismos analíticos acoplados:

### 3.1 Métrica de Mahalanobis Incremental O(d²) (Sherman-Morrison)

El tensor de estado continuo se valida en tiempo real preservando las correlaciones cruzadas mediante la distancia de Mahalanobis:

$$
d_M^2(\Delta s_t) = (\Delta s_t - \mu_n)^T \Sigma_n^{-1} (\Delta s_t - \mu_n)
$$

La matriz de covarianza inversa se actualiza de forma exacta en tiempo $\mathcal{O}(d^2)$ sobre la estimación de Welford sin aproximación diagonal:

$$
\Sigma_n^{-1} = \frac{1}{c} \left( \Sigma_{n-1}^{-1} - \frac{\mathbf{z}_n \mathbf{z}_n^T}{c + \mathbf{u}_n^T \mathbf{z}_n} \right)
$$

Donde $c = \frac{n-2}{n-1}$, $\mathbf{u}_n = \sqrt{\frac{n}{(n-1)^2}}\delta_n$ y $\mathbf{z}_n = \Sigma_{n-1}^{-1}\mathbf{u}_n$, aplicando simetrización activa $\Sigma_n^{-1} \leftarrow \frac{1}{2}(\Sigma_n^{-1} + (\Sigma_n^{-1})^T)$ y re-anclaje periódico por descomposición de Cholesky cada 500 pasos.

### 3.2 Motivos Topológicos Cinemáticos (Negentropía)

La variedad continua se discretiza en arquetipos cinemáticos invariantes a la escala mediante una función de proyección topológica:

$$
\mathbf{M}_t = \Phi_{\text{topo}}(\Delta S_t, \Delta t_t, \mathbf{M}_{t-1})
$$

El espacio se particiona en clases según dirección de Voronoi, aceleración relativa y cadencia temporal, proyectando la probabilidad sobre un simplex acotado $\Delta^{K-1}$ y maximizando la negentropía $J(P)$:

$$
J(P) = D_{\text{KL}}(P \parallel U) = \log_2(K) - H(\Theta \mid D_t)
$$

### 3.3 Random Walk con Muestreo por Importancia Guiado

La generación de trayectorias en el espacio de fases se rige por una mezcla estocástica adaptativa entre el grafo empírico local y un campo director macro inyectado vía puerto hexagonal (`GuidedFieldPort`):

$$
P_{\text{walk}}(M_{t+1} \mid M_t) = (1 - \beta) P_{\text{local}}(M_{t+1} \mid M_t) + \beta Q_{\text{global}}(M_{t+1} \mid M_t)
$$

El campo director global sintetiza la derivada armónica de Fourier, los priors bayesianos de régimen y la consistencia cinemática:

$$
Q_{\text{global}}(M' \mid M_t) \propto \Psi_{\text{Fourier}}(M') \cdot \Psi_{\text{Bayes}}(M') \cdot \mathcal{K}_{\text{kin}}(M_t, M')
$$

Donde $\beta = \exp(-N_{\text{local}}/\tau_{\text{densidad}})$ transiciona suavemente hacia el campo director armónico ante transiciones de fase abruptas o regiones de baja densidad de muestreo.

---

## Certificación y Testing
La estabilidad matemática del orquestador y la suite estocástica están certificadas por pruebas rigurosas.

```bash
# Validar invarianzas físicas y matemáticas del orquestador
pytest -v iot_machine_learning/tests/unit/market/test_zenin_v22_master_equation.py \
          iot_machine_learning/tests/unit/market/test_kuramoto_resonance.py \
          iot_machine_learning/tests/unit/market/test_master_orchestrator_invariance.py

# Validar métricas de Mahalanobis, Motivos y Random Walk Guiado
pytest -v iot_machine_learning/tests/unit/rosa_roja/test_mahalanobis_sherman_morrison.py \
          iot_machine_learning/tests/unit/rosa_roja/test_motif_and_guided_field.py \
          iot_machine_learning/tests/unit/rosa_roja/test_guided_random_walk.py

# Ejecutar auditoría completa del Motor (700+ tests)
pytest -v iot_machine_learning/tests/unit/market/
```