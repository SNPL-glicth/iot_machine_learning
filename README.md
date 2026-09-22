# ZENIN: Stochastic Decision Engine & Wave Resonance Orchestrator
> **Motor Agnóstico de Inferencia Continua basado en Caos Determinista y Sistemas Dinámicos**

ZENIN no opera mediante heurísticas estáticas o árboles de decisión booleanos. Es un orquestador matemático diseñado para tomar decisiones bajo incertidumbre extrema (mercados financieros o telemetría IoT) modelando las variables del entorno como **ondas en un espacio de fases**. 

La ejecución de una acción no depende de un cruce de indicadores, sino de la **interferencia constructiva** (resonancia) entre el riesgo, la inercia temporal y la confianza epistémica.

---

## 1. La Ecuación de Acción (Soberanía de Ejecución)
El orquestador emite comandos al mundo real a través de una función de estado continuo modulada por la inercia del sistema:

$$\mathcal{O}_{\text{Master}}(t) = \mu_{\text{obj}}(t) \cdot \Theta\Big( \Phi_{\text{certeza}}(t) - \gamma_{\text{exec}} \Big) \cdot \mathcal{V}_{\text{mom}}(t)$$

* **$\mu_{\text{obj}}(t)$**: Magnitud del vector objetivo calculada por consenso bayesiano.
* **$\Theta(\cdot)$**: Función de activación (Gating) que evalúa si la certeza resonante supera el umbral crítico $\gamma_{\text{exec}}$.
* **$\mathcal{V}_{\text{mom}}(t)$**: Filtro de Momentum Cinético que bloquea la ejecución si la orden va en contra de la aceleración del entorno (banda muerta estocástica).

---

## 2. Interferencia de Ondas y Parámetro de Kuramoto
El corazón de ZENIN es el cálculo de $\Phi_{\text{certeza}}(t)$. En lugar de colapsar probabilidades prematuramente, el sistema proyecta el Riesgo Estocástico ($I_{\text{CVaR}}$), el Ritmo Cronométrico ($\Lambda(t)$) y la Confianza de los Expertos ($\Phi_{\text{MoE,base}}$) como osciladores acoplados:

$$\Phi_{\text{Master}}(t) = I_{\text{CVaR}} \cdot \Lambda(t) \cdot \Phi_{\text{MoE,base}} \cdot \Big( r(t) \cdot \alpha_{\text{align}} \Big)$$

Para evitar el desvanecimiento de certeza (*Vanishing Certainty*), la alineación del sistema se evalúa midiendo la coherencia de fase a través del **Parámetro de Orden de Kuramoto $r(t)$**:

$$r(t) = \left\vert{} \frac{1}{N}\sum_{k=1}^N e^{i \theta_k(t)} \right\vert{} = \sqrt{ \left(\frac{1}{N}\sum_{k=1}^N \cos\theta_k\right)^2 + \left(\frac{1}{N}\sum_{k=1}^N \sin\theta_k\right)^2 }$$

> **Interpretación Física:** Si el riesgo se mitiga, el tiempo es óptimo y los expertos coinciden, los osciladores se sincronizan ($\theta_k \approx \theta_j$) y la interferencia es constructiva ($r(t) \to 1$). Si hay ruido blanco o fracturas en el mercado, la interferencia es destructiva ($r(t) \to 0$) y el sistema se protege asumiendo un estado de `HOLD` inquebrantable.

---

## 3. Filtrado de Estado y Distancia de Mahalanobis (Módulo Rosa Roja)
Para que las ondas no se contaminen con *shocks* o datos corruptos (ruido de sensores o *flash crashes*), el tensor de estado pasa por un filtro multidimensional online $\mathcal{O}(1)$:

$$d_M^2(\Delta s_t) = (\Delta s_t - \mu_n)^T \Sigma_n^{-1} (\Delta s_t - \mu_n)$$

Cualquier vector de estado $\Delta s_t$ que supere la tolerancia topológica $\tau_{\text{noise}}$ en la matriz de covarianza $\Sigma_n$ es rechazado antes de que pueda perturbar el atractor continuo.

---

## 🧪 Certificación y Testing
La estabilidad matemática del orquestador y la invarianza de Kuramoto están garantizadas por una suite de pruebas ISO-compliant.

```bash
# Validar invarianzas físicas y matemáticas del orquestador
pytest -v iot_machine_learning/tests/unit/market/test_zenin_v22_master_equation.py \
          iot_machine_learning/tests/unit/market/test_kuramoto_resonance.py \
          iot_machine_learning/tests/unit/market/test_master_orchestrator_invariance.py

# Ejecutar auditoría completa del Motor (700+ tests)
pytest -v iot_machine_learning/tests/unit/market/