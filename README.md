# ZENIN: Topological Inference Engine & Continuous Geometric Orchestrator

> **Motor Matemático Continuo y Determinista para Inferencia Topológica en Sistemas Dinámicos Complejos**

ZENIN modela la evolución temporal de sistemas dinámicos multivariados como un **flujo continuo sobre variedades riemannianas acopladas bajo la Fibración de Hopf en $\mathbb{C}^2$**. La toma de decisiones no emplea heurísticas discretas ni aproximaciones lineales: se rige por la estabilidad topológica del atractor, la coherencia cuántica de fase $U(1)$ y la dualidad canónica entre cinemática directa (inercia forward $+x$) y cinemática inversa conjugada (rebote elástico $-x$).

---

## 1. La Ecuación Maestra Soberana Completa

> **Espacio Matemático:** Fibrado Compuesto $\mathcal{M} \times (\mathbb{C}^2 \to S^2) \times \mathbb{R}^d$.  
> **Ubicación en Código:** [`master_equation.py:L83-L172`](infrastructure/ml/master_engine/master_equation.py#L83-L172), [`orchestrator.py:L115-L178`](infrastructure/ml/master_engine/orchestrator.py#L115-L178), [`hopf_spinor_state.py:L17-L62`](domain/entities/manifold/hopf_spinor_state.py#L17-L62).

ZENIN gobierna la inferencia topológica integrando sus dos motores canónicos duales: **Rosa Roja** (Cinemática Directa forward $+x$, Polo $z_1$) y **MRT** (Cinemática Inversa conjugada $-x$, Polo $z_2$), modulados por la disipación geométrica de Liouville y protegidos por cortafuegos de admisibilidad estricta en una **única variable continua soberana**.

### Lo que hay: Formulación Matemática de la Ecuación Maestra

#### A. La Ecuación Unificada Soberana: Variable Destino $\mathcal{D}(t)$
En cada instante temporal $t$, el estado completo del sistema dinámico colapsa de forma analítica y continua en una **única variable de salida escalar unificada $\mathcal{D}(t)$**, componiendo multiplicativamente los cuatro pilares del sistema:

$$
\mathcal{D}(t) = \left[ \prod_{k} \mathbb{I}_k(t) \right] \cdot \exp\left(-\max\left(0, \; \mathrm{div}\mathbf{F}(t)\right)\right) \cdot \Pi_{\mathrm{sov}}(t) \cdot \big|S_3(t) + S_1(t)\big|
$$

##### Descomposición de los Cuatro Factores Fundamentales:
1. **$\prod_{k} \mathbb{I}_k(t) \in \{0, 1\}$ (Veto de Admisibilidad Estricto Multicapa):**
   Producto de cortafuegos booleanos deterministas. Si cualquier subsistema detecta violación crítica de invariantes de estabilidad, anula instantáneamente la emisión del sistema ($\prod_k \mathbb{I}_k = 0$):
   * $\mathbb{I}_{\mathrm{mahal}}(t) = \mathbb{I}\big(d_M^2(t) \le \chi_{d, \alpha}^2\big)$: Filtro métrico de ingesta contra perturbaciones y ruido en `module1_ingestion.py`.
   * $\mathbb{I}_{\mathrm{takens}}(t) = \mathbb{I}\big(\Omega_{\mathrm{FNN}}(t) \lt \tau_{\mathrm{FNN}}\big)$: Cortafuegos topológico contra falsos vecinos más cercanos y auto-intersección del atractor en el espacio embebido de Takens.
   * $\mathbb{I}_{\mathrm{risk}}(t) = \mathbb{I}\big(I_{\mathrm{cvar}}(t) \gt 0\big)$: Admisibilidad estocástica de cola y estabilidad variacional.
2. **$\exp\left(-\max\left(0, \; \mathrm{div}\mathbf{F}(t)\right)\right) \in (0, 1]$ (Freno Disipativo de Liouville en la Variedad $\mathcal{M}$):**
   Amortiguamiento volumétrico continuo sobre la variedad Riemanniana $\mathcal{M}$. Si la divergencia del campo $\mathrm{div}\mathbf{F} \gt 0$ (expansión inestable del volumen de fases), atenúa exponencialmente la señal colapsando suavemente la dinámica hacia el reposo, garantizando estabilidad asintótica sin discontinuidades.
3. **$\Pi_{\mathrm{sov}}(t) = \mathrm{sgn}\big(S_3(t) + S_1(t)\big) \in \{-1.0, +1.0\}$ (Polaridad Soberana de Stokes):**
   Sentido direccional de fase emitido por la Fibración de Hopf $\mathbb{C}^2 \to S^2$:
   * $\Pi_{\mathrm{sov}} = +1.0$: Dominio del polo forward inercial $z_1$ (Cinemática Directa hacia adelante).
   * $\Pi_{\mathrm{sov}} = -1.0$: Dominio del polo de rebote elástico $z_2$ (Cinemática Inversa restauradora).
4. **$\big|S_3(t) + S_1(t)\big| = |C_{\mathrm{sovereign}}(t)| \in [0, 1]$ (Magnitud de Certeza Soberana Acoplada):**
   Interferencia analítica cuántica entre Rosa Roja ($z_1$) y MRT ($z_2$) sobre las coordenadas de Stokes de la 2-esfera $S^2$.

---

#### B. Representación Espinorial en $\mathbb{C}^2$ y Proyección de Hopf
> **Espacio Matemático:** Espacio de Hilbert $\mathcal{H} \cong \mathbb{C}^2$ acoplado a la 3-esfera $S^3 \subset \mathbb{C}^2$.  
> **Ubicación en Código:** [`hopf_spinor_state.py:L17-L62`](domain/entities/manifold/hopf_spinor_state.py#L17-L62), [`mrt_hopf_fibration.py:L42-L130`](domain/services/manifold/mrt_hopf_fibration.py#L42-L130).

El estado dinámico acoplado se representa formalmente como un espinor de dos componentes en el espacio de Hilbert $\mathcal{H} \cong \mathbb{C}^2$:

$$
|\psi(t)\rangle = \begin{pmatrix} z_1(t) \\ z_2(t) \end{pmatrix} = \begin{pmatrix} |z_1|e^{i\theta_1} \\ |z_2|e^{i\theta_2} \end{pmatrix} \in \mathbb{C}^2
$$

Las amplitudes de ambos modos conjugados se evalúan de forma analítica y continua:

* **Polo Positivo $z_1 \in \mathbb{C}$ (Rosa Roja — Cinemática Directa $+x$):**
  Amplitud de inercia y ritmo forward en $\mathbb{R}^3$, modulada por el indicador de estabilidad $\mathbb{I}_{\mathrm{estable}}$, sincronía cronométrica $\Lambda(t)$ y alineación de fase Kuramoto ($r \cdot \cos\Delta\phi$):
  $$
  |z_1| = \Phi_{\mathrm{MoE, final}} \cdot \mathbb{I}_{\mathrm{estable}} \cdot \Lambda(t) \cdot (r_{\mathrm{Kuramoto}} \cdot \cos\Delta\phi_{\mathrm{align}})
  $$

* **Polo Negativo $z_2 \in \mathbb{C}$ (MRT — Cinemática Inversa $-x$):**
  Amplitud de deformación de vórtice y rebote elástico en el espacio conjugado 4D $\mathbb{R}^4$:
  $$
  |z_2| = \frac{1}{\sqrt{1 + u^2}} \cdot \mathcal{D}_{\mathrm{Ramanujan}} \cdot C_{\mathrm{nominal}}, \qquad u = \frac{\dot{\mathcal{E}}_{\mathrm{frob}}}{\varepsilon_{\mathrm{strain}}}
  $$

A través de la proyección de Hopf $\pi: S^3 \to S^2$, el espinor proyecta sus coordenadas sobre la esfera de Stokes:

$$
S_0 = |z_1|^2 + |z_2|^2, \qquad S_1 = 2|z_1||z_2|\cos(\theta_1 - \theta_2)
$$

$$
S_2 = 2|z_1||z_2|\sin(\theta_1 - \theta_2), \qquad S_3 = |z_1|^2 - |z_2|^2
$$

Cumpliendo en todo instante la invariancia pitagórica del espacio de fases:

$$
S_1^2 + S_2^2 + S_3^2 \equiv S_0^2
$$

---

#### C. La Ecuación Soberana y Polaridad Continua
> **Espacio Matemático:** Coordenadas de Stokes $(S_1, S_2, S_3) \in S^2 \subset \mathbb{R}^3$.  
> **Ubicación en Código:** [`master_equation.py:L109-L123`](infrastructure/ml/master_engine/master_equation.py#L109-L123).

La **Certeza Soberana** $C_{\mathrm{sovereign}}(t) \in \mathbb{R}$ y la polaridad direccional $\Pi_{\mathrm{sovereign}} \in \{-1.0, +1.0\}$ se evalúan de forma analítica y suave ($C^\infty$):

$$
C_{\mathrm{sovereign}}(t) = S_3 + S_1 = \big(|z_1|^2 - |z_2|^2\big) + 2|z_1||z_2|\cos(\theta_1 - \theta_2)
$$

$$
\Phi_{\mathrm{sovereign}} = \min\big(1.0, \; |C_{\mathrm{sovereign}}|\big), \qquad \Pi_{\mathrm{sovereign}} = \mathrm{sgn}\big(C_{\mathrm{sovereign}}\big)
$$

---

#### D. Modulación del Estado Objetivo y Blindaje de Escala Absoluta
> **Espacio Matemático:** Variedad afín tangente $\mathbb{R}^d$ con invariante de positividad de escala.  
> **Ubicación en Código:** [`master_equation.py:L111-L123`](infrastructure/ml/master_engine/master_equation.py#L111-L123), [`orchestrator.py:L153-L157`](infrastructure/ml/master_engine/orchestrator.py#L153-L157).

Sea $S_{\mathrm{ref}} \gt 0$ la magnitud del estado de referencia actual, $y_1$ el objetivo proyectado por cinemática directa (Rosa Roja) e $y_2$ el objetivo de rebote conjugado (MRT). La polaridad soberana modula **estrictamente el diferencial de transición relativo ($\Delta S$)**, protegiendo la escala absoluta del sistema contra inversiones de signo o valores no físicos:

$$
\Delta S_{\mathrm{blend}} = \frac{|z_1|^2(y_1 - S_{\mathrm{ref}}) + |z_2|^2(y_2 - S_{\mathrm{ref}})}{S_0 + \varepsilon}
$$

$$
\Delta S_{\mathrm{sovereign}} = \Delta S_{\mathrm{blend}} \cdot \Pi_{\mathrm{sovereign}}
$$

$$
S_{\mathrm{target}} = \max\big(10^{-6}, \; S_{\mathrm{ref}} + \Delta S_{\mathrm{sovereign}}\big)
$$

---

#### E. El Operador Soberano de Decisión $\mathbf{U}(t)$
> **Espacio Matemático:** Espacio discreto de políticas de control inducido por estabilidad topológica.  
> **Ubicación en Código:** [`orchestrator.py:L174-L178`](infrastructure/ml/master_engine/orchestrator.py#L174-L178), [`execution_plan.py:L12-L45`](domain/entities/execution_plan.py#L12-L45).

La directiva operativa $\mathbf{U}(t) \in \{\mathrm{EXECUTE}, \mathrm{HOLD}, \mathrm{EMERGENCY\text{-}FLUSH}\}$ es gobernada de forma determinista por las condiciones de admisibilidad y estabilidad topológica:

$$
\mathbf{U}(t) = \begin{cases}
\mathrm{EXECUTE}\big(S_{\mathrm{target}}, \; \Pi_{\mathrm{sovereign}}\big) & \text{si } \Phi_{\mathrm{sovereign}} \ge \gamma_{\mathrm{exec}} \;\land\; \mathbb{I}_{\mathrm{admisibilidad}} = 1 \;\land\; V_{\mathrm{inercia}} \gt 0 \\[6pt]
\mathrm{HOLD} & \text{si } \Phi_{\mathrm{sovereign}} \lt \gamma_{\mathrm{exec}} \;\lor\; V_{\mathrm{inercia}} \le 0 \\[6pt]
\mathrm{EMERGENCY\text{-}FLUSH} & \text{si } d_M^2 \gt \chi_{d, \alpha}^2 \;\lor\; \mathbb{I}_{\mathrm{risk}} = 0 \;\lor\; \Omega_{\mathrm{FNN}} \ge \tau_{\mathrm{FNN}}
\end{cases}
$$

Donde la directiva de ejecución no recurre a heurísticas empíricas sino a criterios analíticos:
* **$\Phi_{\mathrm{sovereign}} \ge \gamma_{\mathrm{exec}}$:** La certeza analítica acoplada supera el umbral crítico de activación variacional.
* **$\mathbb{I}_{\mathrm{admisibilidad}} = \prod_k \mathbb{I}_k = 1$:** Admisibilidad simultánea de todos los invariantes (Mahalanobis, Takens, CVaR).
* **$V_{\mathrm{inercia}} \gt 0$:** La velocidad de fase supera la banda muerta del ruido del sistema.
* **$\mathrm{EMERGENCY\text{-}FLUSH}$:** Se activa de forma no negociable ante anulación métrica ($d_M^2 \gt \chi^2$), colapso de riesgo estocástico ($\mathbb{I}_{\mathrm{risk}} = 0$) o plegamiento topológico del atractor ($\Omega_{\mathrm{FNN}} \ge \tau_{\mathrm{FNN}}$).

### El por qué: Justificación Física y Teórica

1. **Eliminación de Singularidades y Chattering Numérico ($C^0$ Chattering vs. Suavidad $C^\infty$):**  
   Las arquitecturas discretas (`if/else`) o funciones de atenuación basadas en cortes discontinuos poseen derivadas no suaves en la frontera de estabilidad, inyectando impulsos singulares de Dirac en la aceleración del sistema:
   $$
   f_{\mathrm{corte}}(x) = \max(0, x) \implies \frac{df}{dx} = \Theta(x) \implies \frac{d^2f}{dx^2} = \delta(x)
   $$
   La presencia de la distribución impulsiva $\delta(x)$ induce micro-oscilaciones parásitas (*$C^0$ chattering*) en la frontera de decisión. En contraposición, la Fibración de Hopf proyecta el estado mediante un flujo analíticamente suave ($C^\infty$) en todo su dominio:
   $$
   \pi \in C^\infty\big(\mathbb{C}^2 \setminus \{0\}, \; S^2\big) \implies \nabla C_{\mathrm{sovereign}} \in C^\infty
   $$
   Garantizando gradientes continuos, derivadas acotadas y estabilidad asintótica sin singularidades numéricas.

2. **Conservación Estricta de Energía Informacional e Invariante de Casimir:**  
   Se cumple idénticamente la restricción pitagórica como invariante cuadrático de Casimir del álgebra de Lie $\mathfrak{su}(2)$:
   $$
   S_1^2 + S_2^2 + S_3^2 \equiv S_0^2
   $$
   $$
   \mathcal{C}_{\mathfrak{su}(2)} = \sum_{k=1}^3 \sigma_k^2 = \text{const} \implies \frac{d}{dt}\left(\sum_{k=1}^3 S_k^2 - S_0^2\right) = 0
   $$
   Como invariante de Casimir, garantiza que la probabilidad total del espacio de fases se conserva de forma unitaria; toda energía disipada fuera del régimen observable ($z_1$) es absorbida con exactitud por el modo conjugado ($z_2$), impidiendo fugas térmicas o pérdidas de información en la variedad.

---

## 2. El Motor Rosa Roja (Cinemática Directa $+x$, Polo $z_1$)

> **Espacio Matemático:** Cinemática Directa Forward en Espacio Tangente $(+x, +t) \subset T\mathcal{M} \times \mathbb{R}^d$.  
> **Ubicación en Código:** [`engine.py:L27-L163`](infrastructure/ml/engines/rosa_roja/algorithms/engine.py#L27-L163), [`module1_ingestion.py:L26-L180`](infrastructure/ml/engines/rosa_roja/algorithms/modules/module1_ingestion.py#L26-L180), [`rhythm_generator.py:L18-L90`](infrastructure/ml/engines/rosa_roja/algorithms/modules/rhythm_generator.py#L18-L90), [`module3_moe_gating.py:L14-L80`](infrastructure/ml/engines/rosa_roja/algorithms/modules/module3_moe_gating.py#L14-L80).

El motor [RosaRojaMoEEngine](infrastructure/ml/engines/rosa_roja/engine.py) opera en el régimen de flujo forward ($+x, +t$). Modela la inercia del sistema, genera trayectorias geodésicas candidatas y evalúa el consenso no lineal a través de un jurado de expertos con filtrado métrico elíptico.

### Lo que hay: Sistema de Ecuaciones de Rosa Roja

#### A. Ingestión Elíptica y Métrica de Mahalanobis en $\mathcal{S}_{++}^d$
> **Espacio Matemático:** Cono simétrico de matrices definidas positivas $\mathcal{S}_{++}^d \subset \mathbb{R}^{d \times d}$.  
> **Ubicación en Código:** [`module1_ingestion.py:L26-L180`](infrastructure/ml/engines/rosa_roja/algorithms/modules/module1_ingestion.py#L26-L180).

A partir de la transición incremental del estado $\Delta S \in \mathbb{R}^d$, se computa la distancia elíptica de Mahalanobis respecto a la media acumulada $\boldsymbol{\mu}$:

$$
d_M^2(\Delta S) = (\Delta S - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\Delta S - \boldsymbol{\mu})
$$

El filtro garantiza la integridad de la variedad bloqueando perturbaciones anómalas:

$$
d_M^2(\Delta S) \le \chi_{d, \alpha}^2 \implies \text{Paso Válido (Tracking Activo)}
$$

La matriz de precisión $\mathbf{\Sigma}^{-1}$ se actualiza en línea de forma continua en $\mathcal{O}(d^2)$ mediante el lema de Sherman-Morrison con factor de olvido $\alpha$:

$$
\mathbf{\Sigma}_{t+1}^{-1} = \frac{1}{1 - \alpha}\left( \mathbf{\Sigma}_t^{-1} - \frac{\alpha \mathbf{\Sigma}_t^{-1} \mathbf{u} \mathbf{u}^T \mathbf{\Sigma}_t^{-1}}{1 - \alpha + \alpha \mathbf{u}^T \mathbf{\Sigma}_t^{-1} \mathbf{u}} \right), \qquad \mathbf{u} = \Delta S - \boldsymbol{\mu}
$$

#### B. Generador de Trayectorias Rítmicas Forward y Fase $\theta_1$
> **Espacio Matemático:** Espacio de trayectorias en el fibrado tangente $T\mathcal{M}$ y grupo de fase $U(1)$.  
> **Ubicación en Código:** [`rhythm_generator.py:L18-L90`](infrastructure/ml/engines/rosa_roja/algorithms/modules/rhythm_generator.py#L18-L90), [`phi_ritmo_scorer.py:L15-L65`](infrastructure/ml/engines/rosa_roja/algorithms/modules/phi_ritmo_scorer.py#L15-L65).

A partir del movimiento actual, el motor sintetiza un haz de trayectorias candidatas $T_k = \{\mathbf{x}_1, \dots, \mathbf{x}_H\}$ a lo largo del horizonte $H$, gobernadas por la coherencia rítmica:

$$
\Phi_{\text{ritmo}}(T_k) = \frac{1}{H}\sum_{h=1}^H \cos(\omega h \Delta t + \theta_1) \cdot e^{-\gamma_r h \Delta t}
$$

La fase intrínseca del polo forward $\theta_1$ se extrae del vector de velocidad y la curvatura métrica:

$$
\theta_1 = \arctan\left(\frac{v_d}{v_1 + \varepsilon}\right) \pmod{2\pi}
$$

#### C. MoE Gating Multiplicativo y Amplitud del Polo Positivo $|z_1|$
> **Espacio Matemático:** Símplex de probabilidad $\Delta^E$ y proyección de certeza en $[0, 1]$.  
> **Ubicación en Código:** [`module3_moe_gating.py:L14-L80`](infrastructure/ml/engines/rosa_roja/algorithms/modules/module3_moe_gating.py#L14-L80), [`engine.py:L115-L135`](infrastructure/ml/engines/rosa_roja/algorithms/engine.py#L115-L135).

El consenso del jurado de $E$ expertos pondera la evidencia individual $\Psi_e(T)$ sujeta a veto estricto de expertos críticos ($\mathcal{K}_{\text{crít}}$) y penalización cuadrática por varianza entre expertos:

$$
\Phi_{\text{MoE}}(T) = \left[ \prod_{k \in \mathcal{K}_{\text{crít}}} \mathbb{I}\big(\Psi_k(T) \ge \tau_k\big) \right] \cdot \frac{\sum_{e=1}^E w_e \Psi_e(T)}{1 + \gamma_{\text{var}} \text{Var}(\Psi)}
$$

La amplitud del polo positivo $|z_1|$ modula la certeza forward acoplada a la tasa de deriva (*drift* $\lambda_{\text{drift}}$):

$$
|z_1| = \Phi_{\text{MoE, final}} = \max\Big(0.0, \; \min\big(1.0, \; \Phi_{\text{MoE}} \cdot \big(1 - \lambda_{\text{drift}}(1 - \Phi_{\text{ritmo}})\big)\big)\Big)
$$

El objetivo forward directo proyectado es la esperanza condicional de la trayectoria óptima seleccionada:

$$
y_1 = \mu(T_{\text{chosen}}) = \frac{\sum_{i=1}^H w_i x_i}{\sum_{i=1}^H w_i}
$$

### El por qué: Justificación Física y Teórica
1. **Inercia y Propagación Newtoniana Forward:** En régimen laminar o estacionario, el sistema evoluciona impulsado por su momento intrínseco. Rosa Roja modela esta inercia sin sobre-ajustes, proyectando la trayectoria geodésica forward natural ($y_1$).
2. **Filtrado Métrico Elíptico ($\chi^2$):** La distribución multivariada de transiciones no es esférica sino elipsoidal; la distancia de Mahalanobis evita falsos positivos en ejes de alta varianza natural y detecta instantáneamente anomalías en direcciones de alta rigidez.
3. **Consenso Crítico Multiplicativo:** Un promedio lineal permite que un experto con alta confianza compense la alarma de otro experto que detecta peligro. El producto con indicador booleano $\prod \mathbb{I}$ confiere poder de veto absoluto a los sensores críticos, garantizando que el polo positivo $|z_1|$ colapse inmediatamente a cero si cualquier invariante físico es vulnerado.

---

## 3. El Motor MRT (Maxwell-Ramanujan-Tesla)

> **Espacio Matemático:** Cinemática Inversa en Espacio Conjugado $(-x, -t) \subset T\mathcal{M} \times \mathbb{R}^4$.  
> **Ubicación en Código:** [`mrt_engine.py:L61-L113`](infrastructure/ml/engines/mrt/mrt_engine.py#L61-L113), [`mrt_pipeline.py:L62-L128`](infrastructure/ml/engines/mrt/algorithms/mrt_pipeline.py#L62-L128).

El motor [MRTEngine](infrastructure/ml/engines/mrt/mrt_engine.py) opera en el espacio conjugado ($-x, -t$) como un **Espejo de Fase Conjugada** (Phase Conjugate Mirror - PCM). Resuelve el vector de recuperación elástica cuando el sistema entra en sobretensión o régimen turbulento.

### Lo que hay: Sistema de Ecuaciones de MRT

#### A. Maxwell: Vorticidad Rotacional y Circulación de Momento
> **Espacio Matemático:** Espacio de fases $(x, v, a) \in \mathbb{R}^3$ con álgebra de espín antisimétrico $\mathfrak{so}(3)$.  
> **Ubicación en Código:** [`maxwell_curl_field.py:L16-L74`](infrastructure/ml/engines/mrt/algorithms/modules/maxwell_curl_field.py#L16-L74), [`phase_conjugator.py:L46-L107`](infrastructure/ml/engines/mrt/phase_conjugator.py#L46-L107).

A partir de la serie temporal de observaciones $x(t)$ con paso $\Delta t$, se construye el espacio de derivadas $(x, v, a)$ con $v = \dot{x}$, $a = \ddot{x}$ y $j = \dot{a}$ (jerk). El momento angular en el espacio de fases $\mathbf{L} = \mathbf{v} \times \mathbf{a}$ induce la circulación magnética:

$$
L_x = (a_1 \cdot j) - (v_1 \cdot a_0), \quad L_y = (v_1 \cdot j) - (a_1 \cdot v_0), \quad L_z = (v_1 \cdot a_1) - (v_0 \cdot a_0)
$$

$$
\|\nabla \times \mathbf{B}\| = \frac{\sqrt{L_x^2 + L_y^2 + L_z^2}}{|v_1| + \varepsilon}
$$

El tensor de espín antisimétrico asociado al Jacobiano es:

$$
\mathbf{\Omega} = \frac{1}{2}\left(\mathbf{J} - \mathbf{J}^T\right) \implies \|\nabla \times \mathbf{B}\|_F = \sqrt{2}\|\mathbf{\Omega}\|_F
$$

#### B. Ramanujan: Cristal Simpléctico y Amortiguador de Liouville
> **Espacio Matemático:** Espacio simpléctico aumentado $\mathbb{R}^4$ con tensor de deformación métrica $\dot{\mathcal{E}}_{\text{frob}}$.  
> **Ubicación en Código:** [`ramanujan_crystal.py:L16-L89`](infrastructure/ml/engines/mrt/algorithms/modules/ramanujan_crystal.py#L16-L89).

El Jacobiano $3\text{D}$ se acopla a la tasa de deformación de Frobenius $\dot{\mathcal{E}}_{\text{frob}} = \sqrt{\left(\frac{|v_1 - v_0|}{\Delta t}\right)^2 + a_1^2}$ en un tensor aumentado $4\text{D}$:

$$
\mathbf{J}_{\text{4D}} = \begin{bmatrix} \mathbf{J}_{\text{3D}} & \mathbf{w} \\ \mathbf{w}^T & -\dot{\mathcal{E}}_{\text{frob}} \end{bmatrix}, \qquad \mathbf{J}^{\star} = \left(\mathbf{J}_{\text{4D}}^T \mathbf{J}_{\text{4D}} + \varepsilon_{\text{crystal}}\mathbf{I}\right)^{-1} \mathbf{J}_{\text{4D}}^T
$$

La traza contractiva del cristal impone la condición de disipación simpléctica:

$$
\mathrm{Tr}(\mathbf{J}^{\star}) \lt 0 \implies \mathcal{D}_{\text{Ramanujan}} = \frac{1}{1 + \exp\big(\mathrm{Tr}(\mathbf{J}^{\star})\big)} \in [0.5, 1.0]
$$

#### C. Tesla: Transporte Disipativo de Fase sobre $U(1)$ y Polo $z_2$
> **Espacio Matemático:** Fibrado principal $U(1) \cong S^1$ (Círculo unitario de fase) y proyección de Hilbert $\mathbb{C}$.  
> **Ubicación en Código:** [`hopf_spinor_field.py:L27-L107`](infrastructure/ml/engines/mrt/algorithms/modules/hopf_spinor_field.py#L27-L107), [`phase_conjugator.py:L80-L107`](infrastructure/ml/engines/mrt/phase_conjugator.py#L80-L107).

La fase angular $\theta_2$ transporta la acumulación de vorticidad con amortiguación disipativa $\gamma$:

$$
\theta_2(t + \Delta t) = \left[\theta_2(t) - \Big(\|\nabla \times \mathbf{B}\| + \dot{\mathcal{E}}_{\text{frob}}\Big)\Delta t\right] e^{-\gamma \Delta t} \pmod{2\pi}
$$

La amplitud cuadrática racional del polo negativo $z_2$ se evalúa mediante sustitución algebraica exacta sin funciones trascendentes lentas:

$$
u = \frac{\dot{\mathcal{E}}_{\text{frob}}}{\varepsilon_{\text{strain}}}, \qquad |z_2| = \frac{1}{\sqrt{1 + u^2}} \cdot \mathcal{D}_{\text{Ramanujan}} \cdot C_{\text{nominal}}
$$

#### D. Vector de Rebote Elástico (Cinemática Inversa)
> **Espacio Matemático:** Fibrado tangente $T\mathcal{M}$ (Espacio de velocidades y trayectorias conjugadas).  
> **Ubicación en Código:** [`mrt_pipeline.py:L62-L128`](infrastructure/ml/engines/mrt/algorithms/mrt_pipeline.py#L62-L128), [`mrt_engine.py:L61-L113`](infrastructure/ml/engines/mrt/mrt_engine.py#L61-L113).

La velocidad de retroceso conjugado $v_{\text{rebound}}$ y la trayectoria multi-paso $x_{\text{rebound}}(k)$ para un horizonte $K$ se proyectan como:

$$
v_{\text{rebound}} = -v_1 \cdot k_{\text{elastic}} \left(1 + \frac{\|\nabla \times \mathbf{B}\|}{1 + \|\nabla \times \mathbf{B}\|}\right)
$$

$$
x_{\text{rebound}}(k) = x(t) + v_{\text{rebound}} \cdot k\Delta t \cdot e^{-\gamma k \Delta t}, \qquad k \in \{1, \dots, K\}
$$

### El por qué: Justificación Física y Teórica
1. **Maxwell (Ortogonalidad e Inducción):** En colisiones de alta energía, la aceleración no debe absorberse frontalmente en el eje de inercia ($\mathbf{E}$). La vorticidad $\mathbf{\Omega}$ transfiere la energía cinética al modo transversal magnético ($\mathbf{E} \cdot \mathbf{B} = 0$), evitando la rotura del tracking.
2. **Ramanujan (Contracción de Liouville):** Por el teorema de Liouville ($\frac{d\Omega_V}{dt} = \mathrm{Tr}(\mathbf{J}^{\star})\Omega_V$), una traza estrictamente negativa ($\mathrm{Tr}(\mathbf{J}^{\star}) \lt 0$) colapsa exponencialmente el volumen del espacio de fases, transformando trayectorias que divergían al infinito en geodésicas compactas de recuperación elástica.
3. **Tesla (Espejo Conjugado y Veto Inverso):** Cuando el sistema sufre una perturbación extrema, las fases entran en oposición destructiva ($\Delta\theta \to \pi \implies \cos\Delta\theta \to -1.0$). El término de Stokes $S_1$ se vuelve negativo y supera a $S_3$, induciendo $\Pi_{\text{sovereign}} = -1.0$. Esto invierte la dirección de la respuesta en el instante exacto de compresión máxima, restaurando el equilibrio dinámico.

---

## 4. La Variedad Geométrica Continua y el Tensor Jacobiano

> **Espacio Matemático:** Variedad Riemanniana tridimensional $\mathcal{M} \subset \mathbb{R}^3$ con métrica local $\mathbf{J}(\mathbf{x}) = \nabla\mathbf{F}$.  
> **Ubicación en Código:** [`geometric_manifold_adapter.py:L63-L139`](infrastructure/ml/master_engine/geometric_manifold_adapter.py#L63-L139), [`divergence_compass.py:L20-L80`](domain/services/manifold/divergence_compass.py#L20-L80).

El estado del sistema evoluciona sobre una variedad riemanniana tridimensional $\mathcal{M} \subset \mathbb{R}^3$:

$$
\mathbf{x}(t) = \begin{bmatrix} x_1(t) \\ x_2(t) \\ x_3(t) \end{bmatrix} = \begin{bmatrix} d_M(t) & \text{(Tensión Métrica de Mahalanobis)} \\ r(t) & \text{(Coherencia de Fase de Kuramoto)} \\ P(t) & \text{(Creencia Epistémica Bayesiana)} \end{bmatrix}
$$

El acoplamiento se evalúa mediante el **Tensor Jacobiano analítico** $\mathbf{J}(\mathbf{x}) = \nabla\mathbf{F}$:

$$
\mathbf{J}(\mathbf{x}) = \begin{bmatrix}
\frac{\partial F_1}{\partial x_1} & \frac{\partial F_1}{\partial x_2} & \frac{\partial F_1}{\partial x_3} \\
\frac{\partial F_2}{\partial x_1} & \frac{\partial F_2}{\partial x_2} & \frac{\partial F_2}{\partial x_3} \\
\frac{\partial F_3}{\partial x_1} & \frac{\partial F_3}{\partial x_2} & \frac{\partial F_3}{\partial x_3}
\end{bmatrix}
$$

### Sincronización de Kuramoto
El orden de coherencia de fase entre $N$ subsistemas acoplados es:

$$
r(t) = \left\lvert \frac{1}{N}\sum_{k=1}^N e^{i \theta_k(t)} \right\rvert = \sqrt{ \left(\frac{1}{N}\sum_{k=1}^N \cos\theta_k\right)^2 + \left(\frac{1}{N}\sum_{k=1}^N \sin\theta_k\right)^2 }
$$

Las derivadas cruzadas (ej. $\mathbf{J}_{21}$) absorben anomalías métricas locales drenando suavemente la coherencia de fase sin inducir discontinuidades en el actuador.

---

## 5. Inmersión de Takens: Reconstrucción de Variables Ocultas

> **Espacio Matemático:** Espacio de inmersión euclidiano $\mathbb{R}^m$ inducido por retardo temporal sobre el atractor caótico.  
> **Ubicación en Código:** [`orchestrator.py:L80-L160`](infrastructure/ml/master_engine/orchestrator.py#L80-L160).

Bajo el **Teorema de Inmersión de Takens**, se reconstruye la topología del espacio de fases no observable a partir del retardo temporal de un escalar medido:

$$
\mathbf{y}_t = \big[x_t, \; x_{t-\tau_1}, \; x_{t-\tau_2}, \; \dots, \; x_{t-\tau_{m-1}}\big]^T \in \mathbb{R}^m
$$

### Cortafuegos Topológico (Poder de Veto)
Se evalúa la Dimensión Efectiva de Participación ($D_{\text{eff}}$) y la fracción de **Falsos Vecinos Más Cercanos ($\Omega_{\text{FNN}}$)**:

$$
\mathbb{I}_{\mathrm{takens}}(t) = \mathbb{I}\big(\Omega_{\mathrm{FNN}}(t) \lt \tau_{\mathrm{FNN}}\big) \cdot \mathbb{I}\big(D_{\mathrm{eff}}(t) \ge D_{\min}\big)
$$

Si el atractor se auto-interseca o pliega ($\Omega_{\text{FNN}} \ge \tau_{\text{FNN}}$), el término indicador emite $\mathbb{I} = 0$, suspendiendo la ejecución antes de la bifurcación.

---

## 6. Filtros de Estado Incremental O(d²)

> **Espacio Matemático:** Cono riemanniano de matrices simétricas definidas positivas $\mathcal{S}_{++}^d$.  
> **Ubicación en Código:** [`module1_ingestion.py:L128-L180`](infrastructure/ml/engines/rosa_roja/algorithms/modules/module1_ingestion.py#L128-L180).

Para garantizar latencia ultra-baja en tiempo real sobre flujos de señales de alta frecuencia, la matriz de covarianza inversa $\Sigma^{-1}$ se actualiza sin inversión matricial cúbica mediante el **Algoritmo de Sherman-Morrison**:

$$
\Sigma_n^{-1} = \frac{1}{c} \left( \Sigma_{n-1}^{-1} - \frac{\mathbf{z}_n \mathbf{z}_n^T}{c + \mathbf{u}_n^T \mathbf{z}_n} \right)
$$

Estabilizado periódicamente mediante re-anclaje por descomposición de Cholesky.

---

## 7. Arquitectura de Desacoplamiento y Modos de Ejecución

ZENIN aísla el núcleo matemático variacional de cualquier protocolo de transporte o actuador externo mediante una **arquitectura hexagonal estricta (Puertos y Adaptadores)**.

### Matriz de Especificación del Núcleo

| Dimensión | Especificación del Núcleo |
| :--- | :--- |
| **Naturaleza del Sistema** | Motor determinista y continuo, 100% agnóstico a cualquier aplicación o dominio. |
| **Espacio de Fases** | Flujo continuo sobre variedad Riemanniana $\mathcal{M} \subset \mathbb{R}^3 \times \mathbb{R}$ acoplado a la Fibración de Hopf en $\mathbb{C}^2 \to S^2$. |
| **Dualidad Canónica** | **Rosa Roja** (Cinemática Directa $+x$, Polo $z_1$) $\longleftrightarrow$ **MRT** (Cinemática Inversa $-x$, Polo $z_2$). |
| **Principio de Gobernanza** | Estabilidad topológica del atractor, coherencia de fase $U(1)$ y conservación de energía informacional $S_0$. |
| **Arquitectura de Ejecución** | Hexagonal estricta: desacoplamiento absoluto de puertos de entrada, procesamiento y salida. |

### Matriz de Desacoplamiento Hexagonal (Puertos y Flujo)

| Nivel Arquitectónico | Componentes y Motores | Contrato de Datos | Responsabilidad Operativa |
| :--- | :--- | :--- | :--- |
| **1. Inbound (Puerto de Entrada)** | Ingestores de Señales, Filtros Mahalanobis | Transición $\Delta S \in \mathbb{R}^d, \Delta t \gt 0$ | Normaliza observaciones multivariadas del entorno sin conocimiento del modelo interno. |
| **2. Kernel Soberano (Hexágono)** | Rosa Roja (Forward) + MRT (Conjugado), Hopf $\mathbb{C}^2$ | Espinor $|\psi\rangle \in \mathbb{C}^2 \to S^2$ | Computa trayectorias duales, evalúa $C_{\text{sovereign}} = S_3 + S_1$ y emite la directiva formal. |
| **3. Outbound (Puerto de Salida)** | Adaptadores de Control y Emisión de Acción | `ExecutionPlan` (Acción, $S_{\text{target}}$, Certeza) | Traduce la acción (`EXECUTE`, `HOLD`, `FLUSH`) y magnitud $S_{\text{target}}$ a señales de salida. |

### Banderas de Gobernanza y Modos Sombra (Inventario Consolidado)

| Bandera | Módulo / Ubicación | Default | Efecto Operativo Agnóstico en Producción |
| :--- | :--- | :--- | :--- |
| `master_shadow_mode` | `bot_config.py` / `orchestrator.py` | `False` | Gobierna si la Ecuación Maestra del orquestador toma el control soberano activo (`False`) o si opera en registro pasivo auditando contra la lógica legacy (`True`). |
| `manifold_shadow_mode` | `bot_config.py` / `master_equation.py` | `True` | Ejecuta el Motor de Variedad Geométrica Riemanniana (Divergencia de Liouville 4D y curvatura en $\mathcal{M}$) en paralelo para telemetría diagnóstica; no atenúa certeza ni altera la magnitud del actuador hasta su validación en producción. |
| `dual_engine_shadow_mode` | `bot_config.py` / `master_equation.py` | `True` | Ejecuta el acoplamiento dual Rosa Roja ($z_1$) + MRT ($z_2$) vía Fibración de Hopf en $\mathbb{C}^2 \to S^2$ registrando el espinor, parámetros de Stokes ($S_0, S_1, S_3$) y certeza soberana $C_{\text{sovereign}}$ en telemetría; preserva la certeza y magnitud nominales de Rosa Roja sin aplicar veto ni modulación activa hasta su aprobación institucional. |

---

## 8. Mapa de Trazabilidad: Espacios Matemáticos y Código Fuente

| Ecuación / Dinámica Central | Espacio Matemático | Módulo / Archivo Fuente | Líneas Clave |
| :--- | :--- | :--- | :--- |
| **Espinor Cuántico e Invariante de Stokes** | Espacio de Hilbert $\mathcal{H} \cong \mathbb{C}^2$, $\pi: \mathbb{C}^2 \to S^2 \subset \mathbb{R}^3$ | [`hopf_spinor_state.py`](domain/entities/manifold/hopf_spinor_state.py) | [L17–L62](domain/entities/manifold/hopf_spinor_state.py#L17-L62) |
| **Proyección Racional de Hopf** | Fibración de Hopf $\pi: S^3 \to S^2$ | [`mrt_hopf_fibration.py`](domain/services/manifold/mrt_hopf_fibration.py) | [L42–L130](domain/services/manifold/mrt_hopf_fibration.py#L42-L130) |
| **Variable Destino $\mathcal{D}(t)$ y Ecuación Soberana** | Fibrado Compuesto $\mathcal{M} \times S^2 \times \mathbb{R}$ | [`master_equation.py`](infrastructure/ml/master_engine/master_equation.py) | [L87–L172](infrastructure/ml/master_engine/master_equation.py#L87-L172) |
| **Motor Rosa Roja y MoE Gating** | Fibrado Tangente $T\mathcal{M}$ (Cinemática Directa $+x$) | [`engine.py`](infrastructure/ml/engines/rosa_roja/algorithms/engine.py) | [L27–L163](infrastructure/ml/engines/rosa_roja/algorithms/engine.py#L27-L163) |
| **Generador de Trayectorias Rítmicas** | Espacio de Fases $T\mathcal{M}$ y Fase $U(1)$ | [`rhythm_generator.py`](infrastructure/ml/engines/rosa_roja/algorithms/modules/rhythm_generator.py) | [L18–L90](infrastructure/ml/engines/rosa_roja/algorithms/modules/rhythm_generator.py#L18-L90) |
| **Vorticidad Rotacional de Maxwell** | Espacio de Fases $(x, v, a) \in \mathbb{R}^3$, Álgebra $\mathfrak{so}(3)$ | [`maxwell_curl_field.py`](infrastructure/ml/engines/mrt/algorithms/modules/maxwell_curl_field.py) | [L16–L74](infrastructure/ml/engines/mrt/algorithms/modules/maxwell_curl_field.py#L16-L74) |
| **Cristal Simpléctico de Ramanujan** | Espacio Extendido 4D $\mathbb{R}^4$ | [`ramanujan_crystal.py`](infrastructure/ml/engines/mrt/algorithms/modules/ramanujan_crystal.py) | [L16–L89](infrastructure/ml/engines/mrt/algorithms/modules/ramanujan_crystal.py#L16-L89) |
| **Transporte Disipativo de Fase $U(1)$** | Grupo de Norma Abeliano $U(1) \cong S^1$ | [`phase_conjugator.py`](infrastructure/ml/engines/mrt/phase_conjugator.py) | [L46–L107](infrastructure/ml/engines/mrt/phase_conjugator.py#L46-L107) |
| **Amplitud Racional Polo $z_2$** | Fibración Cuántica $\mathbb{C}$ | [`hopf_spinor_field.py`](infrastructure/ml/engines/mrt/algorithms/modules/hopf_spinor_field.py) | [L27–L107](infrastructure/ml/engines/mrt/algorithms/modules/hopf_spinor_field.py#L27-L107) |
| **Vector de Rebote Elástico Conjugado** | Espacio Tangente $T\mathcal{M}$ (Cinemática Inversa $-x$) | [`mrt_pipeline.py`](infrastructure/ml/engines/mrt/algorithms/mrt_pipeline.py) | [L62–L128](infrastructure/ml/engines/mrt/algorithms/mrt_pipeline.py#L62-L128) |
| **Orquestación MRT End-to-End** | Motor Dual Conjugado | [`mrt_engine.py`](infrastructure/ml/engines/mrt/mrt_engine.py) | [L61–L113](infrastructure/ml/engines/mrt/mrt_engine.py#L61-L113) |
| **Variedad Riemanniana y Jacobiano** | Variedad Continua $\mathcal{M} \subset \mathbb{R}^3$ | [`geometric_manifold_adapter.py`](infrastructure/ml/master_engine/geometric_manifold_adapter.py) | [L63–L139](infrastructure/ml/master_engine/geometric_manifold_adapter.py#L63-L139) |
| **Inversión Covarianza Sherman-Morrison** | Cono Simétrico $\mathcal{S}_{++}^d$ | [`module1_ingestion.py`](infrastructure/ml/engines/rosa_roja/algorithms/modules/module1_ingestion.py) | [L128–L180](infrastructure/ml/engines/rosa_roja/algorithms/modules/module1_ingestion.py#L128-L180) |

---

## 9. Certificación y Testing Institucional

Validación determinista bajo normas ISO/IEC 25010 y 22989:

```bash
# Validar algoritmos analíticos de Rosa Roja (Filtro Mahalanobis y Sherman-Morrison O(d²))
pytest -v tests/unit/rosa_roja/test_mahalanobis_sherman_morrison.py

# Validar algoritmos analíticos de MRT (Maxwell, Ramanujan, Hopf Spinor)
pytest -v tests/unit/infrastructure/engines/test_mrt_algorithms.py tests/unit/infrastructure/engines/test_mrt_engine.py

# Validar invarianza de la Fibración de Hopf y acoplamiento dual simétrico
pytest -v tests/unit/market/test_mrt_hopf_fibration.py

# Validar Variable Destino D(t), composición Liouville-MRT y desacoplamiento
pytest -v tests/unit/market/test_unified_destination_variable.py

# Validar invarianzas de Modo Sombra (35 ciclos exactos + Veto End-to-End)
pytest -v tests/unit/market/test_manifold_shadow_invariance.py tests/unit/market/test_dual_engine_shadow_invariance.py

# Validar motor topológico de Takens (Inmersión Espectral y Falsos Vecinos)
pytest -v tests/unit/market/test_takens_infra.py tests/unit/domain/test_takens_*.py

# Ejecutar suite completa institucional (700+ tests con cero fallos)
pytest -v tests/unit/market/ tests/unit/infrastructure/engines/
```
