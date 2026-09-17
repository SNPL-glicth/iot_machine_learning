# Arquitectura ZENIN: Motor de Decisión y Metacognición

## Contexto del Proyecto

ZENIN nació para procesar datos de sensores de Internet de las Cosas (IoT). Como el mundo real está lleno de ruido, retrasos y fallos físicos, el sistema aprendió una regla vital: **nunca confiar a ciegas en los datos ni en sus propias predicciones**.

Hoy, el núcleo se ha convertido en un motor de decisiones universal. A diferencia de la Inteligencia Artificial tradicional, que asume que siempre tiene la razón y falla sin avisar cuando se confunde, ZENIN tiene *metacognición*: sabe cuándo no sabe. Antes de actuar (ya sea invirtiendo dinero en un mercado o moviendo un robot), calcula matemáticamente su nivel de duda, el riesgo de equivocarse y si el momento exacto es el adecuado para moverse.

---

## La Ecuación Maestra (ZENIN v2.2)

Esta ecuación es el cerebro del sistema. Su mayor logro es separar tres cosas que la IA normal suele mezclar en un solo número: **Qué hacer**, **Qué tan seguros estamos** y **Si el viento sopla a favor**.

### 1. Motor de Decisión (El Gatillo)
Determina la acción final que el sistema ejecutará en la realidad.

$$
\mathcal{O}_{\text{ZENIN}}(t) = \underbrace{ \left( \sum_{i=1}^{N} w_i \Psi_i(t) \right) }_{\text{Magnitud Objetivo}} \cdot \underbrace{ \Theta \Big( \Phi_{\text{RedRose}}(t) - \tau_{\text{exec}} \Big) }_{\text{Filtro de Umbral}} \cdot \underbrace{ \Theta \left( \overline{\frac{dy}{dt}} \cdot \sum_{i=1}^{N} w_i \Psi_i(t) - (\tau_{\text{mom}} \cdot \sigma_{\text{mom}}) \right) }_{\text{Filtro de Momentum}}
$$

En términos sencillos, el sistema multiplica tres factores:
* **El Objetivo (Magnitud):** Lo que los modelos recomiendan hacer (ej. "comprar 50 acciones").
* **El Permiso (Umbral):** Un interruptor de seguridad. Si el nivel de certeza del sistema no supera un umbral mínimo, este valor se vuelve cero y cancela la operación entera.
* **El Chequeo de Último Instante (Momentum):** Justo antes de actuar, mira la inercia de la realidad. Si el sistema quiere "comprar" pero en ese preciso milisegundo el precio se está desplomando bruscamente, bloquea la acción para no estrellarse.

### 2. Motor de Certeza Rosa Roja (El Escudo)
Esta fórmula es la que decide si el interruptor de "Permiso" del paso anterior se enciende o se apaga. Califica la seguridad del sistema en una escala de 0.0 (duda total) a 1.0 (certeza absoluta).

$$
\Phi_{\text{RedRose}}(t) = \underbrace{\mathbb{I}(\text{CVaR}_t \le L_{\text{max}}(t))}_{\text{Veto de Riesgo}} \cdot \underbrace{\exp \left( - \left| \frac{\big| \partial S / \partial t \big|}{\max\left(|\partial R / \partial t|, \epsilon \cdot \sigma_{\partial R / \partial t}\right)} - 1 \right| \right)}_{\text{Sincronía de Tiempo } \Lambda(t)} \cdot \underbrace{\Phi_{\text{epistémica}}(t)}_{\text{Confianza del Jurado}}
$$

Para que el sistema confíe en sí mismo, debe pasar tres filtros:
1. **Límite de Pérdida (Riesgo):** Calcula el peor escenario posible. Si la pérdida máxima esperada supera lo que el sistema tiene permitido arriesgar, la certeza cae a cero instantáneamente. La supervivencia va antes que la ganancia.
2. **Sincronía del Reloj (Tiempo):** Compara la velocidad a la que están ocurriendo las cosas frente a la velocidad a la que *deberían* ocurrir. Si el mercado o el entorno entra en pánico y se mueve demasiado rápido, el sistema detecta el caos y reduce drásticamente su nivel de confianza.
3. **El Debate Interno (Confianza del Jurado):** Consulta la opinión de sus propios modelos internos, detallada a continuación.

### 3. La Confianza del Jurado (Rosa Roja Base)
Aquí es donde el sistema evalúa a sus propios expertos internos.

$$
\Phi_{\text{epistémica}}(t) = \underbrace{\left[ \frac{\frac{\sum_{i=1}^{M} w_i s_i(t)}{\sum_{i=1}^M w_i}}{1 + \gamma \text{Var}(s_i)} \right]}_{\text{Consenso y Varianza}} \cdot \underbrace{\Big( 1 - \lambda_t (1 - \Phi_{\text{ritmo}}) \Big)}_{\text{Reconocimiento de Ignorancia}}
$$

El comportamiento se resume en dos reglas prácticas:
* **Penalización por desacuerdo:** El sistema promedia lo que opinan sus diferentes modelos matemáticos. Sin embargo, si los modelos se contradicen entre sí (uno dice sube, otro dice baja), se aplica un castigo severo a la confianza total. Solo se actúa si hay consenso.
* **Humildad algorítmica:** El sistema monitorea constantemente qué tan predecible está siendo el entorno. Si detecta que las reglas del juego están cambiando rápido, aumenta su índice de "ignorancia" y frena sus acciones hasta volver a entender el terreno.
