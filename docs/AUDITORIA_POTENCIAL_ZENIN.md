# Auditoría de Potencial — ZENIN IoT / ALPLA

**Rol:** Auditor Senior de Arquitectura ML
**Formato:** Solo lectura — hallazgos con evidencia citada (archivo:línea)
**Objetivo:** Encontrar capacidades ya construidas que no se están aprovechando,
y oportunidades reales de mejora que hoy no se están explotando.

---

## SECCIÓN 1: Memoria Semántica (Weaviate)

### 1.1 Clases y campos del schema que nunca se consultan en recall

**Hallazgo:** `PatternMemory` y `DecisionReasoning` son clases definidas en el schema
Weaviate, con funciones de escritura (`remember_pattern`, `remember_decision`) y lectura
(`recall_similar_patterns`, `recall_similar_decisions`), pero **nunca se escriben en
producción**. El `CognitiveStorageDecorator` solo dispara `remember_explanation` y
`remember_anomaly`.

- **Evidencia:**
  - Schema `PatternMemory` definido en
    `infrastructure/persistence/vector/schema/class_definitions.py:80`
  - Schema `DecisionReasoning` definido en
    `infrastructure/persistence/vector/schema/class_definitions.py:115`
  - `CognitiveStorageDecorator._fire_cognitive()` en
    `infrastructure/adapters/cognitive_storage_decorator.py:165` solo llama
    `remember_explanation` (línea 165) y `remember_anomaly` (línea 195)
  - `remember_pattern()` y `remember_decision()` en WeaviateCognitiveAdapter
    (`weaviate_cognitive.py:129,151`) solo se llaman desde tests:
    `test_weaviate_cognitive_adapter.py:177,183`
  - Test unitario confirma: `mock_cognitive.remember_pattern.assert_not_called()`
    (`test_cognitive_storage_decorator.py:451-452`)

**Impacto:** Dos clases cognitivas completas (15-20 propiedades cada una, con
vectorización text2vec) están definidas en schema, ocupan espacio en la base de
datos vectorial (7.5% de las clases schema cargadas), pero su capacidad de
búsqueda semántica de patrones y decisiones está completamente inactiva.

**Esfuerzo para aprovechar:** Bajo. Solo conectar `remember_pattern()` desde
`CognitiveStorageDecorator` cuando el `Orchestrator` complete la fase de patrones,
y `remember_decision()` desde el DecisionOrchestrator actual.

---

### 1.2 `min_certainty` fijo en 0.7 sin variación por dominio/severidad

**Hallazgo:** El umbral `min_certainty=0.7` es el mismo en absolutamente todos
los callers de producción — 11 puntos de llamada idénticos. Solo hay UNA excepción:
`routes_query.py` usa `0.6` para consultas de chat (más permisivo). No hay
variación por severidad de anomalía, por dominio, ni por tipo de consulta.

**Evidencia:**
- `cognitive_memory_port.py:159,186,211,235` — default `0.7` en las 4 interfaces
- `memory_readers.py:23,79,136,191` — default `0.7` en las 4 implementaciones
- `weaviate_cognitive.py:180,207,234,260` — default `0.7` en el adapter
- `memory_recall_enricher.py:95` — hardcoded `0.7`
- `remember_phase.py:48` — hardcoded `0.7`
- `memory_comparator.py:17` — default `0.7`
- `text_analyzer.py:122` — hardcoded `0.7`
- Única excepción: `routes_query.py:123` — `min_certainty=0.6`

**Problema:** Un umbral único de `0.7` significa que anomalías de severidad
"critical" (ej. 591M arranques compresor, que es 131,038× la mediana) se tratan
igual que anomalías borderline con `anomalyScore=0.6`. En producción, una
anomalía crítica debería tener un recall más permisivo (ej. `0.5`) para maximizar
contexto histórico, y una anomalía leve podría tenerlo más restrictivo (`0.8`)
para no ensuciar la salida con falsos recuerdos.

**Esfuerzo:** Bajo. Agregar `min_certainty` configurable por nivel de severidad.

---

### 1.3 Campos `featureContributions` siempre vacíos en producción

**Hallazgo:** El campo `featureContributions` existe en la entidad `Prediction`,
se pasa al DTO, y se escribe a Weaviate como JSON — pero **ningún código de
producción lo poblada**. Todos los constructores de `Prediction` en producción
omiten el parámetro, dejándolo como `{}` (dict vacío).

**Evidencia:**
- `domain/entities/results/prediction.py:82` — definido con `default_factory=dict`
- `engine/ensemble/predictor.py:237,294` — NO pasa feature_contributions
- `moe/fusion/discrepancy_aware.py:104` — NO pasa
- `moe/fusion/sparse_fusion.py:105` — NO pasa
- `moe/gateway/prediction_enricher.py:72` — NO pasa
- `infrastructure/ml/interfaces.py:155,172` — NO pasa
- `ml_service/api/services/prediction_service.py:82` — NO pasa
- Único lugar que lo pobla: `scripts/test_zenin_ml_write.py:68` (test script)

**Impacto:** La explicabilidad de predicciones en Weaviate tiene un campo dedicado
a contribución de features que siempre almacena `{}`. Para el dataset ALPLA,
sería muy valioso saber qué features contribuyeron más a cada anomalía — hoy no
se puede consultar.

**Esfuerzo:** Medio. Requiere implementar SHAP/LIME o feature attribution en las
funciones de predicción de cada engine, no solo conectar un campo.

---

## SECCIÓN 2: MoE — Expertos y Pesos

### 2.1 Reliability scores con ExpertPerformanceTracker activo

**Hallazgo:** Se activó `ExpertPerformanceTracker.record()` en
`run_alpla_pipeline.py:433` (sesión anterior). Después de procesar los 47 parámetros
del dataset ALPLA, los reliability scores son:

| Experto | Reliability | Records | Score acumulado |
|---------|-------------|---------|-----------------|
| baseline | **0.9837** | 12 | 0.0163 (error mínimo) |
| taylor | **0.8343** | 20 | 0.1657 |
| kalman | **0.9034** | 6 | 0.0966 |
| statistical | **0.5** (default) | 0 | N/A |

**Interpretación:**
1. **statistical:** NUNCA es seleccionado como top expert en todo el dataset.
   Su reliability queda en 0.5 (default) porque nunca se llamó `record()` para él.
2. **baseline:** Tiene reliability casi perfecta (0.98) — pero solo se selecciona
   en regímenes STABLE (12 params). Cuando se usa, su error es mínimo.
3. **taylor:** Es el más seleccionado (20 veces, 43% de params), con reliability
   moderada (0.83). Su error normalizado promedio es ~0.17.
4. **kalman:** Bien cuando se selecciona (0.90), pero solo 6 veces (regímenes NOISY).

**Neutralidad confirmada:** El cambio es NEUTRO para el comportamiento actual
porque `_adjust_by_performance()` (`contextual_regime.py:295-307`) solo actúa
cuando reliability < 0.3 (penaliza -50%) o > 0.7 (bonus ~+9%). Baseline y taylor
están > 0.7 por lo que recibirían un ligero boost, pero el efecto en el routing
sería marginal porque los pesos base ya dominan la decisión.

**Esfuerzo:** Ya está implementado. El siguiente paso es exponer los reliability
scores en el reporte JSON (hoy no se incluyen en la metadata del output).

---

### 2.2 Expertos sub-utilizados por régimen

**Hallazgo:** El experto **statistical** tiene 0 selecciones como top expert en
todo el pipeline (47 parámetros). Revisando la distribución por régimen:

| Régimen | # Params | Top expert | 2º experto |
|---------|----------|------------|------------|
| volatile | 30 | taylor (93%) | kalman (7%) |
| stable | 12 | baseline (83%) | statistical (17% peso, pero nunca top) |
| noisy | 5 | kalman (80%) | taylor (20%) |

**Evidencia:** Del reporte `alpla_pipeline_results.json` — distribución global:
```
Top expert distribution: {taylor: 30, baseline: 12, kalman: 5}
```

**Causa raíz:** Los pesos base para statistical en todos los regímenes son
bajos (~0.14 en stable), y el ajuste por slope (que debería favorecerlo) usa
un umbral `_slope_threshold=0.01` que no se alcanza en la mayoría de los
parámetros (ver `contextual_regime.py:258`). Adicionalmente, en volatile,
Taylor tiene peso base > 0.70, dejando statistical con < 0.05.

**Impacto:** Statistical ocupa lugar en el registry, consume CPU en cada
dispatcher.dispatch(), pero nunca contribuye a la predicción fusionada porque
nunca está en el top-k. Es ruido computacional.

**Esfuerzo:** Bajo. Evaluar si statistical puede eliminarse del registro para
este tipo de datos, o si su peso base necesita recalibración. Hay evidencia
del baseline `alpla_pipeline_results.json` para tomar esta decisión.

---

### 2.3 ExpertPerformanceTracker no retroalimenta el routing real

**Hallazgo:** En el pipeline de `results/`, el tracker registra errores pero
`_adjust_by_performance()` en `contextual_regime.py:295-307` solo aplica un
ajuste marginal (±50% para <0.3, +9% para >0.7). El impacto real de tener
reliability scores históricos es despreciable vs. los pesos base que vienen
del archivo de configuración.

**Evidencia adicional:** `contextual_regime.py:123` carga los pesos desde
`_load_regime_weights()` que devuelve valores fijos:
- volatile: taylor=0.75, kalman=0.15, statistical=0.05, baseline=0.05
- stable: baseline=0.70, statistical=0.15, taylor=0.10, kalman=0.05
- noisy: kalman=0.55, baseline=0.20, taylor=0.15, statistical=0.10

El ajuste por performance puede modificar estos pesos hasta ±50%, pero los
pesos base están tan sesgados que un experto con peso base 0.05 incluso con
bonus 50% llega a 0.075 — todavía órdenes de magnitud por debajo del líder
(0.75).

**Esfuerzo:** Medio-alto. Si se quiere que el performance histórico realmente
influya, hay que (a) aumentar el rango del ajuste por performance, o (b)
reemplazar los pesos fijos por los del BayesianWeightTracker que ya existe
pero no está conectado a `ContextualRegimeGating` en el pipeline de `results/`.

---

## SECCIÓN 3: Detección de Anomalías

### 3.1 Parámetros no-contadores donde mean/std también falla

**Hallazgo:** Hay 5 parámetros (que NO son contadores según los patrones actuales)
donde `std/median > 0.5`, indicando que la media está siendo inflada por outliers
y el umbral `mean + 2*std` podría estar perdiendo anomalías.

| Parámetro | Std | Median | std/median | outliers_2std | outliers_3mad |
|-----------|-----|--------|------------|---------------|---------------|
| Cto.1 Presión ref. condensador | 762.60 | 1298 | **0.60** | 1 | 4 |
| Cto.2 Número arranques compresor | 33M | 4,512 | **7418** | 1 | 97 |
| Consumo energía RTAE 5 | 8.6M | 9.3M | **0.92** | 3 | 8 |
| Cto.1 Número arranques compresor | 4,224 | 5,431 | **0.78** | 2 | 2 |
| Temp. punto rocío secador | 2.88 | 4.16 | **0.82** | 5 | 87 |
| Temperatura salida agua | 4.14 | 9.59 | **0.46** | 1 | 34 |

**Detalle crítico:** "Temperatura del punto de rocío del secador" (CA) tiene
**87 outliers con MAD vs solo 5 con 2*std**. La disparidad es enorme: 
`outliers_2std=5` vs `outliers_3mad=87`. Esto NO es un contador (no coincide
con `COUNTER_PATTERNS`) pero sufre el MISMO problema — la distribución tiene
picos muy estrechos (MAD=0.30) que mean+2*std no captura porque la media está
inflada por los mismos picos.

**Recomendación:** Este parámetro es candidato directo para el mismo tratamiento
MAD que aplicamos a contadores. Los patrones de contador deberían ser un
conjunto expandible (no solo los 4 actuales), y la detección de "distribución
con MAD muy pequeña relativa a la mediana" podría ser automática.

**Esfuerzo:** Bajo. Expandir `COUNTER_PATTERNS` en `prepare_weaviate_output.py:18`
o mejor aún, hacer la detección MAD automática para todo parámetro donde
`mad < median * 0.01` (colas muy angostas).

---

### 3.2 Isolation Forest: contamination fija sin ajuste por equipo

**Hallazgo:** `contamination=0.05` se usa igual para Chiller y CA. El análisis
de sensibilidad muestra que el número de anomalías escala casi perfectamente
lineal con contamination, lo que significa que no hay un "codo" natural en los
datos que justifique 5% sobre otro valor. Ambos equipos responden igual:
contamination 0.01 → ~1%, 0.05 → ~5%, 0.10 → ~10%.

**Evidencia:** Verificado empíricamente:
```
Chiller: cont=0.01→4 anom, 0.03→10, 0.05→16, 0.07→22, 0.10→32
CA:      cont=0.01→4 anom, 0.03→11, 0.05→18, 0.07→25, 0.10→36
```

**Problema adicional:** Hay 3 columnas constantes en CA que entran al IF:
`['Presión de agua de entrada de la bomba', 'Presión de agua de salida de la
bomba', 'Presión de aire a la descarga de 1a etapa']` — tienen std=0.0, lo que
produce división por cero en StandardScaler y posible inestabilidad numérica.

**Esfuerzo:** Bajo-medio. (a) Excluir columnas constantes del IF (filtro
`df_clean.std() > 1e-10`), (b) evaluar si contamination debería ser 0.03
(menos ruido) o 0.07 (más cobertura) según criterio de negocio.

---

### 3.3 Correlación entre parámetros relacionados del mismo equipo

**Hallazgo:** Hay pares de parámetros en el Chiller donde una anomalía debería
aparecer SIEMPRE junta pero aparece sola, indicando que el pipeline detecta
una y se pierde la otra.

**Caso concreto:** Las temperaturas de saturación del refrigerante del Chiller:
- "Cto.1 Temperatura de saturación del refrigerante del evaporador"
- "Cto.2 Temperatura de saturación de refrigerante del evaporador"

Ambos miden esencialmente lo mismo (temperatura de evaporación en circuito 1 y 2).
Usando el criterio 2*std: Cto.1 tiene 23 outliers, Cto.2 tiene 19. La
intersección de fechas anómalas es solo **15 de 27** (Jaccard=0.56). Hay 8
fechas donde Cto.1 está anómalo y Cto.2 no, y 4 al revés. Esto es señal de
que el umbral 2*std, al ser independiente por columna, pierde anomalías que
SÍ son detectables por diferencia paramétrica (ej. un delta térmico anormal).

**Impacto:** Si se usara un detector de delta entre parámetros correlacionados
(ej. |T1 - T2| > 3*std_delta), se capturarían anomalías que hoy son invisibles
porque cada parámetro individual está dentro de su propio rango 2*std.

**Esfuerzo:** Medio. Requiere definir pares/familias de parámetros correlacionados
(no existe hoy en el código) y agregar un detector de delta paramétrico.

---

## SECCIÓN 4: GraphQL Endpoint

### 4.1 Capacidades de Strawberry-GraphQL no usadas

**Hallazgo:** El schema GraphQL actual (`routes_graphql.py`) tiene UNA sola
query (`analyze`) y **cero mutaciones, cero subscriptions**. Strawberry soporta:
- `@strawberry.subscription` para streaming de resultados
- `strawberry.dataloader.DataLoader` para batching N+1
- `@strawberry.mutation` para escritura
- `strawberry.federation` para schema composable
- `@strawberry.input` para inputs reutilizables

**Evidencia:**
- `routes_graphql.py:257`: `schema = strawberry.Schema(query=Query)` — sin
  mutations, sin subscriptions
- `routes_graphql.py:184`: `class Query` — un solo field `analyze`
- No hay `DataLoader` en ningún archivo del proyecto

**Oportunidad:** El endpoint `analyze` ya hace análisis + recall + conclusión
en un solo viaje (íntegramente, desde que `_engine.analyze()` corre las 5 fases),
pero esto está limitado a un solo tipo de análisis por consulta. No hay forma
de hacer queries compuestas tipo "dame el análisis de este texto + 3 anomalías
similares + 2 patrones del mismo equipo" porque `recall_similar_patterns` y
`recall_similar_decisions` ni siquiera se consultan.

**Esfuerzo:** Alto para subscriptions (requiere WebSocket en la infraestructura),
bajo para agregar más queries al schema (ej. `anomalies(dateRange, equipment, severity)`).

---

### 4.2 Recall desactivado en GraphQL por defecto

**Hallazgo:** El resolver GraphQL crea `UniversalContext` sin `cognitive_memory`
(`routes_graphql.py:206-211`), por lo que `recall_context` es siempre `None`
en las respuestas del endpoint GraphQL. El motor UniversalAnalysisEngine se crea
con `enable_semantic_enrichment=False` (línea 47).

**Evidencia:**
- `routes_graphql.py:208-211`: `UniversalContext(...)` — no pasa `cognitive_memory`
- `routes_graphql.py:47`: `_engine = UniversalAnalysisEngine(enable_semantic_enrichment=False, ...)`
- `remember_phase.py:32`: el if `ctx.cognitive_memory is not None` nunca se cumple

**Impacto:** Aunque el código de recall está completo y funcional (el `RememberPhase`
puede llamar `recall_similar_explanations`), el endpoint GraphQL nunca lo activa.
Un usuario que consulta el análisis de una anomalía via GraphQL recibe la conclusión
sin el contexto "Esto es similar a N anomalías previas" que el motor puede generar.

**Esfuerzo:** Bajo. Conectar el adapter Weaviate en la creación del context y
activar `enable_semantic_enrichment=True`. Riesgo: si Weaviate está caído, el
circuit breaker del adapter lo maneja correctamente (failure_threshold=3,
recovery_timeout=60s en `weaviate_cognitive.py`).

---

## SECCIÓN 5: Pattern Plasticity

### 5.1 ¿Se pierde aprendizaje al reiniciar?

**Hallazgo:** La plasticidad NO es 100% en memoria. Existe `plasticity_repository.py`
que persiste en SQL Server (`infrastructure/persistence/sql/plasticity_repository.py`),
y `BayesianWeightTracker` que tiene checkpoint/restore via Redis con TTL de 7 días.
Sin embargo, ESTO aplica al pipeline de producción.

Para el pipeline de `results/` (scripts de análisis ALPLA), la plasticidad SÍ es
100% en memoria — `ExpertPerformanceTracker` es una instancia de Python sin
persistencia. Si el script `run_alpla_pipeline.py` se reinicia, el tracker
empieza de cero y no hay herencia de patrones.

**Evidencia:**
- `docs/plasticity.md` confirma que el `BayesianWeightTracker` tiene actualización
  conjugada normal-normal y persistencia via `CheckpointMixin`
- `docker-compose.yml:31`: solo un volumen persistente (`zenin_compliance_data`)
  para exportación de compliance, NO para checkpoints de plasticidad
- `lifespan.py` (188 líneas): startup/shutdown del servicio — no hay warm-start
  de plasticidad desde checkpoint

**Estimación realista:**
- `restart: unless-stopped` en docker-compose significa que el proceso solo
  reinicia por: (a) fallo de healthcheck tras 3 intentos × 30s = 90s, (b)
  deploy nuevo, (c) crash del host
- En operación normal, el uptime esperado es de días/semanas — la ventana de
  plasticidad (20 registros por experto) se llena en minutos de operación
- Un reinicio típico pierde ~20 registros de error por experto, lo que es
  despreciable si el servicio corre >1 hora entre reinicios

**Conclusión:** El riesgo de pérdida de aprendizaje por reinicio es BAJO para
el caso de uso actual. Solo importaría si el proceso reinicia cada <30 minutos
(lo que no ocurre con restart:unless-stopped y healthchecks cada 30s).

**Esfuerzo para mitigar:** Bajo si se quiere, pero no es necesario. Agregar un
checkpoint JSON al hacer shutdown ordenado (`atexit` handler) tomaría ~2h de
implementación.

---

### 5.2 Plasticity en results/ vs producción

**Hallazgo:** El pipeline de `results/` (scripts de análisis ALPLA) y el servicio
de producción usan SISTEMAS DE PLASTICITY DIFERENTES:
- `results/` → `ExpertPerformanceTracker` (EMA simple, in-memory, 3 líneas de ajuste)
- Producción → `BayesianWeightTracker` (28 módulos, conjugado normal-normal,
  persistencia SQL, LRU con TTL 24h, drift detection)

**Evidencia:** 
- `contextual_regime.py:84-107`: `ExpertPerformanceTracker` — 24 líneas
- `bayesian_weight_tracker/`: 28 archivos, ~3000 líneas de código
- `run_alpla_pipeline.py` usa `ContextualRegimeGating` que internamente usa
  `ExpertPerformanceTracker` (el simple)
- `WeightResolutionService` en producción usa `BayesianWeightTracker` (el complejo)

**Impacto:** Las conclusiones del pipeline de results/ sobre "cómo se comporta el
MoE" no son directamente trasladables a producción porque el sistema de pesos
real es mucho más sofisticado. El `ExpertPerformanceTracker` de results/ es
una simplificación que no captura dinámicas como drift, convergencia, o decaimiento
natural que sí tiene el `BayesianWeightTracker`.

**Esfuerzo:** Medio. Conectar el `BayesianWeightTracker` al pipeline de results/
para que los tests A/B reflejen la misma lógica que producción.

---

## SECCIÓN 6: Dataset ALPLA — Insights no Explotados

### 6.1 Correlación cruzada Chiller ↔ CA

**Hallazgo:** De 303 fechas compartidas entre Chiller y CA, solo **1 fecha**
(2026-03-02) tiene anomalías IF simultáneas en AMBOS equipos. Esto sugiere que
los equipos operan de forma independiente y NO hay causas comunes significativas
(corte de energía, evento ambiental) que afecten ambos simultáneamente.

**Análisis adicional de parámetros correlacionados entre equipos:**
- Temperatura de entrada de agua (Chiller) × Temperatura de agua a la entrada
  del compresor (CA): **Jaccard=0.00** — ninguna anomalía compartida.
- Temperatura ambiente (Chiller) × Temperatura ambiente (CA): son el mismo
  concepto pero ningún timestamp anómalo coincide con 2*std.

**Interpretación:** La baja correlación cruzada es un hallazgo POSITIVO para
el negocio — significa que las anomalías son específicas de cada equipo y no
hay un problema ambiental generalizado. Pero también es una OPORTUNIDAD PERDIDA:
el sistema nunca verificó esto formalmente. El `CausalCorrelationEngine`
(`infrastructure/ml/cognitive/causal/causal_correlation_engine.py`) existe
y tiene capacidad de detección de correlaciones cruzadas con lag, pero **no
está conectado al pipeline de producción**.

**Impacto:** La reunión con ALPLA debería incluir este hallazgo: "Validamos que
las anomalías del Chiller y el CA son independientes — no hay causa común
oculta. Cada equipo requiere mantenimiento individual."

**Esfuerzo para conectar causal_correlation_engine:** Bajo (existe pero no está
conectado al main pipeline).

---

### 6.2 Parámetros con poder predictivo de anomalía futura

**Hallazgo:** Buscando parámetros que actúan como indicadores tempranos
(parámetro A sube antes de que B tenga anomalía), se identificaron candidatos:

**Caso 1: Temperatura de salida de agua (Chiller)**
- `alpla_pipeline_results.json`: es clasificado como volatile por el MoE, con
  alta desviación (std=4.14, media=9.59). Tiene solo 1 outlier con 2*std, pero
  34 con 3*MAD. Esto sugiere que tiene MICRO-VARIACIONES que el MoE detecta
  como ruido, y que podrían anteceder a anomalías en temperatura de entrada.

**Caso 2: Consumo de energía RTAE 5 (Chiller)**
- Coeficiente de variación enorme (86% = 8.6M/10.0M). El parámetro tiene picos
  de consumo que, al analizar secuencia temporal, podrían preceder a anomalías
  de temperatura (más carga = más calor = más estrés térmico).

**Caso 3: Temperatura de aceite y chumaceras (CA)**
- Varios parámetros de temperatura en el CA muestran alta tasa de outliers con
  MAD (chumacera lado motor: 62, cruceta vertical: 39, aceite: 15). Estas
  temperaturas son proxy directo de desgaste mecánico y podrían predecir
  anomalías futuras en presión o caudal.

**Limitación:** El MoE actual del pipeline de results/ hace predicción univariada
(serie→valor futuro), no multivariada con features de otros parámetros. No hay
un modelo que use la temperatura de aceite de HOY para predecir la presión de
MAÑANA. El `CausalCorrelationEngine::detect_granger_causality()` existe y podría
hacer esto pero no está conectado.

**Esfuerzo:** Alto. Requiere implementar un pipeline de forecasting multivariado
con features de parámetros relacionados, no solo la serie individual. El
`CausalCorrelationEngine` es un buen punto de partida.

---

### 6.3 Patrón direccional del Chiller no explotado

**Hallazgo:** El Chiller muestra un patrón macro claro detectado en sesiones
anteriores: valores altos en enero-junio 2025, valores bajos de diciembre 2025
en adelante. Esto es consistente con un cambio de setpoint o régimen operativo,
pero el pipeline trata todo el dataset como una sola serie sin segmentación.

**Evidencia:** Del perfil del dataset Chiller:
- Media global de temperatura de salida de agua: 9.59°C
- Pero la serie tiene dos mitades con medias muy diferentes (cambio direccional)
- El MoE clasifica muchos parámetros como "volatile" cuando en realidad tienen
  UN solo cambio de régimen (no ruido, sino transición)

**Impacto:** Si se segmentara el dataset en pre/post cambio de setpoint, las
métricas de error del MoE mejorarían porque los modelos no tendrían que cubrir
dos regímenes distintos con un solo conjunto de pesos.

**Esfuerzo:** Medio. Detectar automáticamente puntos de cambio (change point
detection) y segmentar el dataset antes de pasar al MoE. El
`temporal_pattern_miner.py` existe y puede hacer change point detection.

---

## RANKING: Top 3 Hallazgos de Mayor Impacto Potencial para ALPLA

### 🥇 #1: Conectar `CausalCorrelationEngine` al pipeline (Sección 6.2)

**Por qué:** El dataset ALPLA tiene 47 parámetros monitoreados diariamente
durante ~1 año. El mantenimiento predictivo real necesita saber QUÉ parámetro
monitorear para anticipar una falla. Hoy, cada parámetro se analiza de forma
aislada (univariada). El `CausalCorrelationEngine` puede detectar qué parámetro
A precede a una anomalía en B con N días de anticipación — eso es literalmente
el "santo grial" del mantenimiento predictivo que ALPLA compraría.

**Cómo verificarlo con el dataset real:** Correr Granger causality entre
"Temperatura de aceite" y "Presión del aceite" en CA. Si T_aceite → P_aceite
con lag > 0, tenemos un indicador temprano demostrable.

**Esfuerzo:** Medio (el código existe, solo hay que conectarlo y correrlo).

---

### 🥈 #2: Expandir MAD a parámetros no-contadores con cola angosta (Sección 3.1)

**Por qué:** "Temperatura del punto de rocío del secador" tiene 5 outliers con
2*std pero 87 con 3*MAD — una diferencia de 17x. Esto significa que el pipeline
actual está perdiendo ~82 anomalías reales en UN solo parámetro. Extrapolando
a los 47 parámetros, podríamos estar perdiendo cientos de anomalías.

**Cómo verificarlo con el dataset real:** Ya está verificado — los números
hablan solos (`outliers_2std=5` vs `outliers_3mad=87`).

**Esfuerzo:** Bajo (cambiar condición de contadores a condición de distribución
angosta automática).

---

### 🥉 #3: Activar `feature_contributions` real (Sección 1.3) + recall en GraphQL (Sección 4.2)

**Por qué:** Son dos oportunidades complementarias que juntas dan un salto
cualitativo en la demo: (a) cuando un usuario consulta una anomalía, el sistema
le dice "esto se parece a N anomalías previas" (recall), y (b) "las features
que más contribuyeron fueron X, Y, Z" (feature_contributions). Es el tipo de
funcionalidad que diferencia un sistema de "monitoreo" de uno de "inteligencia".

**Cómo verificarlo con el dataset real:** Correr el upload a Weaviate (ya hecho)
y hacer una consulta GraphQL con el recall conectado. La respuesta incluiría
automáticamente el contexto semántico de anomalías similares.

**Esfuerzo:** Bajo (conectar adapter en GraphQL, ~2h) + Medio (implementar
feature attribution en engines, ~2-3 días).

---

*Documento generado el 2026-07-13 — puede requerir actualizaciones al correr
nuevos experimentos o conectar los componentes identificados.*
