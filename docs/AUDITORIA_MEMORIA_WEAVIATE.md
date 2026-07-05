# Auditoría de Memoria Cognitiva Weaviate

> **Fecha**: 2026-07-05  
> **Propósito**: Determinar si Weaviate como memoria vectorial realmente mejora la
> precisión y las decisiones del sistema, o si es infraestructura sin impacto medible.  
> **Estado**: DOCUMENTO VIVO — se irá completando en sesiones futuras.

---

## 1. Schema actual en Weaviate

### 1.1 Clases definidas

Todas viven en `infrastructure/persistence/vector/schema/class_definitions.py`.

| Clase | Vectorizer | Propiedades | ¿Se escribe? |
|---|---|---|---|
| `MLExplanation` | `text2vec-transformers` | 14 props: seriesId, domainName, engineName, explanationText, trend, confidenceScore, confidenceLevel, predictedValue, horizonSteps, featureContributions, sourceRecordId, auditTraceId, createdAt, metadata | Solo vía `CognitiveStorageDecorator` (ver §2) |
| `AnomalyMemory` | `text2vec-transformers` | 16 props: seriesId, domainName, isAnomaly, anomalyScore, confidence, severity, explanationText, methodVotes, eventCode, behaviorPattern, operationalContext, sourceRecordId, relatedPredictionId, auditTraceId, createdAt, metadata | Misma vía decorator |
| `PatternMemory` | `text2vec-transformers` | 15 props: seriesId, domainName, patternType, confidence, descriptionText, changePointIndex, changeMagnitude, spikeClassification, regimeName, regimeMeanValue, persistenceScore, sourceRecordId, auditTraceId, createdAt, metadata | Solo tests |
| `DecisionReasoning` | `text2vec-transformers` | 19 props: deviceId, domainName, patternSignature, decisionType, priority, severity, titleText, summaryText, explanationText, recommendedActions, affectedSeriesIds, eventCount, confidenceScore, isRecurring, historicalResolutionRate, reasonTrace, sourceRecordId, auditTraceId, createdAt, metadata | Solo tests |

Todas usan `skip_vectorization=True` en la mayoría de campos. Solo
`explanationText`, `descriptionText`, `operationalContext`, `summaryText` y
`explanationText` (en DecisionReasoning) se vectorizan.

Las 4 clases tienen `vectorizer: text2vec-transformers` con
`vectorizeClassName: False` — Weaviate genera los embeddings **server-side**
al momento de insertar cada objeto.

### 1.2 Clases definidas pero nunca escritas

- **PatternMemory**: No hay llamadas a `remember_pattern` en el flujo productivo.
  Solo aparece en tests unitarios y de integración.
- **DecisionReasoning**: Ídem. Solo en tests.

`MLExplanation` tiene escritura real vía `CognitiveStorageDecorator.save_prediction`
→ `remember_explanation`. `AnomalyMemory` tiene escritura vía
`CognitiveStorageDecorator.save_anomaly_event` → `remember_anomaly`.

### 1.3 Ruta alternativa (legacy / muerta)

El módulo `infrastructure/ml/cognitive/memory/anomaly_memory_store.py` define
`AnomalyMemoryStore` que escribe a la clase `OperationalMemory` **que no existe
en el schema actual** (`class_definitions.py`). Este módulo usa embeddings MD5
dummy. No parece estar conectado a ningún flujo productivo — no hay imports
desde el pipeline actual. Probablemente es código sobrante de una iteración
anterior.

---

## 2. Qué se ESCRIBE en memoria (y con qué embeddings reales)

### 2.1 Ruta de escritura productiva

El camino real es:

```
save_prediction() / save_anomaly_event()
  → CognitiveStorageDecorator (SQL first, then cognitive dual-write)
    → WeaviateCognitiveAdapter.remember_*()
      → memory_writers.remember_*()
        → create_object()  [HTTP POST /v1/objects]
```

Archivos clave:
- `infrastructure/adapters/cognitive_storage_decorator.py:159-175` — dual-write
- `infrastructure/adapters/cognitive_storage_factory.py:53-65` — wiring DI
- `infrastructure/adapters/weaviate/memory_writers.py:17-287` — construye payloads
- `infrastructure/adapters/weaviate/object_operations.py:13-60` — HTTP POST

### 2.2 Embeddings: NO son MD5

**Las escrituras vía `memory_writers.py` NO envían vector explícito.**
Weaviate lo genera server-side usando `text2vec-transformers` porque el schema
declara `vectorizer: text2vec-transformers`.

Esto significa que las escrituras reales sí producen embeddings semánticos
válidos. El TODO de embeddings MD5 en `anomaly_memory_store.py:175-184`
es código muerto (nunca se llama desde el pipeline productivo).

### 2.3 ¿Qué contenido se guarda realmente?

En `remember_explanation` se guarda:
- `seriesId`, `domainName`, `engineName`
- `explanationText` ← **campo vectorizado**
- `trend`, `confidenceScore`, `confidenceLevel`
- `predictedValue`, `horizonSteps`
- `featureContributions` (JSON)
- `sourceRecordId` (back-reference a SQL)

En `remember_anomaly` se guarda:
- `seriesId`, `domainName`, `isAnomaly`, `anomalyScore`, `confidence`
- `severity`, `explanationText` ← **campo vectorizado**
- `methodVotes`, `eventCode`, `behaviorPattern`, `operationalContext`
- `sourceRecordId`

### 2.4 ¿Cuándo se escriben realmente estos datos?

Depende de:
1. `ML_ENABLE_COGNITIVE_MEMORY=true`
2. `ML_COGNITIVE_MEMORY_URL` configurado (≠ vacío)
3. Que `CognitiveStorageDecorator` esté activo (se crea en `build_storage`)

Si alguna condición falla, se usa `NullCognitiveAdapter` — no escribe nada
(ver `cognitive_storage_factory.py:53-63`).

---

## 3. Qué se LEE de memoria (recall)

### 3.1 Ruta de lectura productiva

Hay **dos caminos de lectura completamente separados**:

#### A) Vía `CognitiveMemoryPort.recall_similar_explanations` (UniversalAnalysisEngine)

```
UniversalAnalysisEngine.analyze()
  → RememberPhase.execute()
    → ctx.cognitive_memory.recall_similar_explanations(query, series_id, limit=3, min_certainty=0.7)
      → WeaviateCognitiveAdapter.recall_similar_explanations()
        → memory_readers.recall_similar_explanations()
          → query_operations.graphql_near_text()
            → HTTP POST /v1/graphql  [nearText con certainty=0.7]
```

Archivos: `engine.py:151`, `remember_phase.py:39-56`, `memory_readers.py:16-68`,
`query_operations.py:14-86`.

**Parámetros**:
- `limit=3` (en `remember_phase.py`)
- `min_certainty=0.7`
- Filtro opcional por `series_id`
- Busca en clase `MLExplanation`
- Retorna `list[MemorySearchResult]`

#### B) Vía `recall_similar_documents` (routes_query.py)

```
ml_query()
  → _safe_recall()
    → text_recall.recall_similar_documents()
      → HTTP POST /v1/graphql  [nearText con certainty, limit]
```

Archivos: `routes_query.py:111-127`, `text_recall.py:57-119`.

**Parámetros**:
- `limit=5` (hardcode en `routes_query.py:122`)
- `min_certainty=0.6` (más permisivo que el A)
- Filtro fijo `domainName="zenin_docs"`
- Busca en clase `MLExplanation`

#### C) Vía `routes_cognitive.py` (endpoint REST)

```
semantic_search()
  → WeaviateCognitiveAdapter.recall_similar_documents()
    → weaviate_cognitive_adapter._build_graphql_query()
      → HTTP POST /v1/graphql
```

Usa un adapter **diferente** (`weaviate_cognitive_adapter.py`). Parámetros
vienen del request HTTP.

### 3.2 ¿El `recall_context` en `UniversalResult` se llena en producción?

**NO.** La variable `recall_context` en `UniversalResult.recall_context`
(types.py:71) solo se llena cuando `ctx.cognitive_memory` no es `None`
(remember_phase.py:39).

El único lugar que inyecta `cognitive_memory` a `UniversalContext` es
`universal_bridge.py:215`. **Pero** el factory principal
(`document_analyzer_factory.py:165`) hardcodea `cognitive_memory=None`.
Por lo tanto:

- **Ruta productiva (DocumentAnalyzer → _EngineAdapter)**: `recall_context` siempre `None`.
- **Ruta routes_cognitive.py**: Tiene su propio adapter con Weaviate real, pero
  no llama a `UniversalAnalysisEngine`.
- **Ruta routes_query.py**: Llama a `TextCognitiveEngine` (no `UniversalAnalysisEngine`),
  y `TextCognitiveEngine` no tiene `recall_context` en su resultado.

**Conclusión: `recall_context` en `UniversalResult` nunca tiene datos reales
en el flujo productivo actual.**

---

## 4. Cómo el recall afecta las DECISIONES reales

### 4.1 Confidence bonus (`engine.py:357-358`)

```python
if has_recall:
    confidence += 0.05
```

- **Condición**: `has_recall` es `True` solo si `recall_ctx is not None`
- **Impacto máximo**: +0.05 (5 puntos porcentuales)
- **Rango total**: confidence se mueve entre `[0.10, 0.95]`
- **En producción**: `has_recall` siempre es `False` porque `cognitive_memory=None`

**Hallazgo**: El código para bonus por recall existe, pero nunca se ejecuta
en el flujo productivo actual. Incluso si se activara, el bonus es pequeño
(+0.05) comparado con el rango que ya aportan data quality (+0.15),
consensus factor (+0.40), e inhibition penalty (-0.15).

### 4.2 Severity

`engine.py:_classify_severity()` no usa memoria en absoluto. La severidad se
clasifica basada en fused_value + percepciones de los sub-analyzers. No hay
ningún `classify_text_severity` ni policy que ajuste el risk_level basado en
resultados de recall.

### 4.3 Conclusion

`build_semantic_conclusion` en `conclusion_builder.py:57` recibe `recall_results`
como parámetro (línea 57). En las líneas 116-120, **sí los usa**:

```python
if recall_results:
    parts.append(
        f"Semantic recall: {len(recall_results)} similar documents "
        f"(max score {max(r.score for r in recall_results):.3f})."
    )
```

Pero esta función (`build_semantic_conclusion`) es la función "legacy" usada por
`text_analyzer.py`, NO por la ruta moderna (`build_conclusion` →
`format_conclusion` en `conclusion_formatter.py`).

La ruta moderna (`format_conclusion` en `domain/services/cognitive/conclusion_formatter.py`)
no recibe `recall_results`. Solo recibe `comparison_result`.

**Hallazgo**: El parámetro `recall_results` en `build_semantic_conclusion` es
semi-fantasma: el código lo procesa, pero ningún caller actual lo invoca con
datos reales.

### 4.4 Pattern plasticity

`PatternPlasticityTracker` en `pattern_plasticity.py` es **100% en memoria**.
Así lo documenta explícitamente el propio archivo (líneas 14-17):

> In-memory only (no persistence) — resets on restart

El `ReasonPhase` (engine.py:201-207) actualiza plasticity tras cada análisis,
pero los pesos se pierden al reiniciar el proceso. No hay persistencia en
Weaviate ni en SQL para estos pesos.

- `plasticity_repository.py` existe con implementaciones in-memory y SQL, pero
  `ReasonPhase` no las usa. Usa directamente `BayesianWeightTracker`
  (import lazy en `reason_phase.py:16-18`).

---

## 5. Evidencia de impacto medible

### 5.1 Tests A/B con/sin memoria

**No existen.** No hay ningún test, log, o métrica que compare
confidence/severity con memoria vs. sin memoria para el mismo input.

Los tests existentes verifican:
- Que el `CognitiveStorageDecorator` llame a `remember_explanation` después de
  SQL (`test_cognitive_storage_decorator.py`)
- Que los adapters funcionen en modo dry-run (`test_weaviate_cognitive_adapter.py`)
- Que el `NullCognitiveAdapter` devuelva valores seguros

Pero ningún test mide si la calidad de la salida (confianza, severidad)
mejora cuando hay datos en Weaviate.

### 5.2 Datos reales en Weaviate

En el entorno de desarrollo actual, no hay forma de verificar cuántos
documentos existen en Weaviate sin conectarse al servicio. Sin embargo:

- Las escrituras requieren `ML_ENABLE_COGNITIVE_MEMORY=true` + URL configurada
  + decorator activo.
- Incluso con todo activo, las escrituras ocurren **solo** cuando se llama a
  `save_prediction` o `save_anomaly_event`.

Sin una conexión activa a Weaviate en este entorno, la memoria está vacía
y cualquier prueba de recall devolvería 0 resultados.

### 5.3 ¿Qué se necesita para medir impacto?

Se necesitaría:
1. Weaviate corriendo con datos históricos insertados
2. Un test que ejecute el mismo input dos veces: una con `cognitive_memory`
   poblada y otra con `None`
3. Comparar confidence, severity, y conclusion en ambos casos

Ninguno de estos elementos existe hoy.

---

## 6. Rutas que SÍ y NO usan memoria hoy

| Ruta | Usa memoria (cognitive\_memory) | Evidencia |
|---|---|---|
| **DocumentAnalyzer → _EngineAdapter** (flujo productivo principal) | **NO** — `cognitive_memory=None` hardcodeado | `document_analyzer_factory.py:165` |
| **routes\_query.py → ml\_query** | **NO (parcial)** — usa `text_recall.recall_similar_documents` para consultar Weaviate directamente, pero `TextCognitiveEngine` no recibe el `cognitive_memory` port | `routes_query.py:155` llama a `_safe_recall()` que es un HTTP directo a Weaviate, no vía el port |
| **routes\_cognitive.py → semantic\_search** | **SÍ** — crea `WeaviateCognitiveAdapter` con URL real | `routes_cognitive.py:49-52` |
| **routes\_cognitive.py → index-document** | **SÍ** — escribe a Weaviate | `routes_cognitive.py:85-91` |
| **routes\_graphql.py → analyze** | **NO** — `enable_semantic_enrichment=False` y `UniversalContext` no recibe `cognitive_memory` | `routes_graphql.py:46-49` (engine sin memoria), el resolver usa `analyze()` sin ctx.cognitive_memory |
| **UniversalComparativeEngine** | **SÍ condicional** — si `cognitive_memory` se pasa, intenta recall | `engine.py:43-44` en comparative: chequea `ctx.cognitive_memory` |

Nota sobre routes\_graphql.py: No es descuido — es decisión deliberada de
diseño. El engine se crea con `enable_plasticity=False`,
`enable_monte_carlo=False`, `enable_semantic_enrichment=False` y sin
cognitive\_memory porque el GraphQL endpoint es una vista simplificada
para clientes ligeros.

---

## 7. Hallazgos priorizados

Ordenados por impacto en la pregunta "¿la memoria realmente ayuda hoy?"
(no por facilidad de arreglo).

### 🔴 H1 — `cognitive_memory=None` en el flujo productivo principal

**Severidad**: CRÍTICO  
**Archivo**: `document_analyzer_factory.py:165`  
**Impacto**: El 100% del tráfico que pasa por `DocumentAnalyzer` (el camino
principal) inyecta `cognitive_memory=None` en `UniversalContext`. Esto significa
que `RememberPhase.execute()` (remember_phase.py:39) **nunca** ejecuta recall
en producción. Todo el pipeline de memoria está diseñado para recibir un port
pero el único caller real no lo pasa.

### 🔴 H2 — `recall_context` nunca se usa en `build_conclusion`

**Severidad**: CRÍTICO  
**Archivo**: `domain/services/cognitive/conclusion_formatter.py`  
**Impacto**: Incluso si H1 se corrigiera y `recall_ctx` se poblara, la función
`format_conclusion()` (que genera la conclusión visible al usuario) **no usa**
`recall_context` en absoluto. Solo usa `comparison_result`. La información
recuperada de Weaviate se perdería.

### 🟡 H3 — Sin métricas de impacto (esto es ciego)

**Severidad**: ALTO  
**Impacto**: No existe ningún test A/B, log estructurado, o dashboard que
compare la salida del sistema con memoria vs. sin memoria. Es imposible saber
si los embeddings de `text2vec-transformers` realmente mejoran algo porque
nunca se ha medido. La memoria podría estar generando ruido o resultados
irrelevantes sin que nadie lo note.

### 🟡 H4 — `PatternMemory` y `DecisionReasoning` nunca se escriben en producción

**Severidad**: MEDIO  
**Archivos**: `memory_writers.py:149-204` (remember\_pattern),
`memory_writers.py:207-287` (remember\_decision)  
**Impacto**: El schema define estas clases, el adapter tiene código para
escribirlas, pero ningún flujo productivo las invoca. Solo `remember_explanation`
y `remember_anomaly` tienen callers reales (via `CognitiveStorageDecorator`).

### 🟡 H5 — `build_semantic_conclusion` recibe `recall_results` como parámetro fantasma

**Severidad**: MEDIO  
**Archivo**: `infrastructure/ml/cognitive/text/conclusion_builder.py:57`  
**Impacto**: El parámetro `recall_results` existe y el código lo procesa
(líneas 116-120), pero ningún caller actual de `build_semantic_conclusion`
lo pasa con datos reales. La ruta moderna (`format_conclusion`) no usa
recall\_context en absoluto.

### 🟢 H6 — Pattern plasticity es in-memory (se pierde al reiniciar)

**Severidad**: BAJO  
**Archivo**: `pattern_plasticity.py:14-17`  
**Impacto**: Documentado explícitamente como decisión de diseño. Los pesos
de patrones se pierden al reiniciar el proceso. No afecta la corrección
del sistema, pero el "aprendizaje" de patrones es efímero.

### 🟢 H7 — AnomalyMemoryStore (MD5) es código muerto

**Severidad**: BAJO  
**Archivo**: `infrastructure/ml/cognitive/memory/anomaly_memory_store.py`  
**Impacto**: El TODO de embeddings MD5 (`_generate_embedding` línea 175-184)
está en un módulo que aparentemente no se usa en el flujo productivo. No
contamina las escrituras reales.

### 🟢 H8 — Confidence bonus (+0.05) es marginal

**Severidad**: BAJO  
**Archivo**: `engine.py:357-358`  
**Impacto**: Incluso si se corrigiera H1, el bonus es solo +0.05, comparado
con +0.40 del consensus factor. Dudoso que mueva la aguja en decisiones
reales.

---

## Preguntas abiertas (próxima pasada)

1. **Conectar a Weaviate**: ¿Cuántos objetos existen hoy en cada clase?
   ¿Hay datos de prueba, o la BD vectorial está vacía?

2. **Calidad de embeddings**: Los writes no envían vector explícito —
   Weaviate usa `text2vec-transformers` server-side. ¿Qué modelo corre
   realmente el Weaviate server? ¿Está configurado correctamente?

3. **Cobertura de tests**: ¿Hay planes para agregar tests A/B que midan
   impacto de memoria? ¿Existe un entorno de staging con Weaviate real?

4. **Ruta `cognitive_memory` en `universal_bridge.py`**: ¿Por qué
   `document_analyzer_factory.py` pasa `None`? ¿Fue un descuido al
   refactorizar o decisión deliberada que quedó sin documentar?

5. **`remember_pattern` y `remember_decision`**: ¿Hay planes de conectar
   estos flujos? ¿O las clases `PatternMemory` y `DecisionReasoning` deberían
   eliminarse del schema?

6. **Consistencia del adapter**: Existen **dos** implementaciones de
   `WeaviateCognitiveAdapter`:
   - `infrastructure/adapters/weaviate/weaviate_cognitive.py` (nuevo, con
     circuit breaker, sin `remember_document`)
   - `infrastructure/adapters/weaviate_cognitive_adapter.py` (antiguo, con
     `remember_document`/`recall_similar_documents`)
   ¿Deberían unificarse?

7. **Monitoreo**: ¿Hay logs o métricas que permitan saber si Weaviate está
   respondiendo? ¿O las fallas silenciosas (`logger.warning` + retorno `[]`)
   ocultan problemas de conectividad?

---

*Documento vivo — completar en próxima sesión de auditoría.*
