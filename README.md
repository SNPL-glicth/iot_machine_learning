# iot_machine_learning

## Qué hace esta parte del sistema

Este módulo contiene la lógica de **Machine Learning** y de **enriquecimiento de contexto** para el sistema IoT (Sandevistan).

En el estado actual, aquí viven:

- Un **servicio FastAPI** (`iot_machine_learning/ml_service/main.py`) para predicción puntual.
- Un **runner batch** (`ml_service/runners/ml_batch_runner.py`) que genera filas en `dbo.predictions` y puede crear/actualizar eventos en `dbo.ml_events`.
- Un **runner online** (`ml_service/runners/ml_stream_runner.py`) que consume lecturas vía la interfaz `ReadingBroker` y materializa eventos en `dbo.ml_events` + notificaciones en `dbo.alert_notifications`.
- Un **orquestador de predicciones enriquecidas** (`ml_service/orchestrator/prediction_orchestrator.py`) que construye payloads y explicaciones “con contexto” (correlación, memoria, contexto operacional).
- Un **cliente hacia AI Explainer** (HTTP) para generar explicaciones con LLM cuando está disponible (`ml_service/explain/contextual_explainer.py`).

## Qué problema resuelve

- Convertir lecturas históricas (en SQL Server) en:
  - Predicciones (`dbo.predictions`).
  - Eventos ML interpretables/accionables (`dbo.ml_events`).
  - Explicaciones estructuradas y/o textuales para UI.
- Reducir “ruido” y eventos incorrectos aplicando reglas explícitas:
  - Respetar umbrales definidos por usuario (`dbo.alert_thresholds`).
  - Evitar eventos cuando el sensor está en estados operacionales no confiables (`INITIALIZING`, `STALE`) según `SensorStateManager`.

## Cómo funciona internamente (flujo real)

### Predicción puntual (HTTP)

- Entrada: `POST /ml/predict` (FastAPI, `ml_service/main.py`).
- Fuente de datos: consulta `dbo.sensor_readings` (ventana configurable con `window`).
- Modelo usado: baseline simple de **media móvil** (`iot_machine_learning/ml/baseline.py`).
- Persistencia:
  - Si no existe un modelo activo en `dbo.ml_models` para el sensor, lo crea y lo marca activo.
  - Inserta una fila en `dbo.predictions`.
- Post-proceso (dominio): evalúa si la predicción viola umbrales y, si aplica, inserta un evento `PRED_THRESHOLD_BREACH` en `dbo.ml_events`.
  - Regla crítica observada: si el valor predicho está dentro del rango WARNING del usuario, **no** genera evento.

### Pipeline batch (cron/loop)

- Entry point: `python -m iot_machine_learning.ml_service.runners.ml_batch_runner`.
- Recorre sensores activos (vía repositorio) y para cada sensor:
  - Carga series de `dbo.sensor_readings`.
  - Entrena/regenera modelos (regresión y/o `IsolationForest`) según configuración.
  - Inserta predicción “enriquecida” en `dbo.predictions` (incluye trend/anomaly/risk/severity/explanation).
  - Puede:
    - Upsert de eventos en `dbo.ml_events` (se busca idempotencia).
    - Resolver eventos cuando la condición vuelve a normalidad.
  - Validaciones explícitas:
    - `SensorStateManager`: bloquea eventos ML si el sensor está en `INITIALIZING` o `STALE`.
    - Umbrales del usuario: suprimen eventos ML cuando el valor cae dentro del rango configurado.

### Pipeline online (stream)

- Entry point: `python -m iot_machine_learning.ml_service.runners.ml_stream_runner`.
- Consume lecturas por callbacks desde `ReadingBroker`.
- Mantiene buffers deslizantes 1s/5s/10s por sensor (`SlidingWindowBuffer`).
- Decide “patrones” (estable, oscilante, drift, spikes, etc.) y emite eventos en BD cuando hay transición significativa.
- Persiste:
  - `dbo.ml_events` (eventos técnicos/semánticos).
  - `dbo.alert_notifications` (para “campanita” / unread).
- Regla crítica observada:
  - Si el valor está dentro del rango WARNING del usuario, el pipeline online fuerza severidad `NORMAL` y **no** emite eventos `WARN/CRITICAL`.

### Enriquecimiento + explicación

- `PredictionOrchestrator` integra:
  - Metadata del sensor/dispositivo.
  - Correlaciones entre sensores.
  - Memoria histórica de decisiones (`ml_decision_memory`, etc.).
  - Contexto operacional (turno, disponibilidad, impacto, etc.).
  - Explicación contextual (plantillas o AI Explainer).
- `ContextualExplainer`:
  - Construye `EnrichedContext` desde la BD.
  - Intenta llamar al servicio externo `AI_EXPLAINER_URL` (`/explain/anomaly`).
  - Si falla, usa templates determinísticos.

## Cómo se comunica con las otras partes

- **SQL Server (`iot_database`)**:
  - Lee: `dbo.sensor_readings`, `dbo.alert_thresholds`, `dbo.sensors`, `dbo.devices`, etc.
  - Escribe: `dbo.predictions`, `dbo.ml_models`, `dbo.ml_events`, `dbo.alert_notifications`, tablas de memoria/decisión.
- **Ingesta (`iot_ingest_services`)**:
  - Publica lecturas en `ReadingBroker` (en el estado actual, broker en memoria dentro del proceso de ingesta).
  - Comparte infraestructura de conexión a BD (`iot_ingest_services.common.db`).
- **AI Explainer (`ai-explainer`)**:
  - Se consume vía HTTP (`/explain/anomaly`) si está disponible.
- **Backend (`iot_monitor_backend`)**:
  - Expone `predictions`, `ml_events`, `notifications` hacia Flutter.
  - Contiene lógica adicional de diagnóstico/conversión prediction→event en algunos escenarios.

## Ventajas del enfoque actual

- **Reglas de dominio explícitas** (umbrales de usuario primero; estados operacionales bloquean ML).
- **Dos modos de operación**:
  - Batch para análisis periódico.
  - Online para señales rápidas (patrones por ventanas).
- **Fallbacks claros**: si AI Explainer (LLM) no responde, se generan explicaciones por template.

## Desventajas o limitaciones actuales

- **Dependencia fuerte de SQL Server** como fuente de datos y como bus de integración (eventos/estado se materializan ahí).
- El **broker online** es “in-memory” (según el código), por lo que:
  - No hay distribución entre procesos.
  - Se pierde el stream si reinicia el proceso que sostiene el broker.
- La parte de “orquestación enriquecida” agrega complejidad y requiere que varias tablas/migraciones estén presentes.

## Decisiones técnicas tomadas y por qué

- **FastAPI** para predicción puntual: reduce fricción al exponer endpoints simples.
- **Persistir todo en BD** (`predictions`, `ml_events`, `alert_notifications`) para:
  - Que el backend Nest y la UI Flutter consuman “snapshots” sin recalcular.
  - Tener trazabilidad por tablas.
- **Umbrales del usuario como prioridad**: el ML se subordina a la configuración de negocio.
- **AI Explainer desacoplado**: permite reemplazar/caer a templates sin bloquear el pipeline.

## Qué NO hace esta parte

- No captura lecturas desde hardware directamente.
- No gestiona usuarios/auth.
- No garantiza “tiempo real” end-to-end (no hay streaming distribuido en el estado actual).
- No define el esquema SQL ni ejecuta migraciones (eso vive en `iot_database`).

## Preguntas tipo debate o entrevista

### ¿Por qué mantener batch y online a la vez?

Porque el código actual cubre dos necesidades distintas:

- El batch genera predicciones y análisis sobre histórico (con más contexto y entrenamiento).
- El online detecta patrones de corto plazo (ventanas 1s/5s/10s) sin esperar a un ciclo batch.

### ¿El sistema “ML” decide alertas?

Parcialmente y bajo restricciones. El ML puede crear `ml_events`, pero:

- Respeta umbrales definidos por usuario.
- Bloquea eventos si el sensor no está en un estado operacional válido.
- Las alertas “operacionales” (umbral físico) se generan desde BD (SP/trigger), no desde aquí.

### ¿Qué pasa si el AI Explainer no está disponible?

El `ContextualExplainer` cae a `template` y devuelve una explicación determinística. Esto evita que el pipeline se bloquee por un servicio externo.
