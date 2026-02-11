# UTSAE — Documento de Arquitectura y Migración
## Universal Time Series Analysis Engine
### Parte 1: Análisis, Diseño y Estrategia de Migración

**Fecha:** 2026-02-10  
**Versión:** 1.0  
**Estado:** Análisis completo del sistema actual + diseño de evolución

---

## ÍNDICE

1. [Mapa de Dependencias Completo](#1-mapa-de-dependencias-completo)
2. [Puntos de Extensión Sin Rotura](#2-puntos-de-extensión-sin-rotura)
3. [Estrategia de Migración Gradual](#3-estrategia-de-migración-gradual)
4. [Matriz de Riesgos](#4-matriz-de-riesgos)
5. [Decisiones Arquitectónicas Críticas](#5-decisiones-arquitectónicas-críticas)
6. [Preguntas para el Equipo](#6-preguntas-para-el-equipo)

---

## 1. MAPA DE DEPENDENCIAS COMPLETO

### 1.1 Diagrama de Capas Actual (ASCII)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                                 │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │ FastAPI       │  │ ml_batch_runner  │  │ ml_stream_runner      │ │
│  │ POST /ml/     │  │ (cron/loop)      │  │ (ReadingBroker)       │ │
│  │ predict       │  │                  │  │                       │ │
│  └──────┬───────┘  └────────┬─────────┘  └───────────┬───────────┘ │
│         │                   │                         │             │
├─────────┼───────────────────┼─────────────────────────┼─────────────┤
│         ▼                   ▼                         ▼             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    SERVICIOS DE NEGOCIO                      │   │
│  │  PredictionService │ SensorProcessor │ SimpleMlOnlineProc   │   │
│  │  ThresholdService  │ SeverityClassif │ WindowAnalyzer       │   │
│  │  ModelService      │ EventWriter     │ ThresholdValidator   │   │
│  │                    │ PredictionWriter│ ExplanationBuilder   │   │
│  │                    │                 │ MLEventPersister     │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
├─────────────────────────────┼───────────────────────────────────────┤
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    CAPA ML / MODELOS                         │   │
│  │  ml/baseline.py        → predict_moving_average             │   │
│  │  trainers/regression   → Ridge/Linear + predict_clamped     │   │
│  │  trainers/isolation    → IsolationForest (anomalía)         │   │
│  │  ml/pattern_detector   → PatternDetector (diagnóstico)      │   │
│  │  features/ml_features  → MLFeaturesProducer (observable)    │   │
│  │  sliding_window_buffer → WindowStats (1s/5s/10s)            │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
├─────────────────────────────┼───────────────────────────────────────┤
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    ENRIQUECIMIENTO                           │   │
│  │  orchestrator/prediction_orchestrator → EnrichedPrediction   │   │
│  │  context/operational_context          → WorkShift, Impact    │   │
│  │  context/decision_context             → RecommendedAction    │   │
│  │  correlation/sensor_correlator        → CorrelationResult    │   │
│  │  memory/decision_memory               → HistoricalInsight    │   │
│  │  explain/contextual_explainer         → ExplanationResult    │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
├─────────────────────────────┼───────────────────────────────────────┤
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    INFRAESTRUCTURA                           │   │
│  │  iot_ingest_services.common.db  → SQLAlchemy Engine         │   │
│  │  broker/ (Redis / InMemory)     → ReadingBroker Protocol    │   │
│  │  SensorStateManager             → Estado operacional        │   │
│  │  AI Explainer (HTTP externo)    → /explain/anomaly          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Dependencias por Módulo

#### MÓDULO: `ml_service/main.py` (FastAPI App)
```
├─ DEPENDENCIAS_ENTRADA
│  ├─ Broker: broker_factory.get_broker() → singleton ReadingBroker
│  └─ Router: api/routes.py
├─ DEPENDENCIAS_SALIDA
│  └─ Ninguna directa (delega a router)
└─ CONTRATOS_API
   ├─ GET  /           → {"service", "version", "status"}
   ├─ GET  /health     → HealthResponse {status, broker, version}
   ├─ POST /ml/predict → PredictResponse {sensor_id, model_id, prediction_id, ...}
   ├─ GET  /ml/broker/health → dict
   └─ GET  /ml/metrics → dict
```

#### MÓDULO: `ml_service/api/services/prediction_service.py` (PredictionService)
```
├─ DEPENDENCIAS_ENTRADA
│  ├─ BD_READ: dbo.sensor_readings (SELECT TOP :limit WHERE sensor_id)
│  ├─ BD_READ: dbo.ml_models (SELECT WHERE sensor_id AND is_active=1)
│  ├─ BD_READ: dbo.sensors (SELECT device_id WHERE id)
│  ├─ BD_READ: dbo.alert_thresholds (SELECT WHERE sensor_id AND is_active=1)
│  ├─ BD_READ: dbo.ml_events (SELECT para dedupe)
│  └─ ML: ml/baseline.py → predict_moving_average()
├─ DEPENDENCIAS_SALIDA
│  ├─ BD_WRITE: dbo.ml_models (INSERT si no existe modelo activo)
│  ├─ BD_WRITE: dbo.predictions (INSERT predicción)
│  └─ BD_WRITE: dbo.ml_events (INSERT PRED_THRESHOLD_BREACH si viola umbral)
└─ CONTRATOS_API
   └─ predict() → dict {sensor_id, model_id, prediction_id, predicted_value, ...}
```

#### MÓDULO: `ml_service/runners/ml_batch_runner.py` (MLBatchRunner)
```
├─ DEPENDENCIAS_ENTRADA
│  ├─ BD_READ: dbo.sensors (SELECT id WHERE is_active=1)
│  ├─ BD_READ: dbo.sensor_readings (SELECT TOP :limit)
│  ├─ BD_READ: dbo.sensors + dbo.devices (JOIN para metadata)
│  ├─ BD_READ: dbo.alert_thresholds (umbrales usuario)
│  ├─ BD_READ: dbo.ml_models (SELECT modelo activo)
│  ├─ BD_READ: dbo.ml_events (dedupe, delta_spike check)
│  ├─ Config: GlobalMLConfig (RegressionConfig, AnomalyConfig)
│  └─ Infra: iot_ingest_services.common.db.get_engine()
├─ DEPENDENCIAS_SALIDA
│  ├─ BD_WRITE: dbo.ml_models (INSERT/UPSERT modelo sklearn)
│  ├─ BD_WRITE: dbo.predictions (INSERT con trend/anomaly/risk/severity/explanation)
│  ├─ BD_WRITE: dbo.ml_events (UPSERT PRED_THRESHOLD_BREACH, ANOMALY_DETECTED)
│  └─ BD_UPDATE: dbo.ml_events (UPDATE status='resolved' cuando normaliza)
│  └─ BD_UPDATE: dbo.predictions (UPDATE severity si delta_spike reciente)
├─ ML_INTERNO
│  ├─ trainers/regression_trainer → Ridge/Linear + predict_future_value_clamped
│  ├─ trainers/isolation_trainer → IsolationForest fit + score
│  ├─ runners/common/severity_classifier → SeverityResult
│  └─ explain/explanation_builder → PredictionExplanation
└─ DEPENDENCIA_EXTERNA
   └─ iot_ingest_services.ingest_api.sensor_state.SensorStateManager
      (verifica INITIALIZING/STALE antes de emitir eventos)
```

#### MÓDULO: `ml_service/runners/ml_stream_runner.py` (SimpleMlOnlineProcessor)
```
├─ DEPENDENCIAS_ENTRADA
│  ├─ Broker: ReadingBroker.subscribe(handler) → Reading(sensor_id, sensor_type, value, timestamp)
│  ├─ BD_READ: dbo.alert_thresholds (warning range check)
│  ├─ BD_READ: dbo.predictions (SELECT para autovalidación predicción)
│  ├─ BD_READ: dbo.sensors (device_id lookup)
│  ├─ BD_READ: dbo.ml_events (cooldown + dedupe check)
│  ├─ Config: OnlineBehaviorConfig
│  └─ Infra: SensorStateManager (can_generate_events)
├─ DEPENDENCIAS_SALIDA
│  ├─ BD_WRITE: dbo.ml_events (INSERT eventos de patrón + PREDICTION_DEVIATION)
│  └─ BD_WRITE: dbo.alert_notifications (INSERT notificación por cada ml_event)
├─ ESTADO_EN_MEMORIA
│  ├─ SlidingWindowBuffer (buffers 1s/5s/10s por sensor)
│  ├─ _last_state: Dict[int, SensorState] (último estado por sensor)
│  ├─ MLFeaturesProducer (singleton, ventanas de features)
│  └─ ThresholdValidator._thresholds_cache (cache de umbrales)
└─ ML_INTERNO
   ├─ WindowAnalyzer → OnlineAnalysis
   ├─ ExplanationBuilder → severity, action, explanation
   └─ MLFeaturesProducer → MLFeatures (baseline, z_score, pattern, etc.)
```

#### MÓDULO: `ml_service/orchestrator/prediction_orchestrator.py` (PredictionOrchestrator)
```
├─ DEPENDENCIAS_ENTRADA
│  ├─ BD_READ: dbo.sensors + dbo.devices (metadata)
│  ├─ BD_READ: dbo.sensor_readings (current_value, stats)
│  ├─ BD_READ: dbo.predictions (últimas predicciones por sensor)
│  ├─ BD_READ: dbo.alert_thresholds (umbrales usuario)
│  ├─ BD_READ: dbo.ml_events (eventos correlacionados, historial)
│  ├─ BD_READ: dbo.ml_decision_memory **[CLARIFICAR: tabla exacta]**
│  └─ Servicio: AI Explainer HTTP (POST /explain/anomaly)
├─ DEPENDENCIAS_SALIDA
│  ├─ BD_WRITE: dbo.ml_decision_memory (INSERT registro de decisión)
│  └─ Retorno: EnrichedPrediction (payload completo para UI/backend)
└─ SUBMÓDULOS
   ├─ context/decision_context → DecisionContextBuilder
   ├─ context/operational_context → OperationalContextBuilder
   ├─ correlation/sensor_correlator → SensorCorrelator
   ├─ memory/decision_memory → DecisionMemoryService
   └─ explain/contextual_explainer → ContextualExplainer
```

#### MÓDULO: `ml_service/explain/contextual_explainer.py` (ContextualExplainer)
```
├─ DEPENDENCIAS_ENTRADA
│  ├─ BD_READ: dbo.sensors + dbo.devices (info sensor)
│  ├─ BD_READ: dbo.sensor_readings (valor actual, stats recientes)
│  ├─ BD_READ: dbo.alert_thresholds (umbrales)
│  ├─ BD_READ: dbo.ml_events (eventos correlacionados, historial)
│  └─ HTTP: AI_EXPLAINER_URL/explain/anomaly (POST, async)
├─ DEPENDENCIAS_SALIDA
│  └─ Retorno: ExplanationResult (explanation_text, source: "ai"|"template")
└─ FALLBACK
   └─ TemplateExplanationGenerator (si AI Explainer falla)
```

### 1.3 Mapa Completo de Tablas SQL Server

| Tabla | Operación | Módulos que la usan |
|-------|-----------|---------------------|
| `dbo.sensor_readings` | SELECT | PredictionService, sensor_repository, SensorCorrelator, ExplainerDataLoader |
| `dbo.sensors` | SELECT | sensor_repository, PredictionService, SensorCorrelator, ThresholdValidator |
| `dbo.devices` | SELECT | sensor_repository, SensorCorrelator, ExplainerDataLoader |
| `dbo.alert_thresholds` | SELECT | PredictionService, ThresholdService, SeverityClassifier, ThresholdValidator |
| `dbo.ml_models` | SELECT/INSERT | PredictionService, ModelManager, prediction_repository |
| `dbo.predictions` | SELECT/INSERT/UPDATE | PredictionService, PredictionWriter, ml_stream_runner, SensorCorrelator |
| `dbo.ml_events` | SELECT/INSERT/UPDATE | PredictionService, EventWriter, MLEventPersister, SensorCorrelator |
| `dbo.alert_notifications` | INSERT | MLEventPersister (online runner) |
| `dbo.ml_decision_memory` | SELECT/INSERT | DecisionMemoryService **[CLARIFICAR: nombre exacto]** |

### 1.4 Servicios Externos

| Servicio | Protocolo | Endpoint | Consumido por | Fallback |
|----------|-----------|----------|---------------|----------|
| AI Explainer | HTTP POST | `AI_EXPLAINER_URL/explain/anomaly` | ContextualExplainer | TemplateExplanationGenerator |
| Redis Streams | Redis Protocol | `REDIS_URL` | RedisReadingBroker | InMemoryReadingBroker |
| SQL Server | ODBC/SQLAlchemy | `iot_ingest_services.common.db` | Todos los módulos | Ninguno (crítico) |

### 1.5 Estado Compartido en Memoria

| Componente | Tipo | Scope | Persistencia |
|------------|------|-------|-------------|
| `SlidingWindowBuffer` | Dict[sensor_id → deque] | Proceso stream runner | No persiste (se pierde al reiniciar) |
| `_last_state` | Dict[sensor_id → SensorState] | Proceso stream runner | No persiste |
| `MLFeaturesProducer` | Singleton con ventanas | Proceso stream runner | No persiste |
| `IsolationForestTrainer._estimators` | Dict[sensor_id → IsolationForest] | Proceso batch runner | No persiste (se reentrena cada ciclo) |
| `SensorCorrelator._sensor_cache` | Dict[device_id → DeviceSensorGroup] | Por instancia | No persiste |
| `ThresholdValidator._thresholds_cache` | Dict[sensor_id → dict] | Proceso stream runner | No persiste |
| `_broker_instance` | Singleton ReadingBroker | Proceso FastAPI | No persiste |

---

## 2. PUNTOS DE EXTENSIÓN SIN ROTURA

### 2.1 Patrón Strategy para Motor de Predicción

El punto de inyección más limpio es en la **capa de predicción**. Actualmente hay dos caminos:
- **API (PredictionService):** usa `predict_moving_average()` directamente
- **Batch (SensorProcessor):** usa `train_regression_for_sensor()` + `predict_future_value_clamped()`

**Patrón propuesto:**

```python
# ml/core/prediction_engine.py (NUEVO)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass(frozen=True)
class PredictionResult:
    """Resultado unificado de cualquier motor de predicción."""
    predicted_value: float
    confidence: float
    trend: str  # "up" | "down" | "stable"
    metadata: dict  # Motor-specific info (coefs, order, etc.)

class PredictionEngine(ABC):
    """Interfaz abstracta para motores de predicción."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del motor (para logging/métricas)."""
        ...
    
    @abstractmethod
    def predict(self, values: List[float], timestamps: Optional[List[float]] = None) -> PredictionResult:
        """Genera predicción a partir de una ventana de valores."""
        ...
    
    @abstractmethod
    def can_handle(self, n_points: int) -> bool:
        """Indica si el motor puede operar con n_points datos."""
        ...


# Actual (mantener como fallback)
class BaselineMovingAverageEngine(PredictionEngine):
    @property
    def name(self) -> str:
        return "baseline_moving_average"
    
    def predict(self, values, timestamps=None):
        from iot_machine_learning.ml.baseline import predict_moving_average, BaselineConfig
        cfg = BaselineConfig(window=len(values))
        pred, conf = predict_moving_average(values, cfg)
        return PredictionResult(predicted_value=pred, confidence=conf, trend="stable", metadata={})
    
    def can_handle(self, n_points):
        return n_points >= 1


# Nuevo (Fase 1)
class TaylorPredictionEngine(PredictionEngine):
    """Motor basado en aproximación de Taylor de orden configurable."""
    @property
    def name(self) -> str:
        return "taylor_approximation"
    
    def predict(self, values, timestamps=None):
        # Implementación en Fase 1
        ...
    
    def can_handle(self, n_points):
        return n_points >= 5  # Necesita mínimo para derivadas
```

**Punto de inyección:** `PredictionService.predict()` y `SensorProcessor.process()` seleccionan el engine vía configuración, sin modificar contratos de API ni esquemas de BD.

### 2.2 Patrón Decorator para Filtro Pre/Post Predicción (Kalman)

```python
# ml/core/signal_filter.py (NUEVO)
from abc import ABC, abstractmethod
from typing import List

class SignalFilter(ABC):
    """Filtro aplicable pre o post predicción."""
    
    @abstractmethod
    def filter(self, values: List[float], timestamps: List[float]) -> List[float]:
        """Filtra/suaviza la señal."""
        ...

class IdentityFilter(SignalFilter):
    """No-op filter (comportamiento actual)."""
    def filter(self, values, timestamps):
        return values

class KalmanFilter(SignalFilter):
    """Filtro de Kalman para separación ruido/señal."""
    def filter(self, values, timestamps):
        # Implementación en Fase 1
        ...
```

**Punto de inyección:** Se aplica ANTES de pasar valores al `PredictionEngine`. El flujo actual:
```
sensor_readings → [valores crudos] → PredictionEngine → predicted_value
```
Se convierte en:
```
sensor_readings → [valores crudos] → SignalFilter → [valores filtrados] → PredictionEngine → predicted_value
```

Sin modificar BD, API, ni lógica de umbrales.

### 2.3 Patrón Observer para Capa Cognitiva (Change Points)

```python
# ml/core/analysis_observer.py (NUEVO)
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class AnalysisInsight:
    """Insight generado por un observador de análisis."""
    insight_type: str  # "change_point", "regime_shift", etc.
    confidence: float
    description: str
    metadata: dict

class AnalysisObserver(ABC):
    """Observador que analiza datos y genera insights sin modificar el flujo."""
    
    @abstractmethod
    def observe(self, values: List[float], timestamps: List[float]) -> Optional[AnalysisInsight]:
        ...

class ChangePointDetector(AnalysisObserver):
    """Detecta puntos de cambio en la serie temporal."""
    def observe(self, values, timestamps):
        # Implementación en Fase 2
        ...
```

**Punto de inyección:** Se ejecuta EN PARALELO al flujo de predicción. Los insights se agregan al payload de `PredictionExplanation` sin modificar la lógica de eventos/umbrales existente.

### 2.4 Patrón Factory + Feature Flags para Selección de Engine

```python
# ml/core/engine_factory.py (NUEVO)
from typing import Optional

class EngineFactory:
    """Factory que selecciona el motor de predicción según configuración."""
    
    _registry: dict = {}
    
    @classmethod
    def register(cls, name: str, engine_class):
        cls._registry[name] = engine_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> "PredictionEngine":
        if name not in cls._registry:
            # Fallback seguro al baseline
            return BaselineMovingAverageEngine()
        return cls._registry[name](**kwargs)
    
    @classmethod
    def get_engine_for_sensor(cls, sensor_id: int, config: dict) -> "PredictionEngine":
        """Selecciona engine según feature flags y config del sensor."""
        engine_name = config.get(f"engine.sensor.{sensor_id}", 
                                 config.get("engine.default", "baseline_moving_average"))
        return cls.create(engine_name)
```

### 2.5 Resumen de Puntos de Inyección

| Punto de Inyección | Archivo Actual | Qué se inyecta | Impacto en BD | Impacto en API |
|---------------------|----------------|-----------------|---------------|----------------|
| Motor de predicción (API) | `api/services/prediction_service.py:64` | `PredictionEngine` | Ninguno | Ninguno |
| Motor de predicción (Batch) | `runners/common/sensor_processor.py:101-112` | `PredictionEngine` | Ninguno | N/A |
| Filtro pre-predicción | Antes de `predict_moving_average()` / `train_regression` | `SignalFilter` | Ninguno | Ninguno |
| Análisis cognitivo | Paralelo a predicción en `SensorProcessor.process()` | `AnalysisObserver` | Ninguno (solo enriquece payload) | Ninguno |
| Features observables | `ml_service/features/ml_features.py` | Nuevos features | Ninguno | Ninguno |
| Explicabilidad | `explain/contextual_explainer.py` | Nuevos templates | Ninguno | Ninguno |

---

## 3. ESTRATEGIA DE MIGRACIÓN GRADUAL

### 3.1 Roadmap en 4 Fases

```
Fase 0 ─────────► Fase 1 ─────────► Fase 2 ─────────► Fase 3
Preparación        Core Matemático    Integración        Consolidación
(2 semanas)        (3-4 semanas)      (3-4 semanas)      (2-3 semanas)
Sin cambios        Nueva capa,        Reemplazo          Deprecar
funcionales        viejo API          progresivo         legacy
```

### Fase 0: Preparación (2 semanas) — Sin cambios funcionales

| Tarea | Detalle | Entregable |
|-------|---------|------------|
| Tests de integración | Capturar comportamiento actual de `PredictionService.predict()`, `SensorProcessor.process()`, `SimpleMlOnlineProcessor.handle_reading()` | `tests/integration/test_current_behavior.py` |
| Snapshot de métricas | Registrar MAE/RMSE actual del baseline vs valores reales para N sensores piloto | `docs/metrics_baseline.md` |
| Feature flags | Crear `ml_service/config/feature_flags.py` con flags: `USE_TAYLOR_ENGINE`, `USE_KALMAN_FILTER`, `USE_CHANGE_POINT_DETECTION`, `ENABLE_AB_TESTING` | Archivo de config |
| Interfaces abstractas | Crear `ml/core/prediction_engine.py`, `ml/core/signal_filter.py`, `ml/core/analysis_observer.py` con implementaciones identity/baseline | Código (sin uso aún) |
| Documentar contratos | Documentar schemas exactos de `PredictResponse`, payload de `dbo.predictions`, payload de `dbo.ml_events` | `docs/contracts.md` |

### Fase 1: Core Matemático (3-4 semanas) — Nueva capa, viejo API

| Tarea | Detalle | Entregable |
|-------|---------|------------|
| `TaylorPredictionEngine` | Implementar aproximación de Taylor de orden 1-3 con estabilidad numérica | `ml/core/taylor.py` |
| `KalmanSignalFilter` | Implementar Kalman filter 1D para separación ruido/señal | `ml/core/kalman.py` |
| Registrar en Factory | `EngineFactory.register("taylor", TaylorPredictionEngine)` | `ml/core/engine_factory.py` |
| A/B testing framework | Comparar baseline vs Taylor en paralelo, loggear MAE/RMSE sin cambiar output | `ml_service/metrics/ab_testing.py` |
| Tests unitarios | Taylor: convergencia, estabilidad numérica, edge cases. Kalman: cold start, convergencia | `tests/unit/test_taylor.py`, `tests/unit/test_kalman.py` |
| Config por sensor | Permitir `engine.sensor.{id}` override en config | Extensión de `ml_config.py` |

**Criterio de avance a Fase 2:** MAE de Taylor ≤ MAE de baseline en ≥80% de sensores piloto.

### Fase 2: Integración (3-4 semanas) — Reemplazo progresivo

| Tarea | Detalle | Entregable |
|-------|---------|------------|
| Migrar batch runner | `SensorProcessor` usa `EngineFactory.get_engine_for_sensor()` en lugar de `train_regression_for_sensor()` directo | Modificación de `sensor_processor.py` |
| Mantener online sin cambios | Stream runner sigue con `WindowAnalyzer` actual | Sin cambios |
| Sensores piloto | Activar Taylor+Kalman para 3-5 sensores vía feature flag | Config change |
| Change Point Detection | Implementar `ChangePointDetector` como `AnalysisObserver` | `ml/core/change_points.py` |
| Capa heurística | Extraer cooldown/dedupe/reglas a `ml/core/heuristics.py` | Refactor de `MLEventPersister` |
| Métricas en dashboard | Exponer MAE/RMSE comparativo en `/ml/metrics` | Extensión de `metrics/` |

**Criterio de avance a Fase 3:** 2+ semanas sin regresiones en sensores piloto.

### Fase 3: Consolidación (2-3 semanas) — Deprecar legacy

| Tarea | Detalle | Entregable |
|-------|---------|------------|
| Migrar online runner | `SimpleMlOnlineProcessor` usa `SignalFilter` + `PredictionEngine` | Modificación de `ml_stream_runner.py` |
| Explicabilidad Feynman | Integrar insights de Taylor/Kalman/ChangePoint en `ContextualExplainer` | Templates nuevos |
| Deprecar baseline | Marcar `predict_moving_average` como deprecated (mantener como fallback) | Deprecation warnings |
| Actualizar docs | README, contratos, diagramas | Documentación |
| Cleanup | Remover feature flags de Fase 1, consolidar config | Config simplificada |

### 3.2 Tabla Resumen del Roadmap

| Fase | Duración | Riesgo | Cambios BD | Cambios API | Rollback |
|------|----------|--------|------------|-------------|----------|
| **Fase 0** | 2 sem | BAJO | Ninguno | Ninguno | N/A |
| **Fase 1** | 3-4 sem | BAJO | Ninguno | Ninguno | Desactivar flag |
| **Fase 2** | 3-4 sem | MEDIO | Ninguno | Ninguno | Revertir flag por sensor |
| **Fase 3** | 2-3 sem | MEDIO-BAJO | Ninguno | Opcional: nuevos campos en response | Revertir a Fase 2 |

---

## 4. MATRIZ DE RIESGOS

### 4.1 Riesgos por Componente

| Componente | Cambio Propuesto | Riesgo Rotura | Probabilidad | Impacto | Mitigación | Rollback Plan |
|------------|-----------------|---------------|-------------|---------|------------|---------------|
| `TaylorPredictionEngine` en batch | Reemplazar Ridge por Taylor | MEDIO | 30% | ALTO (predicciones incorrectas) | Feature flag + A/B testing con MAE/RMSE. Criterio: MAE ≤ baseline | Revertir flag → baseline automático |
| `KalmanSignalFilter` pre-predicción | Filtrar valores antes de predecir | MEDIO | 25% | MEDIO (suavizado excesivo) | Parámetro de intensidad configurable. Comparar con/sin filtro | Desactivar filtro → `IdentityFilter` |
| `ChangePointDetector` en batch | Agregar detección de cambios | BAJO | 10% | BAJO (solo enriquece, no decide) | Observer pattern: no modifica flujo principal | Desactivar observer |
| Migrar `SensorProcessor` a Factory | Cambiar cómo se selecciona engine | MEDIO | 20% | ALTO (rompe batch) | Tests de integración previos (Fase 0). Factory con fallback a baseline | Revertir commit, volver a imports directos |
| Migrar `SimpleMlOnlineProcessor` | Agregar SignalFilter al stream | MEDIO-ALTO | 35% | ALTO (afecta detección en tiempo real) | Última fase. Solo después de validar en batch. Feature flag por sensor | Desactivar filtro en online |
| Capa heurística (cooldown/dedupe) | Extraer a módulo separado | BAJO | 15% | MEDIO (eventos duplicados) | Tests de regresión sobre cooldown/dedupe actual | Revertir refactor |
| Explicabilidad Feynman | Nuevos templates de explicación | BAJO | 5% | BAJO (solo texto) | Fallback a templates actuales | Revertir templates |
| Estado Kalman por sensor | Mantener estado en memoria | MEDIO | 25% | MEDIO (cold start) | Inicialización con media de ventana. Timeout para reset | Reset estado → recalcular |

### 4.2 Riesgos Transversales

| Riesgo | Descripción | Probabilidad | Mitigación |
|--------|-------------|-------------|------------|
| **Inestabilidad numérica** | Taylor de orden alto puede diverger con datos ruidosos | MEDIA | Limitar orden a 3. Usar `numeric_precision.py` existente. Clamp como en `predict_future_value_clamped` |
| **Latencia en online** | Kalman + Taylor agregan tiempo de cómputo por lectura | BAJA | Ambos son O(n) con n pequeño (ventana). Medir con `time.time()` existente |
| **Pérdida de estado al reiniciar** | Kalman pierde estado, SlidingWindowBuffer se vacía | MEDIA (ya existe) | Cold start con media de primeras N lecturas. Documentar como limitación conocida |
| **Dependencia de numpy/scipy** | Nuevas dependencias pueden conflictuar | BAJA | numpy ya está en uso. scipy solo si es necesario para Kalman avanzado |
| **Regresión en umbrales de usuario** | Nuevo engine podría generar predicciones que violen umbrales de forma diferente | MEDIA | La lógica de umbrales (`ThresholdValidator`, `SeverityClassifier`) NO se modifica. Opera post-predicción |

---

## 5. DECISIONES ARQUITECTÓNICAS CRÍTICAS

### 5.1 Capa Matemática (Taylor)

**¿Dónde vive el código?**
```
ml/
├── core/                          # NUEVO directorio
│   ├── __init__.py
│   ├── prediction_engine.py       # ABC + PredictionResult
│   ├── taylor.py                  # TaylorPredictionEngine
│   ├── signal_filter.py           # ABC + IdentityFilter
│   ├── engine_factory.py          # Factory + registro
│   └── numeric_stability.py       # Guards numéricos para Taylor
├── baseline.py                    # SIN CAMBIOS (se registra como engine)
├── pattern_detector.py            # SIN CAMBIOS
└── metadata.py                    # SIN CAMBIOS
```

**¿Qué interfaces expone?**
- `TaylorPredictionEngine.predict(values, timestamps) → PredictionResult`
- `TaylorPredictionEngine.can_handle(n_points) → bool`
- `TaylorPredictionEngine.get_coefficients() → List[float]` (para explicabilidad)
- `TaylorPredictionEngine.get_approximation_order() → int`

**¿Cómo se testea en aislamiento?**
```python
# tests/unit/test_taylor.py
def test_taylor_linear_signal():
    """Taylor orden 1 debe coincidir con regresión lineal."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    engine = TaylorPredictionEngine(order=1)
    result = engine.predict(values)
    assert abs(result.predicted_value - 6.0) < 0.1

def test_taylor_stability_with_noise():
    """Taylor no debe diverger con ruido gaussiano."""
    values = [20.0 + random.gauss(0, 0.5) for _ in range(50)]
    engine = TaylorPredictionEngine(order=3)
    result = engine.predict(values)
    assert 15.0 < result.predicted_value < 25.0  # No diverge

def test_taylor_cold_start():
    """Con pocos puntos, debe caer a orden menor o fallback."""
    values = [1.0, 2.0]
    engine = TaylorPredictionEngine(order=3)
    result = engine.predict(values)
    assert result.metadata.get("effective_order", 0) <= 1
```

**¿Qué dependencias numpy/scipy necesita?**
- `numpy` (ya presente): para arrays, polyfit, polyder
- `scipy` (opcional): solo si se implementa Taylor con derivadas numéricas vía `scipy.interpolate`. **Recomendación:** empezar con `numpy.polyfit` + `numpy.polyder` que son suficientes para orden ≤ 3 y no agregan dependencia nueva.

**Implementación sugerida (snippet conceptual):**
```python
import numpy as np

class TaylorPredictionEngine(PredictionEngine):
    def __init__(self, order: int = 2, horizon_steps: int = 1):
        self._order = min(order, 3)  # Limitar para estabilidad
        self._horizon = horizon_steps
    
    def predict(self, values, timestamps=None):
        n = len(values)
        effective_order = min(self._order, n - 1)
        
        x = np.arange(n, dtype=float)
        coeffs = np.polyfit(x, values, effective_order)
        
        # Evaluar en t = n + horizon
        t_future = float(n + self._horizon - 1)
        predicted = float(np.polyval(coeffs, t_future))
        
        # Clamp para estabilidad (reusar patrón existente)
        series_min, series_max = min(values), max(values)
        margin = (series_max - series_min) * 0.5
        predicted = max(series_min - margin, min(predicted, series_max + margin))
        
        # Trend desde derivada primera
        if effective_order >= 1:
            deriv = np.polyder(coeffs)
            slope = float(np.polyval(deriv, t_future))
            trend = "up" if slope > 1e-3 else "down" if slope < -1e-3 else "stable"
        else:
            trend = "stable"
        
        confidence = min(0.95, n / 50.0)  # Escala con datos
        
        return PredictionResult(
            predicted_value=predicted,
            confidence=confidence,
            trend=trend,
            metadata={"order": effective_order, "coefficients": coeffs.tolist()},
        )
```

### 5.2 Capa Probabilística (Kalman)

**¿Se aplica pre-predicción o post-predicción?**

**Pre-predicción.** Razones:
1. El Kalman filter separa señal de ruido. El motor de predicción debe operar sobre la señal limpia.
2. Aplicar post-predicción no tiene sentido matemático (la predicción ya es un estimado puntual).
3. El flujo queda: `raw values → KalmanFilter → clean values → PredictionEngine → prediction`

**¿Mantiene estado por sensor? ¿Dónde se persiste?**

Sí, mantiene estado por sensor (media estimada, covarianza). Se persiste **en memoria** dentro del proceso, igual que `SlidingWindowBuffer` y `_last_state` actuales.

**Justificación:** El sistema ya acepta pérdida de estado en memoria al reiniciar (ver `SlidingWindowBuffer`, `IsolationForestTrainer._estimators`). Agregar persistencia en BD para Kalman sería over-engineering en esta fase.

**Estructura de estado:**
```python
@dataclass
class KalmanState:
    """Estado del filtro de Kalman por sensor."""
    x_hat: float      # Estimación actual
    P: float           # Covarianza del error
    Q: float           # Varianza del proceso (configurable)
    R: float           # Varianza de medición (estimada de datos)
    initialized: bool
```

**¿Cómo se inicializa (arranque en frío)?**

Estrategia de cold start en 3 pasos:
1. **Primeras N lecturas (N=5):** Acumular sin filtrar. Usar media como `x_hat` inicial, varianza como `P` inicial.
2. **Estimación de R:** Varianza de las primeras N lecturas como proxy de ruido de medición.
3. **Q por defecto:** Configurable, default `0.01` (proceso lento). Ajustable por tipo de sensor.

```python
class KalmanSignalFilter(SignalFilter):
    def __init__(self, Q: float = 0.01, cold_start_n: int = 5):
        self._states: Dict[int, KalmanState] = {}
        self._Q = Q
        self._cold_start_n = cold_start_n
        self._cold_start_buffer: Dict[int, List[float]] = {}
    
    def filter_value(self, sensor_id: int, value: float) -> float:
        """Filtra un valor individual (para online)."""
        state = self._states.get(sensor_id)
        if state is None or not state.initialized:
            return self._cold_start(sensor_id, value)
        return self._update(state, value)
    
    def filter(self, values, timestamps):
        """Filtra una serie completa (para batch)."""
        # Cold start con primeros N valores, luego filtrar el resto
        ...
```

**¿Dónde vive el código?**
```
ml/core/kalman.py
```

### 5.3 Capa Heurística (Cooldown, Dedupe, Reglas de Negocio)

**Estado actual:** La lógica heurística está dispersa en:
- `MLEventPersister.insert_ml_event()` → cooldown 5 min, dedupe
- `EventWriter.upsert_event()` → MERGE para idempotencia
- `ThresholdValidator.is_value_within_warning_range()` → supresión por umbrales
- `EventWriter.can_sensor_emit_events()` → bloqueo por estado operacional

**Propuesta:** Extraer a un módulo unificado SIN modificar el comportamiento:

```
ml/core/heuristics.py  →  EventHeuristics (cooldown, dedupe, suppression rules)
```

**No se modifica la lógica**, solo se centraliza para que las nuevas capas (Taylor, Kalman) pasen por las mismas reglas. Esto es un refactor puro, no un cambio funcional.

### 5.4 Capa Cognitiva (Change Points)

**¿Reemplaza o complementa la lógica actual de patrones?**

**Complementa.** El `PatternDetector` actual clasifica patrones (STABLE, SPIKE, DRIFT, etc.) basándose en heurísticas simples. El `ChangePointDetector` agrega una capa de análisis que identifica **cuándo** cambió el régimen del sensor, no solo **qué** patrón tiene ahora.

Ambos coexisten:
- `PatternDetector` → "¿Qué está pasando ahora?" (diagnóstico)
- `ChangePointDetector` → "¿Cuándo cambió el comportamiento?" (contexto temporal)

**¿Cómo se integra con `ml_events`?**

Como `AnalysisObserver`, genera `AnalysisInsight` que se agrega al payload de `PredictionExplanation.explanation` (campo JSON). **No genera eventos propios** en Fase 2. En Fase 3, podría generar un event_code `REGIME_CHANGE` si se valida su utilidad.

**¿Qué librería usar?**

| Opción | Pros | Contras | Recomendación |
|--------|------|---------|---------------|
| `ruptures` | Madura, múltiples algoritmos, bien documentada | Dependencia nueva (~2MB) | ✅ **Recomendada para Fase 2** |
| Implementación propia (CUSUM) | Sin dependencias, simple | Menos robusta, más mantenimiento | Alternativa si `ruptures` es problema |
| `scipy.signal.find_peaks` | Ya semi-disponible (scipy) | No es change point detection real | No recomendada |

**Recomendación:** Usar `ruptures` con algoritmo `Pelt` (penalización lineal, O(n log n)) para detección offline en batch. Para online, implementar CUSUM propio (simple, O(1) por punto).

```python
# ml/core/change_points.py
class ChangePointDetector(AnalysisObserver):
    def __init__(self, min_segment_size: int = 10, penalty: float = 3.0):
        self._min_size = min_segment_size
        self._penalty = penalty
    
    def observe(self, values, timestamps):
        if len(values) < self._min_size * 2:
            return None
        
        import ruptures as rpt
        algo = rpt.Pelt(model="rbf", min_size=self._min_size)
        algo.fit(np.array(values))
        change_points = algo.predict(pen=self._penalty)
        
        if not change_points or change_points == [len(values)]:
            return None
        
        return AnalysisInsight(
            insight_type="change_point",
            confidence=0.8,  # Ajustar según penalización
            description=f"Detectados {len(change_points)-1} cambios de régimen",
            metadata={"change_points": change_points},
        )
```

### 5.5 Explicabilidad Estilo Feynman

**Integración con el sistema actual:**

El `ContextualExplainer` ya tiene un patrón de fallback (AI → template). La explicabilidad Feynman se integra como un **nuevo generador de templates** que usa los insights de Taylor/Kalman/ChangePoint:

```python
# explain/feynman_templates.py (NUEVO)
class FeynmanExplanationGenerator:
    """Genera explicaciones 'estilo Feynman': simples, precisas, con analogías."""
    
    def generate(self, context, taylor_metadata=None, kalman_state=None, change_points=None):
        parts = []
        
        if taylor_metadata:
            order = taylor_metadata.get("order", 1)
            if order == 1:
                parts.append("El sensor sigue una tendencia lineal clara.")
            elif order == 2:
                parts.append("El sensor está acelerando (curva parabólica).")
            # etc.
        
        if kalman_state and kalman_state.initialized:
            noise_ratio = kalman_state.R / max(abs(kalman_state.x_hat), 1e-6)
            if noise_ratio > 0.1:
                parts.append(f"Hay bastante ruido ({noise_ratio:.0%}), la señal real es más estable de lo que parece.")
        
        if change_points:
            parts.append(f"El comportamiento cambió hace {change_points[-1]} lecturas.")
        
        return " ".join(parts)
```

---

## 6. PREGUNTAS PARA EL EQUIPO

### Críticas (bloquean diseño)

1. **[CLARIFICAR] Tabla `ml_decision_memory`:** ¿Cuál es el nombre exacto de la tabla donde `DecisionMemoryService` persiste decisiones? El código referencia `record_decision()` pero no vi el SQL exacto en los archivos leídos. Necesito confirmar el schema para saber si los insights de Taylor/Kalman pueden agregarse ahí.

2. **[CLARIFICAR] Persistencia de estado Kalman:** ¿Es aceptable que el estado del Kalman filter se pierda al reiniciar el proceso (igual que `SlidingWindowBuffer`)? ¿O hay requisito de continuidad entre reinicios?

3. **[CLARIFICAR] Sensores piloto:** ¿Cuáles sensores se usarían para A/B testing en Fase 1? ¿Hay sensores con datos históricos abundantes y variados (spikes, drift, estable)?

4. **[CLARIFICAR] `predictions` schema:** La tabla `dbo.predictions` en batch tiene columnas `trend`, `is_anomaly`, `anomaly_score`, `risk_level`, `severity`, `explanation`. ¿Estas columnas existen también en el schema usado por la API (`PredictionService`)? La API inserta un subset más pequeño. ¿Es intencional?

### Importantes (afectan timeline)

5. **Dependencia `ruptures`:** ¿Hay restricciones para agregar dependencias Python nuevas al proyecto? `ruptures` es ~2MB y solo se usaría en Fase 2.

6. **Redis vs InMemory:** ¿El broker Redis está activo en producción o solo InMemory? Esto afecta si el estado del Kalman filter podría eventualmente persistirse en Redis.

7. **Frecuencia del batch runner:** ¿Cada cuánto corre actualmente? El default es 60s (`--interval-seconds`). Taylor con orden 2-3 sobre 500 puntos es O(n) con numpy, no debería agregar latencia significativa, pero necesito confirmar.

8. **AI Explainer:** ¿Está activo en producción? Si sí, ¿se planea que los insights de Taylor/Kalman se envíen al LLM para explicación enriquecida?

### Nice-to-have (mejoran diseño)

9. **Métricas actuales:** ¿Hay algún dashboard o sistema de métricas donde se pueda visualizar el A/B testing (MAE baseline vs Taylor)?

10. **Tests existentes:** ¿Hay tests automatizados actualmente? No encontré directorio `tests/` en el proyecto. Esto afecta la Fase 0 significativamente.

11. **`iot_ingest_services.ingest_api.sensor_state.SensorStateManager`:** ¿Este módulo es estable o está en evolución? Las nuevas capas dependen de `can_generate_events()` para respetar estados operacionales.

---

## APÉNDICE A: Estructura de Archivos Propuesta (Post Fase 3)

```
iot_machine_learning/
├── ml/
│   ├── __init__.py                    # Re-exports
│   ├── baseline.py                    # SIN CAMBIOS (deprecated pero funcional)
│   ├── metadata.py                    # SIN CAMBIOS
│   ├── pattern_detector.py            # SIN CAMBIOS
│   └── core/                          # ★ NUEVO: Capas UTSAE
│       ├── __init__.py
│       ├── prediction_engine.py       # ABC PredictionEngine + PredictionResult
│       ├── taylor.py                  # TaylorPredictionEngine
│       ├── signal_filter.py           # ABC SignalFilter + IdentityFilter
│       ├── kalman.py                  # KalmanSignalFilter
│       ├── analysis_observer.py       # ABC AnalysisObserver
│       ├── change_points.py           # ChangePointDetector
│       ├── heuristics.py              # EventHeuristics (cooldown, dedupe, suppression)
│       ├── engine_factory.py          # EngineFactory + registro
│       └── numeric_stability.py       # Guards numéricos para Taylor
├── ml_service/
│   ├── config/
│   │   ├── ml_config.py              # EXTENDIDO: + EngineConfig, FilterConfig
│   │   └── feature_flags.py          # ★ NUEVO: Feature flags
│   ├── metrics/
│   │   ├── performance_metrics.py    # SIN CAMBIOS
│   │   └── ab_testing.py             # ★ NUEVO: A/B testing framework
│   ├── explain/
│   │   ├── contextual_explainer.py   # EXTENDIDO: + FeynmanExplanationGenerator
│   │   └── feynman_templates.py      # ★ NUEVO: Templates Feynman
│   └── ... (resto sin cambios)
└── tests/                             # ★ NUEVO
    ├── unit/
    │   ├── test_taylor.py
    │   ├── test_kalman.py
    │   ├── test_change_points.py
    │   └── test_engine_factory.py
    └── integration/
        ├── test_current_behavior.py   # Snapshot del comportamiento actual
        └── test_ab_comparison.py      # Comparación baseline vs UTSAE
```

## APÉNDICE B: Dependencias Python

| Paquete | Versión | Ya presente | Fase | Uso |
|---------|---------|-------------|------|-----|
| `numpy` | ≥1.24 | ✅ Sí | 1 | polyfit, polyder, arrays |
| `scikit-learn` | ≥1.3 | ✅ Sí | - | IsolationForest, Ridge (existente) |
| `sqlalchemy` | ≥2.0 | ✅ Sí | - | BD (existente) |
| `fastapi` | ≥0.100 | ✅ Sí | - | API (existente) |
| `pydantic` | ≥2.0 | ✅ Sí | - | Schemas (existente) |
| `ruptures` | ≥1.1.8 | ❌ No | 2 | Change point detection |
| `scipy` | ≥1.11 | ❓ Verificar | 1 (opcional) | Solo si Kalman avanzado |

---

*Documento generado como análisis de Parte 1. No contiene código completo — solo snippets de patrones y decisiones arquitectónicas.*
