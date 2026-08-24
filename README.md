# ZENIN Machine Learning

Motor cognitivo para análisis y predicción de series temporales IoT, con paper-trading BTC-USD live contra Binance.

---

## Qué hace

**ZENIN Market (MVP 0.1 — activo)**
- Bot paper BTC-USD 24/7 sobre Binance público (REST, sin API key, $0 COP).
- Ciclos stateless con MySQL como estado: crash-proof trivial, reentrante.
- Pipeline: velas cerradas → observación → feature window → predictor raw → *calibration wrapper* (Platt/Isotonic/Regime-aware, fallback jerárquico CONTEXT→HORIZON→REGIME→STRATEGY→GLOBAL→UNCALIBRATED) → *evidence gate* (NO_TRADE si uncalibrado o zona neutral) → predicción persistida → outcome → reward → dashboard live (uptime, no-trade rate, ECE, Brier, net edge, paper PnL, max DD, drift, calibrator version).
- Sin aprendizaje adaptativo en vivo. Refilt offline → artefacto JSON versionable → carga en siguiente arranque.

**Rosa Roja Engine (paralelo — challenger)**
- Orquestador cognitivo 25+ fases en `core/orchestration/rosa_roja/`.
- Mixture of Experts con gating por `(equipment_class, regimen)`.
- Aprendizaje bayesiano online por sensor (clave `namespace:series_id:regimen`), fallback a global, cold-start blend.
- Detección de anomalías ensemble v2.0 (7 detectores, voto ponderado Hampel adaptativo).
- Drift detection (Page-Hinkley + ADWIN), compliance HMAC-SHA256 NDJSON append-only.
- No se promociona a producción sin protocolo congelado.

**Servicio ML (FastAPI)**
- 40+ feature flags, runners batch/stream/CLI, warmup, GraphQL opcional.
- Arquitectura hexagonal (Ports & Adapters). 3600+ tests.

---

## Qué lo diferencia

| Capacidad | Detalle |
|-----------|---------|
| **Bayesiano online per-sensor** | Pesos se actualizan por observación sin retraining; clave `namespace:series_id:regimen`. |
| **MoE equipment-aware** | Routing condicionado a tipo de equipo × régimen (no global). |
| **Calibración con fallback explícito** | Jerarquía CONTEXT→HORIZON→REGIME→STRATEGY→GLOBAL→UNCALIBRATED; gate económico eliminado del veredicto (solo Brier). |
| **Evidence Gate** | UNCALIBRATED → NO_TRADE automático; zona neutral configurable; acción+motivo persistibles. |
| **Ciclos stateless + DB estado** | Paper bot re-ejecuta engine completo cada ciclo; upserts idempotentes; anti-feed_ended poisoning. |
| **Auditoría HMAC-SHA256** | Cada predicción firmada en NDJSON; verificación constant-time. |
| **Hexagonal** | Dominio puro (zero infra deps) + adapters intercambiables (SQL/Redis/Weaviate). |

---

## Stack mínimo

```
Python 3.10+ · FastAPI · NumPy · SciPy · scikit-learn · Redis 7 · MySQL
Opcional: LightGBM · XGBoost · Weaviate · SQL Server · MLflow · Prometheus
```



---

## Estructura relevante

```
iot_machine_learning/
├── core/orchestration/rosa_roja/      # Motor cognitivo (challenger)
├── domain/entities/market/            # Market: calibration, replay, prediction, evidence
├── infrastructure/adapters/market/    # Binance feed, paper runner
├── infrastructure/persistence/sql/zenin_market/  # Migraciones + repos
├── ml_service/                        # FastAPI app, runners, governance
├── infrastructure/ml/                 # Engines, MoE, anomaly, inference, optimization
├── scripts/zenin_paper_btc.py         # Entry point paper bot (wiring completo, pendiente primer experimento live)
├── benchmarks/                        # Latencia Rosa Roja, NAB, ensemble forense
└── tests/                             # 3600+ (unit, integration, load, stress, property)
```

---

## Estado

- Market suite: **426 passed** (calibración, feed, runner, evidencia, gate).
- Rosa Roja: **159 passed** (desde ambas raíces, sin hacks de path).
- Deuda documentada: 24 fallos legacy `integration/cognitive/*` (fases nunca implementadas), benchmarks P99 sensibles a carga de escritorio.