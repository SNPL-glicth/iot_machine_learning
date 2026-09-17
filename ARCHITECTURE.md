# ZENIN MARKET — Architecture Rules

Contrato arquitectónico de ZENIN Market. Guardrail: todo cambio debe
respetar estas reglas; los tests de `tests/unit/market/test_architecture_gate.py`
las verifican permanentemente.

1. **IoT SQL Server permanece aislado** — su persistencia (pymssql) y sus
   entidades (`domain/entities/results/`, `iot/`, ...) no se modifican ni
   se acoplan al dominio Market.
2. **Market usa MySQL** (`zenin_market`) — migraciones SQL idempotentes;
   nunca escribir SQL Server para Market.
3. **Domain no importa infraestructura** — `domain/entities/market/**` no
   importa pymysql/sqlalchemy/redis/alpaca/binance/weaviate. Solo stdlib
   y el propio dominio.
4. **Providers implementan `MarketDataProvider`** (Fase 4) — el dominio
   conoce el port, no la implementación.
5. **Capabilities determinan qué features existen** — `ProviderProfile`
   decide qué observaciones/features son lícitas; no se asume nada que
   el perfil no declare.
6. **Observation → Prediction → Outcome → Evaluation → Reward** — el ciclo
   fluye en ese orden; la evaluación requiere Outcome del mismo símbolo y
   el mismo horizonte.
7. **Prediction no conoce Outcome futuro** — su creación es ciegua al
   desenlace; el Outcome solo se vincula en `WAITING_OUTCOME`.
8. **Reward solo nace de EVALUATED** — la transición `EVALUATED -> REWARDED`
   es la única vía; PENDING/WAITING_OUTCOME/INVALIDATED jamás producen
   reward.
9. **Replay, Shadow y Live comparten dominio** — el mismo pipeline en
   memoria; solo cambia la fuente de datos.
10. **Ningún provider contamina el dominio** — los datos del provider se
    mapean a `MarketObservation`/`Prediction` en capas adapter, nunca en
    `domain/`.
11. **Tests nuevos + regresión existentes son obligatorios** — antes de
    cada fase: suite Market + suite IoT pre-existente sin nuevas fallas.
12. **Archivos >180 líneas requieren revisión** — nueva lógica debe
    modularizarse cuando exista una responsabilidad separada real; no
    dividir solo por cumplir un número.
13. **No secrets en repository** — credenciales solo en `.env` (gitignored);
    `.env.example` sin valores reales.
14. **No refactors no relacionados durante una fase** — el alcance de cada
    fase es cerrado; refactors ajenos se agenda aparte.
15. **Soberanía Absoluta del Motor Matemático (ZENIN Master Equation)** — Zephyr
    es un ejecutor ciego, sordo y mudo; el `MasterEquationOrchestrator` es la
    única autoridad de decisión (`ExecutionPlan`). Zephyr jamás altera,
    sobreescribe ni introduce heurísticas discrecionales a la orden emitida.
16. **Segregación Estricta de Secretos y Configuración Canónica** — Toda
    configuración vive en `zephyr/config/`. Los archivos JSON que van a disco o
    telemetría deben pasar por `to_safe_dict()` o `to_public_dict()`. Jamás
    filtrar API keys a logs o clientes WebSocket.
17. **Gobernanza Modular Estricta (≤180 Líneas)** — Todos los módulos de producción
    en `zephyr/` deben residir en sus subpaquetes especializados (`adapters/`,
    `config/`, `engines/`, `execution/`, `master_engine_adapter/`, `models/`,
    `resilience/`, `risk/`, `runners/`, `scripts/`, `telemetry/`) y cumplir
    estrictamente con $\le 180$ líneas de código por archivo.

## Colisión de nombres (regla crítica)

`domain/entities/prediction.py` es un **módulo** legacy de IoT. El dominio
Market vive en `domain/entities/market/prediction/` (paquete). Si alguien
crea de nuevo `domain/entities/prediction/` (paquete) o cambia los
re-exports de un `__init__.py` de `entities`, el módulo legacy queda
oculto y `IoT Prediction != ZENIN Prediction` se rompe. Lo verifican los
tests de arquitectura.