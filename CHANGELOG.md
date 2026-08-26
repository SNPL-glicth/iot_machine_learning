# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [ZENIN Market — Fase 9.5: Statistical Reality Check] - 2026-08-20

### Diseño (la pregunta de la fase)
FASE 9.4 dejó una pregunta abierta: el edge DECLARADO no sobrevive al
TEST, y el +0.03% de AMD (9.3) puede ser fluctuación. 9.5 somete el
pipeline completo a significancia estadística sobre los outcomes
persistidos (store; engine NO se re-corre, selección con SOLO TRAIN):

    1. Permutación temporal — predicciones intactas, outcomes barajados
       entre timestamps. La dirección predicha se recupera de
       (direction_correct, move) y se re-evalúa contra el movimiento
       barajado: bajo el nulo E[PnL] = 0 antes de costos con las
       distribuciones marginales intactas.
    2. Bootstrap por ventana — IC 95% percentil (2000 remuestras) para
       accuracy, net, sharpe y maxDD del agregado real.
    3. Diferencia vs baseline — ZENIN − (Naive | EMA) por ventana con
       IC 95%; si el IC cruza cero no hay superioridad demostrable.
    4. Permutación del ganador — destruir SOLO la asociación
       contexto → experto ganador (ganador aleatorio por ventana):
       ¿la selección aporta? (FASE 9.3 dijo que la señal vive ahí).
    5. Bootstrap por experto — IC 95% para accuracy, mean_reward y ECE
       de cada estrategia sobre sus filas recompensadas.
    6. Walk-forward agresivo (--val-days) — entrenamiento → validación
       → test; la validación ELIGE el modo de selección por ventana;
       el TEST jamás se usa para decidir.

### Added
- **`domain/entities/market/replay/significance.py`** (módulo puro):
  `recover_predicted_direction`, `permuted_net_returns`,
  `PermWindow`/`permutation_test` (agregado n-ponderado, p de dos
  colas), `WindowRecord`/`block_bootstrap` (+ `weighted_acc`,
  `weighted_net`, `pooled_sharpe`, `window_cumsum_maxdd`),
  `difference_ci`, `random_winner_test`, `bootstrap_expert_metrics`
  (+ `BootstrapCi` con `crosses_zero`, `ExpertMetricsCi`).
- **`scripts/zenin_reality_check.py`** — runner de solo lectura con las
  6 pruebas por símbolo (reporte con modo/seed/nuls/IC/p), parámetros
  auditables (--seed, --n-permutations, --n-boot, --n-random-winner,
  --val-days, --risk-aversion, --min-n, --expert-max-rows).
- **Tests**: `tests/unit/market/test_significance.py` (21): dirección
  recuperada, nulo de la permutación (sin señal → p alto; con señal
  real → el edge NO es explicable por la secuencia temporal),
  determinismo por seed, agregados n-ponderados, CI del block
  bootstrap (serie constante → CI puntual), cruce de cero,
  diferencia vs baseline (mejor consistente → no cruza; empate →
  cruza), ganador aleatorio (varias ventanas: seleccionar acierta
  mejor que azar; sin señal → indistinguible), bootstrap por experto.
  Suite: **371 passed, 1 skipped**; ruff y mypy limpios en módulo y
  script.

### Resultados reales (2026-08-20, seed 42, costos 12 bps / 24 bps)
**Permutación temporal** (500): en TODOS los símbolos y modos el edge
real NO supera al nulo (p ≥ 0.12; NVDA soft 0.028 y AAPL hard 0.024
son SIGNIFICATIVOS EN CONTRA: el resultado real es PEOR que el azar).
**AMD +0.03% es fluctuación**: IC95% de hard_max [−0.36%, −0.05%].
**Bootstrap**: todos los IC de net ≤ 0 o rozando cero; los IC de
sharpe son negativos en todos los modos.
**vs baseline**: TODOS los IC cruzan cero (ZENIN no es superior a
Naive ni a EMA en ningún instrumento).
**Ganador aleatorio**: p ≥ 0.11 en todo — seleccionar el mejor experto
por contexto rinde igual que elegirlo al azar en edge REALIZADO.
**Bootstrap por experto**: la accuracy SIGNIFICATIVA existe (NVDA:
baseline 66.7% IC [57, 76], ema-crossover 59.5% [54, 65], momentum
59.0% [54, 64]) — pero no paga costos.
**Walk-forward agresivo (val 7d, 149 ventanas)**: el modo elegido por
validación (chosen) tampoco sobrevive; el test nunca se usó.

### Veredicto de la fase (la cuenta que lo explica todo)
NO hay evidencia estadística de edge neto operativo en ningún
instrumento, bajo ningún modo de selección ni con validación. La
accuracy > 50% significativa es real pero insuficiente por una cuenta
exacta: el edge bruto de una apuesta direccional es
`(2·acc − 1) × E|move|`; con acc 60% y E|move| 1h ≈ 0.3-0.5%, el edge
bruto es +0.06/+0.10% — menor que los 12 bps de costo y, además,
asumido a toda la varianza del |move|. Necesitaría acc ≥ 65% CON
E|move| ≥ 0.72% (acciones) para que el bruto solo cubra costos.

Consecuencia para el roadmap: 9.5 responde por qué el 9.3 era una
pista, no una estrategia. La mejora NO está en más selección/mezcla:
está en acc decididamente > 60% (o en operar solo cuando
`(2·acc−1)·E|move| − costo − riesgo > 0`, que es el NO TRADE de 9.4
llevado a su forma exacta). El presupuesto experimental queda
científicamente justificado en NO operar todavía.

## [ZENIN Market — Fase 9.4: Adaptive Expert Selection] - 2026-08-20

### Diseño (la hipótesis de la fase)
9.3 demostró que la señal de ZENIN está en la SELECCIÓN del mejor
experto (MoE hard-max) y que la mezcla suave la diluye — pero argmax →
operar a ciegas amplifica el ruido de selección (Experto A 0.61 vs
Experto B 0.39 → "ganó A" → 100% A). 9.4 introduce selección ADAPTATIVA
CONTROLADA con tres modos y, sobre todo, la decisión de NO TRADE:
una inteligencia de trading debe poder decir "no tengo suficiente edge".

El score deja de ser accuracy (que ya engañó: Naive 62.2% en NVDA y aún
pierde dinero) y pasa a:

    score = expected_net × calibration_quality × evidence_strength
    expected_net = expected_return − costo − risk_aversion × PnL_std

La selección usa SOLO información TRAIN (sin lookahead: al inicio de
cada ventana el selector conoce únicamente lo ocurrido hasta train_end)
y se evalúa sobre los MISMOS outcomes TEST persistidos de las corridas
9.1/9.2 (el engine NO se re-corre).

Árbol de decisión por contexto (experto × régimen × horizonte):

    sin expertos con muestra                -> HOLD
    mejor edge neto <= umbral               -> HOLD / NO TRADE
    hard_max  (n, historial, margen 2º)     -> ganador único
    selective (calidad >= mejor × ratio,    -> re-pesaje de sobrevivientes
                tope de expertos)
    soft      (fallback)                    -> softmax sobre todos

La escalera es defensiva: cada modo cae al siguiente si su guardrail
falla.

### Added
- **`domain/entities/market/adaptation/selection.py`** (módulo puro):
  - `SelectionMode` (soft / selective / hard_max) y `SelectionConfig`
    (temperature, min_ratio, max_experts, min_n, min_history_days,
    min_margin, min_expected_net, risk_aversion).
  - `expert_net_scores` — ExpertScores → `ExpertNetScore`:
    `expected_net = expected_return − cost_model.total() −
    risk_aversion × risk_std`; `score = expected_net ×
    (1 − min(calibration, 1)) × min(n/min_n, 1)`. La accuracy NO entra
    al score (métrica secundaria/guardrail).
  - `select_weights` — escalera hard→selective→soft + puerta NO TRADE;
    `SelectionResult` (mode, weights, winner, decision trade/hold,
    reason) totalmente auditable.
- **`ExpertScore.risk_std`** — desviación del PnL direccional por
  contexto (STDDEV de `±|move|` según `direction_correct`), agregada en
  `expert_performance` (FASE 8) y mapeada por el `PerformanceAnalyzer`.
- **`scripts/zenin_selection.py`** — runner de solo lectura sobre el
  store: por ventana, scores TRAIN (régimen del window) → net scores →
  selección por modo → portafolio sobre outcomes TEST. Reporte por
  símbolo: ventanas, HOLDs, n, acc, edge declarado (exp, desde TRAIN),
  net, sharpe, maxDD. Importa `SelectionMode`, `SelectionConfig`,
  `expert_net_scores`, `select_weights` en `adaptation/__init__.py`.
- **Tests**: `tests/unit/market/test_selection.py` (22): score neto y
  sus factores, NO TRADE (sin evidencia, sin edge, umbral custom),
  softmax (suma 1, preferencia, temperatura), selective (ratio, tope,
  fallback sin evidencia), hard_max (guardrails, margen, fallbacks),
  validación de config, auditabilidad. Suite: **350 passed, 1 skipped**;
  ruff y mypy limpios en los módulos nuevos (el `ExpertScoreLike`
  Protocol se descartó por un quirk de inferencia de mypy en scripts
  grandes: selection.py tipa directo contra `ExpertScore`, mismo
  paquete).
- **Matriz real (2026-08-20)** — ventanas 14/7/7, costos 12 bps
  (acciones) / 24 bps (BTC), riesgo_aversion 0.1, min_n 10:

| Símbolo | modo | hold/win | n | acc | exp | net | sharpe | maxDD |
|---|---|---|---|---|---|---|---|---|
| NVDA | soft | 2/8 | 178 | 53.2% | +0.01% | −0.10% | −0.27 | −0.62% |
| NVDA | selective | 2/8 | 178 | 52.0% | +0.53% | −0.09% | −0.10 | −0.48% |
| NVDA | hard_max | 2/8 | 178 | 52.0% | +0.53% | −0.09% | −0.10 | −0.48% |
| AMD | soft | 1/8 | 216 | 54.1% | +0.02% | −0.12% | −0.34 | −0.90% |
| AMD | selective | 1/8 | 216 | 52.4% | +0.47% | −0.18% | −0.20 | −1.63% |
| AMD | hard_max | 1/8 | 216 | 52.4% | +0.47% | −0.18% | −0.20 | −1.63% |
| AAPL | soft | 5/8 | 73 | 56.7% | +0.00% | −0.14% | −0.80 | −0.69% |
| AAPL | selective | 5/8 | 73 | 66.7% | +0.21% | −0.09% | −0.22 | −0.61% |
| AAPL | hard_max | 5/8 | 73 | 66.7% | +0.21% | −0.09% | −0.22 | −0.61% |
| BTC-USD | soft | 93/103 | 1680 | 50.5% | +0.01% | −0.24% | −1.25 | −2.44% |
| BTC-USD | selective | 93/103 | 1680 | 53.6% | +0.45% | −0.25% | −0.46 | −2.46% |
| BTC-USD | hard_max | 93/103 | 1680 | 53.6% | +0.45% | −0.25% | −0.46 | −2.46% |

### Hallazgos (lo que 9.4 revela)
- **NO TRADE funciona y es la herramienta más potente de la fase**:
  BTC declina 93 de 103 ventanas (90%), AAPL 5 de 8. ZENIN ya no está
  obligado a escoger: con información TRAIN sin edge neto esperado, no
  opera. Es exactamente el comportamiento que pedía la fase.
- **selective ≡ hard_max en la corrida real (degeneración del config)**:
  con min_ratio=0.5 y max_experts=2, el filtro por calidad colapsa el
  conjunto de sobrevivientes al ganador → el softmax les da ~1.0. Los
  modos son config-distintos, no implementación-distinta: para ver el
  modo "selective" de verdad hay que admitir más expertos
  (min_ratio más bajo o max_experts mayor). Es un knob, no un bug.
- **El edge declarado NO sobrevive al TEST**: soft declara ~+0.01% y
  selective/hard ~+0.21/+0.53% (desde TRAIN), pero el net realizado es
  negativo en los 4 símbolos con cualquier modo. El `expected_return`
  que los expertos declaran en TRAIN no transfiere al PnL de TEST. Esa
  es la pregunta central que 9.5 (permutation test, bootstrap) debe
  responder: ¿hay relación temporal predicción↔outcome o estamos
  midiendo una propiedad estructural del dataset?
- **Más selectividad ≠ mejor net**: los tres modos terminan negativos;
  la elección del ganador mejora accuracy y edge declarado, no el PnL
  realizado. La señal existe (9.3) pero no es operativamente rentable
  bajo estos costos.

## [ZENIN Market — Fase 9.3: Matriz de Ablations] - 2026-08-20

### Diseño (la pregunta de la fase)
¿De dónde salió el edge bruto de FASE 9.2? No se agrega nada: se
REMUEVE un componente de ZENIN a la vez y se mide sobre los MISMOS
outcomes de las corridas reales (predicciones persistidas en el store;
el engine NO se re-corre — solo cambia el vector de pesos por ventana).
Matriz 7 × 4: Naive/Momentum/EMA crossover (experto solo) + ZENIN sin
memoria / sin régimen / sin MoE / completo × NVDA/AMD/AAPL/BTC-USD.

**Descubrimiento metodológico de la fase**: `outcome_return_realized`
es el movimiento FIRMADO del mercado — idéntico para todos los expertos
en el mismo timestamp; la estrategia solo aporta `direction_correct`.
Por eso el edge realizado NO puede medirse como "retorno del
portafolio" (todas las ablaciones daban lo mismo: es el drift del
mercado): el PnL honesto de la apuesta direccional es
`acierta → +|move|, falla → −|move|`, ponderado por pesos y menos costo.
Ese es el edge que discrimina, junto con la accuracy ponderada.

### Added
- **`domain/entities/market/replay/ablation.py`** (módulo puro, sin SQL):
  - `parse_version_reason` / `active_versions_by_window` — reconstruye la versión activa de cada ventana desde la cadena append-only de `model_versions` (reason "wf {symbol} W{index}"): última del símbolo con índice ≤ W; si no existe, la heredada al iniciar la corrida; si el símbolo nunca adaptó, la última global. Maneja re-corridas (la versión de mayor version_id gana por ventana).
  - `ablation_weights` — los 7 vectores: baselines {experto: 1.0}; sin memoria → uniforme sobre los expertos del contexto; sin régimen → contexto global `*|-|hhss` (o uniforme); sin MoE → contexto resuelto + hard max sobre `reward_adjusted` (empate por orden alfabético); completo → contexto exacto del régimen con fallback al global (igual que `evaluate_window`).
  - `portfolio_net_returns` — PnL direccional por timestamp; `ablation_window_stats` / `AblationWindow` — n, accuracy ponderada, edge declarado (expected), PnL realizado bruto/neto, sharpe y maxDD; `aggregate_ablation` — agregados ponderados por n, sharpe de la serie pooled por timestamp, maxDD del acumulado por ventana (comparable entre símbolos); `render_ablation_matrix` — bloques por símbolo + desglose por régimen.
- **`scripts/zenin_ablation.py`** — runner de solo lectura sobre el store (sin engine): ventanas iguales a las corridas reales (14/7/7), versión activa por ventana, outcomes reales por timestamp, matriz completa.
- **Tests**: `tests/unit/market/test_ablation.py` (23: parse de reason, reconstrucción de versiones con re-corridas, los 7 vectores de pesos, PnL direccional, sharpe/maxDD, agregados, render). Suite: **328 passed, 1 skipped**; ruff y mypy limpios en el módulo nuevo.
- **Matriz real (2026-08-20)** — acc / net después de costos (12 bps acciones, 24 bps BTC):

| Ablación | NVDA | AMD | AAPL | BTC-USD |
|---|---|---|---|---|
| Naive | 62.2% / −0.07% | 51.5% / −0.04% | 53.7% / −0.08% | 50.2% / −0.24% |
| Momentum | 54.2% / −0.07% | 48.2% / −0.17% | 53.7% / −0.09% | 48.2% / −0.24% |
| EMA crossover | 51.8% / **−0.05%** | 50.1% / −0.11% | 59.7% / **−0.07%** | 47.9% / −0.24% |
| ZENIN - memoria | 53.7% / −0.10% | 50.4% / −0.11% | 52.4% / −0.10% | 49.6% / −0.24% |
| ZENIN - régimen | 53.7% / −0.10% | 50.4% / −0.11% | 52.4% / −0.10% | 49.6% / −0.24% |
| ZENIN - MoE | **68.4%** / −0.09% | **60.0%** / **+0.03%** | **62.1%** / −0.10% | **54.1%** / **−0.23%** |
| ZENIN completo | 52.6% / −0.09% | 51.0% / −0.10% | 51.2% / −0.10% | 49.8% / −0.24% |

- **Corridas**: NVDA/AMD/AAPL n=248–251; BTC-USD n=17.147 (103 ventanas).

### Hallazgos (lo que la matriz revela)
- **La señal de ZENIN vive en la SELECCIÓN del mejor experto por contexto (MoE), no en la memoria, no en el régimen, no en la mezcla suave.** El MoE gana en accuracy en los 4 instrumentos (68.4% NVDA, 60.0% AMD, 62.1% AAPL, 54.1% BTC) y es el MENOS negativo en net (única célula positiva del mundo: AMD +0.03%, sharpe +0.04 — marginal).
- **Memoria ≡ régimen ≡ casi-completo**: los pesos reales de las versiones activas son ~uniformes (0.15–0.36 alrededor de 0.25): el guardrail (max_change) y el softmax diluyen la señal hasta casi uniforme. El contexto global `*|-|3600s` ni siquiera existe en las versiones. La ingeniería actual NO traduce la señal en pesos.
- **El escenario 😂 es real**: ZENIN completo (net −0.09/−0.10/−0.10/−0.24%) hace mucha ingeniería para conseguir lo mismo o menos que EMA crossover (−0.05/−0.11/−0.07/−0.24%) y que Naive (62.2% acc en NVDA — la persistencia direccional manda en sesiones intradía).
- **Nadie supera los costos de forma consistente**; BTC es un drenaje para cualquier esquema (24 bps, sharpe ≈ −0.5 incluso en el mejor).
- **Señal para 9.4**: SIMPLIFICAR hacia el ganador-único (hard max del MoE) — el experimento dice que la mezcla suave con guardrail es el cuello de botella, no la selección.

## [ZENIN Market — Fase 9.2: Edge Después de Costos] - 2026-08-19

### Diseño (la prueba más cruel)
"¿Después de pagar por intentar hacerlo, todavía ganó?" Hasta 9.1 los
números del reporte eran **gross**: `reward_execution_costs` viene de
`RewardConfig.execution_costs`, una tasa plana que FASE 7 nunca configuró
(default 0.0). 9.2 introduce el costo por instrumento y pregunta por
predicción: `expected_return` (lo que el modelo declara) − costo esperado
= `expected_net_return`; y contra lo que el mercado PAGÓ (`outcome_return_realized`).
El EDGE se clasifica en una escalera que no premia señales muertas:

`gross_negative` < `cost_negative` (señal bruta positiva que los costos
matan: descubrimiento brutalmente útil) < `risk_negative` (positivo pero
inconsistente) < `cost_positive` (positivo sin ajuste de riesgo) <
`risk_adjusted_positive` (positivo Y sharpe ≥ 0.5).

Un resultado `cost_negative` NO es un fracaso: es capacidad predictiva
real insuficiente para superar los costos del mercado. Esa es
exactamente la información que el presupuesto experimental necesita.

### Added
- **`domain/entities/market/costs.py`** (módulo puro, sin SQL):
  - `CostModel(spread_bps, slippage_bps, commission_bps)` — costo total en bps (0.01%) y como fracción, `net(gross)` = bruto − costo;
  - `COST_PROFILES` — acciones (NVDA/AAPL/AMD/MSFT/QQQ/SPY): 4 + 5 + 3 = **12 bps**; cripto (BTC-USD/ETH-USD): 2 + 2 + 20 = **24 bps** (ejemplo de la fase: +0.12% bruto − 12 bps = 0.00% neto);
  - `classify_edge(gross, net, sharpe=None, sharpe_threshold=0.5)` — la escalera.
- **`domain/entities/market/replay/walk_forward.py`**:
  - `EdgeMetrics` (expected/realized × gross/net + cost_bps + n) en `HorizonEval.edge`;
  - `evaluate_window(..., cost_model=None)` — edge ponderado por los pesos de la versión activa; sin `cost_model` se comporta exactamente como 9.1;
  - `WfRow.cost_bps/sharpe/edge_class` + `realized_gross/realized_net` a nivel ventana;
  - `render_wf_report` — línea `edge` por ventana (`+0.12% -> -0.00% [cost_negative sharpe -0.23]`), línea `exp`/`real` por horizonte y **EDGE agregado** (bruto → neto, clasificado) en el resumen.
- **`infrastructure/persistence/sql/zenin_market/market_prediction_repository.py`** — `expert_performance` agrega `expected_return`, `realized_return` y `execution_costs` (AVG); **`domain/entities/market/adaptation/expert_scores.py`** — `ExpertScore` expone los 3 campos (defaults 0.0, compatibles con FASE 8/9.1).
- **`scripts/zenin_walk_forward.py`** — cost_model = `COST_PROFILES[symbol]` (override `--cost-bps`); sharpe por predicción del TEST (outcomes reales, netos de costos, media/desviación muestral); clasificación del edge por ventana y del agregado.
- **Tests**: `tests/unit/market/test_costs.py` (14: matemática de costos, perfiles, escalera de clasificación, umbral de sharpe) + edge en `test_walk_forward.py` (ponderación, `realized_net < 0` tras costos, render del edge). Suite: **305 passed, 1 skipped**; ruff y mypy limpios en los módulos nuevos.
- **Corridas reales (2026-08-19, 12 bps acciones)**: NVDA 1h n=652 **+0.07% → −0.05% [cost_negative, sharpe −0.32]** · AMD 1h n=580 **+0.03% → −0.09% [cost_negative, sharpe −0.52]** · AAPL 1h n=760 **+0.03% → −0.09% [cost_negative, sharpe −0.64]**. Las tres: bruto positivo (la señal de 9.1 es real), neto negativo (los costos la matan). BTC-USD 1h (24 bps) pendiente (~20 min de cómputo).

## [ZENIN Market — Fase 9.1: Walk-Forward Multi-Instrumento] - 2026-08-19

### Diseño (pregunta de la fase)
"¿Hay realmente una señal predictiva que sobreviva fuera de muestra, entre
instrumentos, regímenes, costos y tiempo?" La Fase 9.1 responde la primera
mitad con **walk-forward real sobre el store**: cada ventana separa TRAIN
(aprende solo lo conocido hasta t, con la cadena completa de FASE 8:
resolver → expert_performance → proposer → guard → model_versions) y TEST
(evalúa el modelo de la versión activa sobre outcomes que el TRAIN jamás
vio). El TEST no alimenta al TRAIN; el solapamiento entre ventanas es
intencional (rolling origin) y la prevención de fuga es por `since/until`
en las consultas, no por el splitter.

### Added
- **`domain/entities/market/replay/walk_forward.py`** (módulo puro, sin SQL):
  - `wf_windows(candles, train_seconds, test_seconds, step_seconds)` — ventanas rolling con `n_train` variable (origen avanza, el tramo test nunca solapa con el TRAIN de esa ventana);
  - `window_regime(candles)` — régimen de la ventana (clasificación del tramo TRAIN completo, requiere ≥21 velas);
  - `ModelMetrics` / `HorizonEval` / `WfRow` / `weighted_model_metrics` — recompensa y precisión del modelo ponderadas por los pesos de la versión activa;
  - `evaluate_window(scores, weights_by_context, regime=...)` — evalúa el TEST con los pesos del contexto exacto `*|régimen|hhss` y cae al contexto global `*|-|hhss`; si el régimen de la ventana no tiene muestra, evalúa con todas las del tramo;
  - `render_wf_report(rows, symbol, interval_label)` — reporte por ventana (régimen, acc/reward/n del modelo) + agregado por régimen.
- **`scripts/zenin_walk_forward.py`** — runner contra MySQL real:
  - TRAIN por ventana: engine × expertos sobre el tramo con **ventana pre-calentada** (el fragmento arranca `lookback × intervalo` antes del inicio y termina `horizonte + intervalo` después del final, de modo que toda predicción del tramo se emite y vence dentro del feed — sin esto, un lookback de 20 velas consumía 3 de 5 sesiones del TEST); persiste → resuelve contra el timeline completo → `expert_performance` acotado al tramo → proposer/guard → `create_version` si acepta;
  - TEST por ventana: mismo engine → resuelve → `evaluate_window` con los pesos de la versión activa → `WfRow`;
  - usa el engine puro (sin `LiveFeed`/`LiveShadowRunner`): el walk-forward reproduce datos históricos completos; la lógica de gaps/degradación es del shadow live.
- **`infrastructure/persistence/sql/zenin_market/adaptation_repository.py`** — `proposal_history` acepta `proposal_id_prefix` (el store compartido acumula corridas reales y el trail TEST- quedaba fuera del LIMIT).
- **`infrastructure/persistence/sql/zenin_market/market_prediction_repository.py`** — `expert_performance` ya no descarta filas con `data_status = NULL` (`(data_status IS NULL OR data_status <> 'stale')`): las predicciones sintéticas/persistidas por entidad se agregaban en silencio.
- **`domain/entities/market/adaptation/proposer.py`** — `propose_vector` sin pesos registrados para un contexto parte de **pesos uniformes sobre los expertos con muestra** (v1 bootstrap, suma = 1): antes el "actual" por defecto (0.25 por experto) sumaba 0.5 y la redistribución no podía alcanzar la suma 1.
- **`data/market/AMD_1h.csv`** — dataset nuevo (5.073 velas, 2023–2026).
- **Tests**:
  - `tests/unit/market/test_walk_forward.py` (15): contigüidad TRAIN/TEST, origen rolling, mínimo de entrenamiento, régimen de ventana, media ponderada, fallback de contexto, render;
  - `tests/integration/market/test_walk_forward_circuit.py` (2): flujo completo TRAIN→resolver→proposer→guard→create_version→TEST→evaluate_window contra MySQL (momentum acierta 20/20, naive falla 0/20: el guard estadístico tiene algo que decidir).
- **Corridas reales (2026-08-19)**: NVDA 1h 8 ventanas n=652 acc 54.7% reward +0.2853 (7/8 positivas) · AMD 1h 8 ventanas n=312 acc 51.2% +0.2064 (7/8) · AAPL 1h 8 ventanas n=328 acc 54.0% +0.2594 (6/8) · BTC-USD 1h 103 ventanas n=43.860 acc 49.7% +0.1640 (103/103). Versiones v31/v32/v33 creadas por aceptaciones reales (audit trail intacto).

### Fixed
- **Fuga de evaluación en TEST**: el engine exige `lookback + 1` velas antes de emitir; con datos de sesión (7 velas/día) un lookback de 20 consumía 3 de 5 días del TEST (n≈10 por ventana). La ventana pre-calentada con datos anteriores al tramo (sin lookahead) recupera la cobertura completa (n≈100+ por ventana).
- **Ventanas sin evaluación**: las versiones previas sin contexto 3600s (bootstrap FASE 8 sobre 1m) dejaban las ventanas 1h sin pesos → `evaluate_window` no emitía `HorizonEval`. Con la primera adaptación 1h real los contextos existen.

## [ZENIN Market — Fase 8: Auditable Adaptation (Propuesta ≠ Cambio)] - 2026-08-17

### Diseño (regla de piedra)
ZENIN **nunca aprende de su propia predicción sin observar el outcome
externo**. El pipeline de adaptación es: Historial (solo outcomes reales)
→ PerformanceAnalyzer → ExpertScores → WeightProposer → **WeightProposal**
(se imprime y se registra ANTES de actuar, no toca el modelo) →
AdaptationGuard → ACCEPT/REJECT → `model_versions` (append-only).
"¿Por qué ZENIN le dio más peso a Momentum?" es una consulta, no una
inferencia: toda propuesta —aceptada o rechazada— queda guardada con sus
chequeos y su razón.

### Added
- **`domain/entities/market/adaptation/expert_scores.py`** — `PerformanceAnalyzer` + `ExpertScore` por (experto, régimen, horizonte): `reward_adjusted = mean_reward × (1 − 0.5 × min(calibración, 1))`. La precisión **no** entra al score: entra al guardrail estadístico (Wilson). El score es deliberadamente simple y defendible.
- **`domain/entities/market/adaptation/proposer.py`** — `WeightProposer` + `WeightProposal` (tarjeta inmutable con expert/regime/horizon/current/proposed/reward/calibration/n/reason/parent_version):
  - softmax sobre `reward_adjusted` por contexto, con `propose_vector` (vector completo por contexto: suma = 1, |Δ| ≤ max_change, piso min_weight — ningún experto desaparece);
  - los tres constraints se resuelven por redistribución iterativa determinista; si no hay solución factible se conservan los pesos actuales (no cambiar es seguro);
  - la razón es una cadena generada con los números reales ("decrease under bear/900s: reward_adjusted -0.6562 (reward -0.8750, cal 0.50, acc 0.0%, n=11)").
- **`domain/entities/market/adaptation/guard.py`** — `AdaptationGuard` + `GuardResult` con los 10 chequeos pedidos (cada uno auditable: nombre + ok + detalle):
  min_n · suficiente historial (días distintos) · fuente limpia (solo rewarded, sin INVALIDATED/STALE) · reward válido · **Wilson lower bound > 0.5** (mejora estadísticamente razonable: 100% con n=6 no es "seguro") · |Δ| ≤ max_change · suma de pesos = 1 · ningún experto < min_weight · versión anterior preservada (parent_version). Render ✓/✗ ACCEPT/REJECT.
- **`infrastructure/persistence/sql/zenin_market/migrations/003_create_model_versions.sql`** — tablas:
  - `model_versions`: snapshot inmutable por versión (weights JSON por contexto, calibration JSON, reason, parent_version_id, proposal_id, guard_checks, is_active) — crear versión desactiva la activa en la misma transacción;
  - `adaptation_proposals`: TODO lo que se consideró cambiar, aceptado o rechazado, con veredicto, motivo del rechazo, chequeos y cadena al parent.
- **`infrastructure/persistence/sql/zenin_market/adaptation_repository.py`** — `AdaptationRepository`: `record_proposal` (siempre se guarda, incluso REJECTED), `proposal_history` (filtros por experto/estado), `latest_version`, `create_version` (append-only + desactiva la activa), `list_versions`, `version`, mapeo puro `proposal_to_row`/`row_to_proposal`.
- **`infrastructure/persistence/sql/zenin_market/market_prediction_repository.py`** — `expert_performance`: rendimiento por (strategy, regime, horizon_seconds) con **solo filas rewarded y sin STALE** + días distintos por contexto (guardrail de historial).
- **`domain/entities/market/replay/engine.py`** — los expertos (FASE 8) califican su `prediction_id` con el nombre de la estrategia (`NVDA-momentum-<ts>-<horizon>`): múltiples expertos coexisten en el store sin pisarse. El baseline (FASE 7) conserva el id legacy: re-correr el fragmento sigue siendo un upsert idempotente.
- **`scripts/zenin_adapt_run.py`** — **ZENIN MARKET — ADAPTATION PROPOSAL MODE**:
  - cada experto corre su propio shadow live sobre el MISMO fragmento (predictor pluggable del engine) y persiste con `strategy=experto`;
  - PerformanceAnalyzer → WeightProposer → tarjeta de propuesta por contexto (formato pedido: Expert/Regime/Horizon/Current/Observed reward/Calibration/Sample size/Proposed/Reason) → AdaptationGuard (10 chequeos) → registro en `adaptation_proposals`;
  - si alguna propuesta pasa: `create_version` (v2 = v1 + aceptadas, parent link, razón, chequeos, snapshot de pesos y calibración) + VERSION DIFF + AUDIT TRAIL;
  - bootstrap: si no hay versión registrada, la primera corrida materializa v1 (pesos uniformes) para que la cadena parent esté completa.
- **Tests**:
  - `tests/unit/market/test_adaptation.py` (32): analyzer (el score castiga calibración rota; accuracy NO entra al score), proposer (softmax, max_change, min_n, piso, reason, flatten por contextos), guard (los 10 chequeos, acepta sano / rechaza cada violación, render), Wilson (n pequeño castigado, convergencia, monotonía).
  - `tests/integration/market/test_adaptation_circuit.py`: circuito real contra MySQL — propuesta sana → ACCEPT y rechazada → REJECT con motivo (`min_n; statistical`), veredictos persistidos con sus 9 chequeos, v1→v2 con parent preservado, `latest_version`/`list_versions`/`proposal_history`, roundtrip del mapping, cleanup TEST- con restauración de la versión activa real.

### Resultado (NVDA 1m real, 2026-08-17, zenin_market)
- 4 expertos (naive, momentum, ema-crossover, mean-reversion) × 156 predicciones sobre el mismo fragmento: **624 predicciones reales** que conviven con el baseline legacy (strategy-calificadas, sin pisarse).
- **24 propuestas evaluadas, 24 REJECTED, 0 cambios** — y eso es el sistema funcionando: momentum muestra 65–82% de acierto bajo bear/neutral, pero el guard rechaza con evidencia (Wilson LB 0.512–0.523 roza 0.5; naive/mean-reversion con 0–35% → LB ≤ 0.19) y con historial (1 día < 2).
- El modelo permanece **v1 (bootstrap, pesos uniformes) intacta**: `model_versions` guarda v12-bootstrap como activa y `adaptation_proposals` conserva las 48 consideraciones (2 corridas) con sus razones — "¿por qué no cambió nada?" es una consulta SQL, no una opinión.

## [ZENIN Market — Fase 7.5: Observability & Evaluation Dashboard] - 2026-08-16

### Added
- **`domain/entities/market/replay/calibration.py`** — calibración pura (sin SQL):
  - `bucket_calibration`: clasifica cada bucket de confianza como **OK** / **INSUFFICIENT** (n < umbral: no concluye) / **FAIL** (`⚠ CALIBRATION FAILURE` cuando |declarado − realizado| > tolerancia) — la pregunta clave: si ZENIN dice P=0.9, ¿acertó históricamente ~90% o ~55%?
  - `CalibrationThresholds(min_n, tolerance)` — el germen de los guardrails de FASE 8 (no concluir ni aprender con pocas observaciones, no aprender por un único resultado);
  - `CalibrationReport.ece` (Expected Calibration Error, ponderado por muestra) y `calibration_chart`: curva ASCII declarado (o) vs realizado (x) vs diagonal ideal (\\), `*` cuando coinciden.
- **`infrastructure/persistence/sql/zenin_market/market_prediction_repository.py`** — `overall_stats`: conteos por estado (predictions/evaluated/pending/invalidated/archived), hits, **Brier score**, magnitude error y reward total para el overview del tablero. Agregados MySQL (Decimal) normalizados a float/int.
- **`scripts/zenin_dashboard.py`** — **ZENIN MARKET — OBSERVABILITY & EVALUATION DASHBOARD** (solo lectura, append-only):
  - caja ASCII con OVERVIEW, HORIZON (acierto + reward por horizonte), CONFIDENCE (curva de calibración con flags ⚠) y REGIME;
  - bloque EVALUATION: ECE + Brier + reward total + lista de buckets rotos y buckets insuficientes (con el guardrail explícito);
  - curva de calibración ASCII; `--symbol`, `--days`, `--min-n`, `--tolerance` para ajustar los umbrales.
- **`infrastructure/adapters/market/live_shadow.py`** — fix de honestidad en la cuenta: `all_predictions` deduplica por `prediction_id`. El engine materializa los estados terminales (INVALIDATED `feed_ended`) dentro de `predictions`, así que sumar `invalidated` duplicaba 60 ids ("216 emitidas" era en realidad 156 predicciones distintas + 60 duplicados). El store ahora persiste exactamente las predicciones emitidas.
- **Tests**:
  - `tests/unit/market/test_calibration.py` (16): estados de bucket (incl. el caso P=0.9 con 0/6 → FAIL sin maquillar), ECE ponderado, guardrails de umbral, curva ASCII y render del tablero (función pura, sin MySQL, con ancho de caja verificado).
  - `tests/integration/market/test_market_persistence_circuit.py` — extensiones: `overall_stats` (3 predicciones / 1 evaluada / 1 en espera / 1 invalidada / 1 hit / Brier > 0) y buckets de la curva.

### Resultado (NVDA 1m real, 2026-08-16, zenin_market)
- La cuenta real es **156 predicciones** (96 evaluadas + 60 INVALIDATED `feed_ended`), no 216: el doble conteo de `all_predictions` quedó corregido y el store es coherente.
- El tablero expone la verdad del predictor baseline actual: **Brier 0.547** (peor que lanzar una moneda, 0.25) y **ECE 0.6012** — dice P(up)=0.0–0.4 y el alza ocurre el 75–100% de las veces; dice P=0.9 y acierta 0/6.
- **Moraleja honesta**: la precisión por horizonte (60–71%) NO implica señal: la calibración está rota en 7 de 10 buckets. Es exactamente lo que la FASE 7.5 existe para descubrir antes de tocar aprendizaje.
- El sistema sigue sin maquillar: INVALIDATED no recibe reward, `feed_ended` es terminal, los buckets con n < min_n no concluyen (guardrail).

## [ZENIN Market — Fase 7: Live Persistence & Outcome Tracking] - 2026-08-16

### Added
- **`infrastructure/persistence/sql/zenin_market/migrations/002_create_market_predictions.sql`** — tabla `market_predictions` (FASE 7):
  - una fila por predicción: snapshot de la observación (JSON) + predicción (entry, expected_return, probability_up, confidence, intervalo, régimen, estrategia, contexto);
  - columnas NULLables del desenlace: Outcome (`outcome_measured_at/final_price/return_realized`), Evaluation (`direction_correct/magnitude_error/within_interval/calibration_error`) y Reward (componentes + total);
  - `prediction_id` PK → upsert idempotente (re-correr un shadow jamás duplica); índices por (symbol, emitted_at), status, horizon, strategy.
- **`domain/entities/market/prediction/resolver.py`** — `OutcomeResolver` (dominio puro, sin I/O):
  - `PriceLookup` Protocol (`last_close(at_or_before) -> float | None`, sin mirar futuro);
  - espera el vencimiento del horizonte → construye `Outcome` contra el último cierre conocido → recorre el contrato `activate -> to_waiting_outcome -> evaluate -> issue_reward`;
  - sin precio aún → `still_waiting`; terminales (REWARDED/INVALIDATED/ARCHIVED) → `unchanged` (jamás se re-resuelven ni mutan);
  - regla de FASE 7: este componente NO aprende, solo materializa el desenlace.
- **`infrastructure/persistence/sql/zenin_market/market_prediction_repository.py`** — `MarketPredictionRepository`:
  - mapeo puro fila ↔ entidad (`prediction_to_row`/`row_to_prediction` + snapshot JSON de la observación para Candle/Trade/Quote/OrderBookSnapshot, testeado sin MySQL);
  - `save_prediction`/`save_batch` (upsert por `prediction_id`), `pending_outcomes` (status no terminal), `record`/`recent_records`;
  - `performance_history`: agregados por horizonte, estrategia, régimen, bucket de confianza (¿el P(up) declarado ≈ acierto real?) y día (¿cómo se comportó ZENIN en agosto?).
- **`domain/entities/market/replay/engine.py`** — micro-fix de honestidad: al agotarse el feed, las predicciones sin resolver pasan a INVALIDATED con `reason="feed_ended"` (estado terminal real en el store, en vez de quedar colgadas en PENDING); `result.predictions` refleja las copias invalidadas.
- **`infrastructure/adapters/market/live_fragment.py`** — helpers compartidos por los runners live (extraídos de `live_shadow_run.py`): `FragmentFeed`, `DropWindowsFeed`, `RESOLUTIONS`, `fragment_bounds`, `parse_drop`, `drop_windows` (hora de mercado, 09:30 = apertura), `fmt_ts`.
- **`scripts/live_persist_run.py`** — **ZENIN MARKET — LIVE PERSISTENCE (FASE 7, NO REAL MONEY, NO ADAPTIVE LEARNING)**:
  - pipeline completo: Live → ZENIN → Prediction → MySQL → OutcomeResolver → Outcome → Evaluation → Reward → Performance history;
  - aplica migraciones, persiste el lote (upsert idempotente) y resuelve pendientes del store contra el fragmento;
  - `RECORD`: el recuerdo de ZENIN por predicción resuelta (ej. `14:26:00 NVDA 60s P(up)=0.05 expected=-0.04% conf=0.50 actual=-0.10% direction=correct within=true reward=+1.2375`);
  - tablero `PERFORMANCE HISTORY` (horizontes, estrategias, régimen, calibración por bucket, serie diaria);
  - banner explícito `NO ADAPTIVE LEARNING (SOLO REGISTRA)` + próxima etapa (Reward history → Calibration → Expert/Strategy performance → MoE adaptation).
- **Tests**:
  - `tests/unit/market/test_outcome_resolver.py` (7): vencida+precio → ciclo completo a REWARDED; sin precio → espera; ACTIVE/WAITING_OUTCOME re-runs; terminales intactas; invalidada jamás recibe reward; tipo inválido rechazado.
  - `tests/unit/market/test_market_prediction_mapping.py` (10): round trips fila ↔ entidad en PENDING/REWARDED/INVALIDATED y snapshot JSON de los 4 tipos de observación.
  - `tests/integration/market/test_market_persistence_circuit.py` (2, salta sin MySQL): save → pending → resolver → save → record → history de extremo a extremo (ids `TEST-`, limpieza al final).

### Resultado (NVDA 1m real, 2026-08-16, MySQL zenin_market)
- 216 predicciones persistidas (upsert idempotente); migración 002 aplicada.
- Performance history real: 60s → 60.5% acierto, 300s → 70.6%, 900s → 70.8%; baseline 66.7% global; bucket P(up)=0.9 → 0/6 aciertos (dato de calibración que ZENIN usará en la fase de aprendizaje).
- ZENIN ahora recuerda: `P(up), expected return, confidence, actual return, direction, within interval, reward` por predicción resuelta.
- El engine deja INVALIDATED `reason=feed_ended` las que el fragmento no alcanza a resolver (contrato FASE 6 intacto: provider_gap sigue invalidando al emitir).

## [ZENIN Market — Fase 6: Live Shadow Mode] - 2026-08-16

### Added
- **`domain/entities/market/replay/clock.py`** — Protocol `Clock` + `LiveClock` (FASE 6): el engine depende de la abstracción, no de `time.time()`; `ReplayClock` (tiempo lógico) y `LiveClock` (tiempo real) implementan el mismo contrato monótono (`advance_to`, `ClockRollbackError`). Tests en `test_live_clock.py`.
- **`domain/entities/market/connection.py`** — `ConnectionState`: CONNECTED/DEGRADED/DISCONNECTED/RECONNECTING/RECOVERED con `is_healthy()` (condición 4: sin predicciones en contexto incompleto).
- **`domain/entities/market/prediction/prediction.py`** — `Prediction.invalidate(reason)` + campo `invalidation_reason` (ej. `"provider_gap"`); INVALIDATED jamás produce reward.
- **`infrastructure/adapters/market/live_feed.py`** — `LiveFeed` (condiciones 1/3/4):
  - mismo contrato que `HistoricalFeed` (`iter_events` → `MarketObservation` sin alterar el objeto);
  - `GapDetected(expected_timestamp, received_timestamp, gap_seconds)` visible;
  - `StateTransition` con historial: gap → DEGRADED; primer evento sano → RECOVERED → CONNECTED; `update_state` manual para websocket;
  - `symbol`/`state`/`gaps`/`transitions`/`source`/`gap_threshold` expuestos.
- **`infrastructure/adapters/market/live_shadow.py`** — `LiveShadowRunner` + `LiveShadowResult`:
  - pre-scan del feed (pasada seca) → ventanas degradadas `[expected, received + interval]` (contexto incompleto hasta que la vela recibida cierra);
  - predicciones emitidas sobre contexto incompleto → INVALIDATED `reason=provider_gap` **al emitir** (antes de cualquier resolución/reward);
  - resultados inmutables: `predictions` + `invalidated` + `gaps` + `transitions` + `invalidated_by_gap`.
- **`domain/entities/market/replay/engine.py`** — `ReplayEngineConfig.degraded_windows` (FASE 6, opt-in): vacío por defecto → el replay clásico no cambia; con ventanas, invalida al emitir. `_emit_predictions` ahora recibe `Clock` (no `ReplayClock`).
- **`scripts/live_shadow_run.py`** — **ZENIN MARKET — LIVE SHADOW MODE (NO REAL MONEY)**:
  - banner grande + fragmento de sesión (default 09:30 → 10:30 = primera hora de la primera sesión);
  - `--drop HH:MM:SS-HH:MM:SS` (repetible) simula pérdida de datos en hora de mercado;
  - imprime `GAP_DETECTED` (expected/received), transiciones de conexión y predicciones invalidadas `reason=provider_gap`;
  - **paridad Live Shadow vs Replay integrada** (misma secuencia por ambos caminos → timestamp/features/prediction/horizon idénticos);
  - `PERSISTENCE: OFF` explícito (condición 5: temporal; siguiente etapa Live → Prediction → MySQL → Outcome → Reward).
- **`tests/unit/market/test_live_shadow_parity.py`** — 12 tests:
  - paridad sin gaps (replay ≡ live shadow en campos core);
  - `LiveClock` ≡ `ReplayClock` para el engine;
  - `GAP_DETECTED` expected/received;
  - invalidación al emitir antes de cualquier reward (horizonte corto incluido);
  - la predicción afectada mantiene features/predicción idénticas al replay (solo cambia la honestidad);
  - DEGRADED → RECOVERED → CONNECTED visible; observaciones sin retagging (condición 1);
  - replay por defecto jamás produce `provider_gap`.

### Resultado (NVDA 1m real, 2026-08-16)
- Fragmento 13:30-14:30 UTC (primera hora de sesión): 216 predicciones (60m/5m/15m/1h), parity OK, sin gaps.
- Con caídas simuladas (09:50 y 09:55-09:56): 2 `GAP_DETECTED` (120s y 180s), 12 predicciones INVALIDATED `reason=provider_gap`, transiciones CONNECTED → DEGRADED → RECOVERED → CONNECTED registradas.
- Live shadow mantiene el pipeline del replay: la diferencia está en la fuente (LiveFeed) y el reloj (LiveClock), no en la lógica de ZENIN.

## [ZENIN Market — Fase 5.5: Benchmark & Validation] - 2026-08-16

### Added
- **`domain/entities/market/replay/baselines.py`** — predictors pluggables (Protocol `Predictor`, `PredictionSignal` con validación 0.05 ≤ prob ≤ 0.95 y lower ≤ expected ≤ upper):
  - `NaivePredictor` (martingale: prob=0.50, expected=0.0)
  - `MomentumPredictor` (scale configurable, lookback_override)
  - `EmaCrossoverPredictor` (fast=4, slow=12)
  - `MeanReversionPredictor` (span=12)
  - `RandomPredictor` (seed para reproducibilidad)
  - `BASELINES` tuple con instancias default
- **`domain/entities/market/replay/metrics.py`** — `MetricCollector` + `MetricKey` + `PredictionMetrics`:
  - `direction_accuracy = (TP+TN)/N` (arreglado: NO hits/N)
  - `precision/recall/F1`, `Brier/LogLoss/ECE` (10 bins), `MAE/RMSE`
  - `reward`, `max_drawdown` (curva cumsum), `Sharpe/Sortino`, `profit_factor`
  - `confidence_bucket` (6 buckets: 0-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.01)
- **`domain/entities/market/replay/regime.py`** — `MarketRegime` enum + `classify_window`:
  - Prioridades reordenadas: CRASH (señal ≤ −0.75 bajo EMA) → HIGH_VOL (vol_rel ≥ 0.010) → LOW_VOL (vol_rel ≤ 0.004) → TRENDING (|signal| ≥ 0.50) → RANGE
  - Umbrales fijos y documentados para clasificación determinista
- **`domain/entities/market/replay/benchmark.py`** — `TrainTestSplit` + `split_walk_forward` + `TrainedMomentumPredictor`:
  - Grid search lookback{5,10,20,40} × scale{1,2,4} sobre TRAIN
  - Walk-forward con train_seconds, test_seconds, step_seconds, min_train
- **`domain/entities/market/replay/engine.py`** — pluggable predictor:
  - `ReplayEngineConfig.predictor` override (si existe, usa `predictor.name` como strategy)
  - `latency_sample_every` → `ReplayRunResult.latency_ns` (tuple[int, ...])
- **`scripts/benchmark_zenin_matrix.py`** — benchmark completo:
  - Parte A: baselines en NVDA 1h (direction%, Brier, reward)
  - Parte B: walk-forward NVDA 1h (train 90d → test 15d, step 15d; TrainedMomentumPredictor vs naive/random)
  - Parte C: ZENIN PERFORMANCE MATRIX (régimen por tramos de ~200 velas; filas=régimen, columnas=horizonte, celdas=direction%)
- **`scripts/benchmark_replay_capacity.py`** — capacity test:
  - Feeds sintéticos 1K/5K/10K velas (generador sin materializar lista)
  - Throughput: events/sec, predictions/sec
  - Latencias: p50/p95/p99 de `latency_ns`
  - RAM (ru_maxrss), CPU (process_time)
  - MySQL: N/A (engine sin persistencia por diseño)
- **`tests/unit/market/test_benchmark.py`** — 18 tests (baselines, metrics, walk-forward, régimen, predictor pluggable)
- **`domain/entities/market/replay/__init__.py`** — exports actualizados (BASELINES, MetricCollector, MarketRegime, classify_window, etc.)

### Resultado (NVDA 1h real, 2026-08-16)
- Baselines: naive 52.3%/52.9%/53.9% (1h/4h/1d), momentum 51.2%/51.1%/50.9%, random 49.3%/49.4%/50.1%
- Walk-forward: 65 windows, régimen dominante LOW_VOLATILITY (212 segmentos) vs CRASH (87)
- Capacity: ~11K events/s, ~11K predictions/s, p50 ~40us, p95 ~170us, RAM ~126MB estable

## [ZENIN Market — Fase 5: Historical Market Replay] - 2026-08-16

### Added
- **`domain/entities/market/replay/`** — paquete de dominio puro (regla 3; sin red, sin infra):
  - `clock.py` — `ReplayClock` inmutable: `advance_to` monótono, `ClockRollbackError` si retrocede. El reloj es la única fuente de verdad del replay.
  - `feed.py` — contrato `HistoricalFeed` (iterador de observaciones ordenadas no-decreciente; el retroceso se rechaza en el engine).
  - `feature_window.py` — `FeatureWindow`: solo velas **cerradas** bajo el reloj (`ts_close <= clock.now`); derivados puros (returns, mean/std, vwap, typical range) —— deterministas: mismas velas cerradas = mismas features.
  - `predictor.py` — predictor de referencia determinista (drift/vol, sigmoide, sin RNG): baseline honesto para verificar el pipeline completo hoy.
  - `engine.py` — `MarketReplayEngine` (walk-forward): por evento avanza reloj → cierra velas vencidas → resuelve Outcomes con el último cierre conocido → predice cada horizonte al cierre de vela → registra la vela en formación. Feed agotado → pendientes `INVALIDATED`. Rewards materializados en estado `REWARDED` (regla de Fase 3). El mismo engine servirá al live (Fase 6): solo cambia el feed y el reloj.
  - `report.py` — `PerformanceReport` + `HorizonStat`: agregado inmutable de una pasada (predicciones/evaluadas/invalidadas, direction rate, calibration, avg |return error|, reward) + `render_ascii()` → marcador **ZENIN MARKET RUN** por horizonte y por estrategia.
- **`infrastructure/adapters/market/csv_feed.py`** — `HistoricalCsvFeed`: dataset congelado en disco (schema `ts_open,o,h,l,c,v,ts_close`, epoch float), validación de cabecera/columnas/orden/coherencia de cierre; rutas relativas ancladas al repo. El replay jamás toca la red.
- **`scripts/download_market_data.py`** — descarga histórico real (yfinance, dev tool fuera del runtime) a `data/market/<SYMBOL>_<RES>.csv`: NVDA 1m (~7d), 5m (~60d), 1h (~2a), 1d (máx). Datos congelados como artefacto.
- **`scripts/replay_market_run.py`** — ejecuta el replay sobre el histórico real e imprime el marcador (`--all` para todas las resoluciones).
- **`tests/unit/market/test_replay_engine.py`** — 19 tests: reloj monótono, ventana solo con velas ordenadas, warmup honesto (21 velas), invalidation al agotar el feed, **test de oro anti-look-ahead** (feed completo vs cortado: predicciones idénticas en la intersección; reemplazar un precio futuro no cambia las predicciones anteriores), y agregación del report.

### Resultado (NVDA real, 1m — 2026-08-16)
- 10,836 predicciones / 10,755 evaluadas / 81 invalidadas; dirección ~50% en 1m-15m (baseline de momentum, honesto), **cae a 45.4% en 1h con calibration 0.53** — el marcador ya responde "¿hasta qué horizonte funciona?" (la señal del mejor reward está en 15m: +542).

## [ZENIN Market — Fase 4: Provider Adapters] - 2026-08-16

### Added
- **`domain/ports/market_data_provider.py`** — `Protocol` `MarketDataProvider` (provider_name, profile, `parse` despachador, `trade_from_payload`/`quote_from_payload`/`candle_from_payload`/`order_book_from_payload`). Sin imports de infraestructura (regla 3). `symbol`/`interval_seconds`/`received_at` como parámetros de consumer para los payloads que no los declaran.
- **`infrastructure/adapters/market/`** — paquete nuevo (verificado sin colisión con adapters legacy):
  - `alpaca_adapter.py` — `AlpacaAdapter`, perfil con `{TRADES, QUOTES, CANDLES, HISTORICAL_TICKS, HISTORICAL_BARS, REALTIME, VWAP, TRADE_CONDITIONS, NANOSECOND_TIMESTAMP, ADJUSTED_SERIES}`. Mapeo `t`→Trade (i/S/x/p/s/t/c/z/u), `q`→Quote (bx/bp/bs/ax/ap/as/c/z), `b`→Candle (o/h/l/c/v/n/vw, `interval_seconds` requerido: el stream no lo anuncia). ISO-8601 con Z → epoch float. `order_book_from_payload` lanza `NotImplementedError` (sin Capability ORDER_BOOK_L2).
  - `binance_adapter.py` — `BinanceAdapter`, perfil con `{TRADES, QUOTES, CANDLES, ORDER_BOOK_L2, REALTIME, SECOND_AGGREGATES, VWAP}`. Mapeo `aggTrade`→Trade (epoch ms→float s; `m`→taker_side sell/buy), `kline`→Candle (intervalo derivado de `k.i` vía tabla 1s..1M), `bookTicker`→Quote (`received_at` requerido: el payload no lleva timestamp), snapshot REST depth→OrderBookSnapshot (`symbol` del consumer: el endpoint lo recibe por URL). Ambos sin límites inventados (`max_ws_symbols=None`).
- **Fixtures congelados** — `tests/unit/market/adapters/fixtures/`: 7 JSON reales de docs públicas (alpaca_trade/quote/bar, binance_aggtrade/bookticker/kline/depth). Sin red ni claves API.
- **Contract tests** — `tests/unit/market/adapters/test_provider_adapters.py` (32 tests): mismo conjunto contra ambos adaptadores con `pytest.skip` honesto por capability (ORDER_BOOK_L2 solo Binance) + specs específicos: errores provider-mismatch, `received_at` requerido (bookTicker/depth), `symbol` requerido (depth), `interval_seconds` requerido (bar Alpaca), kline con intervalo desconocido (ej. "7h") falla salvo override explícito, m=False→buy.
- Los adapters implementan el Protocol (verificable en runtime: `@runtime_checkable`).

### Changed
- `domain/ports/__init__.py` se dejó intacto (legacy): el port se importa por ruta explícita.

## [ZENIN Market — Architecture Gate] - 2026-08-16

### Added
- **`ARCHITECTURE.md`** — contrato de 14 reglas (guardrail): IoT aislado, MySQL para Market, dominio sin infraestructura, ring de datos Capabilities→Features, ciclo Observation→Prediction→Outcome→Evaluation→Reward, reward solo desde EVALUATED, >180 líneas requiere revisión, no secrets, no refactors fuera de fase, y la colisión `prediction.py` documentada.
- **`tests/unit/market/test_architecture_gate.py`** — prueba permanente de arquitectura (9 tests): `IoT Prediction != ZENIN Market Prediction`, `domain.entities.prediction` sigue siendo módulo .py (no paquete), coexistencia en el mismo intérprete, `entities/__init__` no oculta el legacy, y regla 3 (market no importa pymysql/sqlalchemy/redis/weaviate/alpaca/binance/requests/websockets).
- **`tests/unit/market/test_market_domain_flow.py`** — integración de dominio sintética (5 tests): NVDA 100.00 → +1% @15m P(up)=0.70 → cierra 101.20 → direction correct + within_interval → reward positivo; camino adverso provider-disconnected → INVALIDATED → reward None con `InvalidTransitionError`; y 15m invalidada no afecta a la 5m del mismo feed.
- Veredicto segunda pasada quirúrgica: `observations.py` (262) y `prediction.py` (224) **se quedan** — responsabilidades únicas (jerarquía de observaciones; entidad+transiciones); lo separable ya se extrajo en el refactor previo. Sin fragmentación por número.

## [ZENIN Market — Refactor Fase 2/3] - 2026-08-16

### Changed
- **Modularización del dominio** (sin cambio de comportamiento; 99 tests verdes):
  - `domain/entities/market/validators.py` — validadores compartidos extraídos de `observations.py` (symbol/timestamp/price/size/unit interval), reutilizados por predicción.
  - `domain/entities/market/prediction/` — paquete de predicción movido desde `domain/entities/prediction/` para **evitar colisión con el módulo legacy IoT `domain/entities/prediction.py`** (el paquete lo ocultaba y rompía semánticamente los imports IoT de `Prediction`).
  - `prediction/types.py` — `Regime`, `PredictionInterval`, `InputContext` (responsabilidad separada de la entidad).
  - `prediction/validation.py` — validación numérica pura (horizonte, expected_return, coherencia de intervalo).
  - `prediction/lifecycle.py` — `validate_state_consistency()` como función pura (reglas de estado fuera de la entidad).
  - `prediction.py` pasa de 330 → 224 líneas; `observations.py` de 281 → 262.
- Los archivos legacy IoT (>180 líneas, ~216) se clasificaron y **no se tocaron** (sin red de seguridad de tests ni permiso de romper IoT).

### Fixed
- **Colisión de nombres** `domain/entities/prediction/` vs `domain/entities/prediction.py` (regla: no romper IoT). Mover el paquete a `market/prediction/` restaura el módulo legacy.

## [ZENIN Market — Fase 3] - 2026-08-16

### Dominio de predicción (nuevo paquete `domain/entities/prediction/`)
Ciclo en memoria Prediction -> Outcome -> Evaluation -> Reward, sin infraestructura (sin Alpaca/Binance/MySQL/Redis/Weaviate).

### Added
- **`lifecycle.py`** — `PredictionStatus` (PENDING->ACTIVE->WAITING_OUTCOME->EVALUATED->REWARDED->ARCHIVED, INVALIDATED), máquina de estados (`validate_transition`), `InvalidTransitionError`.
  - Regla crítica: solo `EVALUATED -> REWARDED` materializa reward; `PENDING`, `WAITING_OUTCOME` e `INVALIDATED` jamás producen reward.
- **`prediction.py`** — `Prediction` (frozen/slots/kw_only): horizon_seconds, entry_price, expected_return, probability_up, confidence, interval (`PredictionInterval` con coherencia lower<upper y media contenida), regime (`Regime`), strategy, input_context (`InputContext`). Timestamps consistentes: `timestamp >= observation.timestamp`; estados imposibles rechazados en `__post_init__`.
- **`outcome.py`** — `Outcome` + factory `from_prices` (calcula `return_realized`; `measured_at` en vencimiento del horizonte).
- **`evaluation.py`** — `Evaluation` + funciones puras: dirección, magnitud, `within_interval`, calibración (`|probability_up - acierto|`).
- **`reward.py`** — `Reward` multi-dimensión + `RewardConfig`: dirección x magnitud x calibración − costos − slippage − risk_penalty.
- **Tests** — `tests/unit/market/test_prediction_domain.py`: 65 casos (creación, rangos, intervalos, lifecycle completo + transiciones inválidas, outcome con horizonte equivocado, evaluación, reward +/−, costos, PENDING/INVALIDATED → no reward, inmutabilidad, timestamps, **3 horizontes independientes** — el cierre de 1m no cierra 5m/15m).

### Changed
- `domain/entities/market/__init__.py`: sin cambios (prediction es paquete hermano).

### Fixed
- Bug CPython frozen+slots+kw_only en `super().__post_init__` (workaround: llamada directa a la clase base) — ya contemplado desde FASE 2.
- `archive()` conserva la historia (outcome/evaluation/reward) para diagnóstico; `INVALIDATED` conserva outcome pero nunca evaluation/reward.

## [ZENIN Market — Fase 2] - 2026-08-16

### Dominio de mercado (nuevo paquete `domain/entities/market/`)
Adaptación de ZENIN ML a predicción bursátil (fases 0–2).

### Added
- **Dominio** (`domain/entities/market/`):
  - `capability.py` — enum `Capability` (13 capacidades) y `ProviderProfile` (frozen/slots, validación, `has_order_book`/`has_realtime`)
  - `data_status.py` — enum `DataStatus` (REALTIME/DELAYED/EOD/REPLAY/STALE/UNVERIFIED) con `is_live_signal`
  - `observations.py` — `MarketObservation` (base inmutable) + `Trade`, `Quote`, `Candle`, `OrderBookSnapshot` (frozen/slots/kw_only, validación en `__post_init__`; libros cruzados permitidos; libro L2 ordenado; propiedades `spread`/`midpoint`/`body`/`is_bullish`/`best_bid`/`best_ask`/`imbalance`)
- **Infraestructura MySQL** (`infrastructure/persistence/sql/zenin_market/`):
  - `market_db_connection.py` — `ZeninMarketDbConnection` (singleton, pool, pre_ping, health_check)
  - `migrations/001_create_provider_profiles.sql` + `migrations/runner.py` (idempotente)
- **Tests**: 31 unitarios de entidades (`tests/unit/market/`) y 3 de integración (`tests/integration/market/`) — circuito dominio → MySQL → dominio usando la entidad real con mapeo en el lado del test.
- `pymysql` añadido a `requirements.txt` y `pyproject.toml`.

### Changed
- Nada del pipeline IoT existente se modificó (SQL Server intacto; suite IoT con errores de colección pre-existentes, fuera del alcance).

### Notes
- Contrato v1: ProviderProfile como ciudadano de primera clase, DataStatus por observación, sin leakage en rewards (pendiente Fase 3+). Proveedores: Alpaca (núcleo, equities) + Binance (cripto); yfinance/Finnhub descartados.

## [2.0.0] - 2026-06-23

### Added
- **Pipeline 25+ fases** — Evolucion de 15 a 25+ fases cognitivas con nuevos modulos:
  - ContextPhase, PredictionReadinessGate, DriftResponse, CausalPhase, MemoryPhase, ShadowEvaluation, Observability
- **Anomaly Ensemble v2.0** — Clean architecture refactor (F1: 0.164 → 0.2857, FP: 73 → 24)
- **MoE Package** — `infrastructure/ml/moe/` con gating tree, sparse fusion, expert registry, rollout
- **Inference Module** — `infrastructure/ml/inference/` con MLE, Bayesian (prior/likelihood/posterior), Naive Bayes, Platt scaling
- **Optimization Toolkit** — `infrastructure/ml/optimization/` con SGD, L-BFGS, Newton, genetico, PSO
- **Governance System** — 9 componentes: ParameterRegistry, BoundsEnforcer, DynamicTuner, TemperatureScaler, CorrelationAnalyzer, Decorrelator, Watchdog, RecoveryManager, LoopBoundsMonitor
- **Kalman Engine** — `infrastructure/ml/engines/kalman/` con filtro Kalman CV, Q adaptativo
- **Multivariate Engine** — `infrastructure/ml/engines/multivariate/` con PCA online
- **Seasonal Engine** — `infrastructure/ml/engines/seasonal/` con deteccion FFT de ciclos
- **RUL Estimator** — `infrastructure/ml/anomaly/rul/` para estimacion de vida util residual
- **Cognitive Memory** — Integracion con Weaviate para memoria episodica y semantica
- **Dynamic Features** — Pipeline de features dinamicos: ventanas moviles, derivadas, lags, cross-features
- **Warmup System** — Precarga de modelos y cache al iniciar (`ml_service/warmup.py`)
- **Prometheus Metrics** — Endpoint `/metrics` con metricas de sistema y A/B testing
- **Circuit Breaker** — Redis-backed circuit breaker con backoff exponencial

### Changed
- `CONFIDENCE.MIN_CONFIDENCE`: 0.3 → **0.5** (para datos industriales)
- Default `AnomalyDetectorConfig.voting_threshold`: 0.5 → 0.75 (validado NAB)
- Default `RollingZScoreDetector` parameters: `long_window`: 50 → 400, `hysteresis`: 1 → 7, `z_threshold`: 3.0 → 3.5
- Default `AnomalyDetectorConfig.contamination`: 0.1 → 0.005
- EngineFactory: imports FQN → relativos para evitar duplicacion de clases
- Adapters deprecados eliminados (BaselinePredictionAdapter, TaylorPredictionAdapter)
- Confidence floor unificado via `core/parameters/numerical_constants.py`

### Fixed
- **Duplicate EngineFactory** — imports FQN vs relativos creaban dos clases en memoria
- **`confidence` vs `confidence_score`** — AttributeError silencioso en MoE fusion
- **Doble penalizacion** — MoE + runner aplicaban penalizaciones por separado
- **Anomaly v1.0 adaptativo** — pesos recalculados silenciosamente causaban inestabilidad (eliminado en v2.0)
- **Drift coupling en anomalia** — sobreescribia pesos configurados sin advertencia (eliminado en v2.0)

## [1.0.0] - 2026-05-21

### Added
- **v1.0 Production Configuration** - Validated anomaly detection configuration
  - RollingZScoreDetector: long_window=400, short_window=10, hysteresis=7, z_threshold=3.5
  - Ensemble threshold: 0.75
  - RollingZ weight: 0.20
  - Validation results on NAB machine_temperature_system_failure:
    - F1: 0.2222 (≥ 0.22 ✓)
    - FP: 36 (≤ 40 ✓)
    - Cliff's delta: 0.7261 (≥ 0.70 ✓)
  - Grid search performed over 243 hyperparameter combinations
  - Results saved to: benchmarks/results/grid_search_v2_real.csv
  - Validation saved to: benchmarks/results/validation_v1_production.csv

### Changed
- Default `AnomalyDetectorConfig.voting_threshold`: 0.5 → 0.75
- Default `RollingZScoreDetector` parameters:
  - `short_window`: 10 → 10 (no change)
  - `long_window`: 50 → 400
  - `hysteresis`: 1 → 7
  - `lower`: STAT_THRESHOLDS.Z_SCORE_LOWER → 3.5
  - `upper`: STAT_THRESHOLDS.Z_SCORE_UPPER → 3.5
- Default `AnomalyDetectorConfig.weights['rolling_z']`: 0.20 (no change, validated)

### Benchmark Scripts
- `benchmarks/rolling_z_grid_search_v2_real.py` - Grid search with real detector pipeline
- `benchmarks/validate_v1_production.py` - Validation script for v1.0 config

## [Unreleased]

### Added
- **TextCognitiveEngine completo** — Reconstrucción de `infrastructure/ml/cognitive/text/` con 7 módulos:
  - 6 sub-analyzers: sentiment, urgency, readability, structural, text_structure, patterns
  - Fusión cognitiva ponderada (5 engines) con cálculo de confianza por entropía normalizada
  - `RegexEntityExtractor` — 6 patrones de entidades (EQUIPMENT, METRIC, ALERT, TEMPORAL, LOCATION, OPERATIONAL)
  - `HybridEntityDetector` — regex passthrough o Weaviate server-side text2vec (gateado por flag)
  - Stream endpoint `/graphql` via Strawberry-GraphQL (agnóstico a tipo de serie temporal)
- **8 feature flags** — `ML_ENABLE_TEXT_ANALYSIS`, `ML_ENABLE_TEXT_PERCEPTION`, `ML_ENABLE_HYBRID_EMBEDDINGS`, `ML_ENABLE_GRAPHQL_API` + 4 flags híbridos (todos false por defecto)

### Fixed
- **Conclusion engañosa para inputs no-TEXT** — `format_conclusion()` mostraba "Sentiment: neutral" en análisis numéricos; ahora chequea `input_type` antes de incluir líneas de Urgency/Sentiment/Entities (`conclusion_formatter.py`)
- **Entity dedup case-insensitive** — regex EQUIPMENT retornaba "comp" y "COMP" como entidades separadas; fixeado con pase final de consolidación en `RegexEntityExtractor` + `_extract_entities_regex` en GraphQL resolver
- **ML_ENABLE_HYBRID_EMBEDDINGS default True → False** — era dead config; el módulo `embeddings/` nunca existió hasta ahora, el flag True siempre caía a ImportError

## [10.5] — 2026-08-23 — Calibración integrada en el pipeline de emisión

Cierre del ítem 10.5: la calibración deja de ser solo medición (dashboard 7.5)
y pasa a CORREGIR `probability_up` en vivo, con los 4 no-negociables cumplidos.

### Pipeline
```
raw prediction → active calibrator → calibrated probability
→ evidence gate → prediction persistence → outcome → evaluation
```

### Added
- **`calibration/pipeline.py`** — integración dominio-puro sin I/O:
  - `CalibratedPredictor` / `wrap_predictor()`: wrapper del Protocol `Predictor`.
    Envolver `cfg.predictor` NO requiere cambios en engine ni emitter.
    Passthrough idéntico mientras no haya calibrador aceptado; solo recalibra
    `probability_up` (clamp [0.05,0.95]); `expected_return`/intervalo intactos.
  - `CalibrationEvidence` + `evidence_log` + `export_state/import_state`:
    coexistencia raw vs calibrated por predicción (`prob_raw`, `prob_calibrated`,
    `fallback_level`, `calibrator_version`, `prediction_id` reconstruible).
  - `collect_training_pairs()` y `try_refit()` para refit walk-forward.
- **Nivel HORIZON en la fallback hierarchy**: context(s·hz·r) → **horizon(s·hz)**
  → regime(s·r) → strategy(s) → global → UNCALIBRATED. Sin evidencia se marca
  `UNCALIBRATED` explícito; nunca se inventa confianza.

### Fixed
- **Fallback no-CONTEXT nunca aplicaba calibración (crítico)**: el lookup
  seleccionaba el calibrador correcto pero `apply_with_fallback` llamaba a
  `calibrate()` con la clave fina original; los params viven bajo la clave
  gruesa del nivel → miss garantizado → prob cruda silenciosa. Fix:
  `resolve_fallback_context()` resuelve la clave canónica por nivel.
- **Nivel GLOBAL siempre identidad (misma familia)**: se ajustaba con los
  contextos finos originales pero se buscaba bajo `("GLOBAL",0,"ALL")`. Fix:
  reetiquetado a clave canónica antes de fit.
- **Swap NO conservador (violaba no-negociable 4)**: `train_and_evaluate`
  guardaba `_calibrators` incondicionalmente aunque TODOS los veredictos
  fueran REJECTED. Ahora el candidato solo se activa si todo lo evaluado fue
  ACCEPTED; si no, la versión activa queda intacta (return None) y las
  comparaciones del candidato se registran como diagnóstico.
- **Gate económico espurio**: la aceptación exigía `economic_impact >= tol`;
  un modelo crudo sobreconfiado con suerte direccional en test RECHAZABA su
  propia recalibración honesta (observado: Brier +0.15, econ -0.033 → REJECTED).
  El protocolo 10.5 gatea por Brier en test congelado; el edge económico queda
  como métrica visible para dashboards, no como condición.
- **DEBUG prints en dominio** (`prediction_transitions.py`) eliminados;
  `Prediction.can_produce_reward` normalizado a property (contrato ambiguo
  método/atributo entre tests); `test_can_produce_reward_flag` reparado
  (quedó a nivel de módulo con `self` fantasma tras sesión de debug previa).

### Tests
- `tests/unit/market/test_calibration_pipeline.py` (17):
  passthrough honesto sin calibrador; corrección P=0.90-sobre-moneda → ~0.5-0.7
  con ECE/Brier mejorando; contrato [0.05,0.95]; jerarquía cae a HORIZON cuando
  celdas finas <30 con horizontes con evidencia; swap conservador (v1 intacta
  byte-a-byte si el candidato pierde); versión fluye a evidencia; split
  temporal 60/20/20 ordenado/disjunto; equivalencia exacta fit-vs-train-only
  (no leakage); round-trip de evidencia; cola acotada.

### Known issues (pre-existentes, fuera de alcance)
- `tests/integration/cognitive/*`: 24 fallos committeados en HEAD — tests de
  fases (`pattern_phase`, `dynamic_phase`, `regime_phase`) nunca implementadas
  en `orchestration/phases/`. Requiere decisión: implementar o eliminar tests.
- Benchmarks P99 (`tests/benchmarks/test_phase3_latency.py`) sensibles a carga
  de escritorio: P50 estable (~2.1ms « target), P99 oscila 4-7ms con load>3.
  Medir en máquina quieta o gatear por P50.

## [MVP-0.1] — 2026-08-23 — Paper Bot BTC-USD (wiring mínimo)

Primer experimento live de ZENIN sobre un mercado real. Sin dinero, sin
órdenes, sin aprendizaje adaptativo: solo mirar, registrar y responder.

### Pipeline
```
BINANCE LIVE (klines públicas) → MarketObservation → FeatureWindow
→ raw predictor → Calibration wrapper (10.5) → Evidence Gate
→ Prediction + Evidence → MySQL → Outcome → Evaluation → Reward
```

### Added
- **`infrastructure/adapters/market/binance_klines_feed.py`** — poller REST
  público `/api/v3/klines` (sin API key): solo velas CERRADAS, dedup por
  timestamp, gap tracking explícito, buffer acotado como PriceLookup,
  transporte HTTP inyectable para tests sin red.
- **`infrastructure/adapters/market/paper_runner.py`** — `PaperBotRunner`:
  ciclos STATELESS con MySQL como estado (crash-proof trivial). Regla anti-
  envenenamiento: las feed_ended históricas no se persisten (el siguiente
  ciclo las re-emite resueltas vía upsert idempotente), EXCEPCIÓN la señal
  fresca de la penúltima vela (el engine emite una vela tarde por diseño).
- **`domain/.../calibration/gate.py`** — `EvidenceGate`: NO_TRADE si
  UNCALIBRATED o zona neutral; LONG/SHORT con margen configurable. Cada
  decisión registra acción+motivo (`PaperDecision`, `EvidenceRecord`).
- **Artefacto de calibrador versionable**: `export_calibrator_state` /
  `import_calibrator_state` (JSON puro, auditable). Ciclo MVP: correr
  UNCALIBRADO → refit offline → cargar artefacto aceptado.
- **`migrations/006_create_calibration_evidence.sql`** +
  `CalibrationEvidenceRepository`: "¿con qué calibrador trabajaba ZENIN a
  las 14:37?" → prediction_id, prob_raw/calibrated, fallback_level,
  calibrator_version, paper_action, gate_reason. Upsert idempotente.
- **`scripts/zenin_paper_btc.py`** — runner live: banner PAPER TRADING /
  NO REAL MONEY / $0 COP, SIGINT/SIGTERM graceful, --no-db explícito e
  inservible para reproducibilidad, status line por ciclo (uptime, velas,
  pred, NO-TRADE rate, P, cal=versión, conn, gaps, err).

### Smoke test contra datos reales (2026-08-23)
1 ciclo real Binance BTCUSDT 1m: 299 velas cerradas, 459 predicciones,
3 señales frescas gateadas NO_TRADE=100% (UNCALIBRATED honesto),
shutdown limpio por señal.

### Tests
- `test_evidence_gate.py` (8): matriz del gate + round-trip artefacto.
- `test_binance_klines_feed.py` (7): parse kline REST, cerradas vs en
  formación, dedup/startTime, error de red degradado, gaps contados.
- `test_calibration_evidence_repository.py` (3): mapeo fila↔registro puro.
- `test_paper_runner.py` (6): ciclo completo con fakes, idle, no-duplicación,
  artefacto cargado fluye a status, loop max_cycles/stop.

## [REFACTOR-CONSOLIDACIÓN] — 2026-08-23 — Todo vive en iot_machine_learning

Fin de la estructura de doble árbol en la raíz ST. Motivación: el paquete
`infrastructure` existía dos veces (raíz e iot_ml) → guerra de namespaces,
hacks de sys.path en cada conftest, y todo lo externo al repo git quedaba
sin versionar.

### Movido a su hogar canónico
| Origen (raíz ST) | Destino |
|---|---|
| `src/core/orchestration/rosa_roja/` | `core/orchestration/rosa_roja/` (imports internos relativos intactos) |
| `infrastructure/ml/adapters/*` | `infrastructure/ml/adapters/` (fin de la sombra de namespaces) |
| `src/infrastructure/market/execution/rosa_roja_market_handler.py` | `infrastructure/adapters/market/rosa_roja_market_handler.py` |
| `tests/core/orchestration/rosa_roja/*` | `tests/unit/rosa_roja/` (+synthetic/, +test_phase3_latency) |
| `benchmark_quick.py`, `benchmark_rosa_roja.py` | `benchmarks/` |

### Rewrites
- Global: `from src.core.orchestration.rosa_roja` → `from core.orchestration.rosa_roja`
  (factory, handler, adapters, tests, benchmarks).
- Adapters: `from iot_machine_learning.infrastructure.ml.` → `from infrastructure.ml.`
- Tests synthetic envs: `tests.core.orchestration...envs` → `tests.unit.rosa_roja.synthetic.envs`.
- Eliminados: shims raíz (`infrastructure/__init__.py`), conftest con hacks
  de 5 niveles de sys.path, árboles origen completos. La raíz ST queda solo
  con los AUDIT_REPORT_*.md + este repo.

### Verificación
- tests/unit/market: 426 passed, 1 skipped.
- tests/unit/rosa_roja: 159 passed desde AMBAS raíces (ST e iot_ml).
- Colección cognitive desde raíz ST: 63 tests sin errores de import.
- Smoke: create_rosa_roja_engine, infrastructure.ml.adapters, engine y handler importan OK.
- Los benchmarks P99 siguen flaky bajo carga de escritorio (deuda documentada).

## [FASE-4] — 2026-08-23 — Live Bot Event-Driven (BTCUSDT, Binance)

Implementación completa del bot live event-driven para trading en Binance Futures (USDT-M).

### Added
- **BinanceWSClient** (`ws_client.py`): WebSocket client con reconexión exponencial, ping/pong keepalive, buffer con backpressure, métricas de latencia
- **OrderBookL2** (`order_book_state.py`): Libro L2 sincronizado (snapshot REST + deltas WS @100ms), métricas OBI, spread, microprice, VWAP
- **BinanceWSFeed** (`ws_feed.py`): Feed asíncrono que combina depth@100ms + aggTrade + bookTicker, emite `MarketObservation`
- **MarketFeatureExtractor** (`rosa_roja_features.py`): 10 features (log_return, volatility, spread_bps, volume, vpin, obi, candle_body, range, trade_intensity)
- **BinanceOrderClient** (`order_client.py`): Cliente REST firmado HMAC-SHA256, rate limiting (token bucket), retry exponencial, order types LIMIT GTX/IOC, MARKET, STOP_MARKET, TAKE_PROFIT
- **BinanceAccount** (`account.py`): Sincronización balance/posiciones, PnL realizado/no realizado, margen, equity
- **LiveBotConfig** (`live_config.py`): Configuración completa tipada, cooldown dinámico según lambda_t, presets conservative/aggressive/testnet
- **LiveBotRunner** (`live_runner.py`): Loop event-driven tick→features→engine→handler→orders, state persistence (JSON), health checks, graceful shutdown (SIGTERM/SIGINT)
- **RosaRojaExpert** (`rosa_roja_expert.py`): Adapter para integrar Rosa Roja como challenger expert en MoE principal
- **MoE Integration**: Registro en `ExpertRegistry` con pesos por régimen (8-10% challenger), feature flag `ML_ENABLE_ROSA_ROJA_EXPERT`
- **Script CLI** (`zenin_live_btc.py`): Entry point con args, presets, dry-run, config file
- **Config examples**: `config/live_btc_testnet.json`, `config/live_btc_mainnet.json`

### Tests
- **OrderBookL2**: 16 tests (snapshot, deltas, gaps, metrics, snapshots, truncation)
- **LiveBotRunner**: 8 tests (cooldown, price_change, phi/lambda thresholds, dynamic cooldown, state persistence, shutdown)
- **MoE Adapter**: 15 tests (contracts, registry, dispatcher, fallback, determinism)
- **Total**: 596 tests passing (excl. 5 flaky latency benchmarks)

### Architecture
```
Binance WS (depth@100ms + aggTrade + bookTicker)
         │
         ▼
BinanceWSFeed → Ring Buffer → MarketFeatureExtractor (10 features)
                                                    │
                                                    ▼
                                           RosaRojaEngine (Master Equation)
                                                    │
                                                    ▼
                                         RosaRojaMarketExecutionHandler
                                                    │
                                                    ▼
                                         BinanceOrderClient (REST signed)
```

### Decisiones Técnicas
- **Event-driven puro**: Sin polling, reactivo a ticks WS
- **Cooldown dinámico**: lambda_t > 0.8 → 100ms, 0.5-0.8 → 50%, <0.5 → normal
- **Post-Only por defecto**: GTX para maker, fallback MARKET solo si phi_moe ≥ 0.7 + alta accel
- **Emergency Flush**: lambda_t ≥ 0.95 O cos(θ) < -0.1 → cancel all + close @ market
- **State persistence**: JSON cada 5min + shutdown, crash-recovery < 5s
- **Audit log**: NDJSON rotación diaria, telemetry_hash + decision_trace
- **Dry-run mode**: Simulación completa sin órdenes reales

### Deuda Conocida
- Benchmarks latencia P99 flaky bajo carga (5 tests Phase 3)
- Tests legacy cognitive 24 fallos (fases no implementadas)
- aiofiles requerido para persistencia (añadido a deps)
