# ROI y Casos de Uso Empresariales

**Última actualización:** 2026-05-04
**Audiencia:** CTO, Director de Operaciones, Gerente de Mantenimiento

---

## ROI Expandido

### ¿Qué retorno puede esperar una planta industrial?

| Métrica | Estimación conservadora | Capacidad ZENIN que lo genera | Supuesto |
|---|---|---|---|
| Reducción de paradas no planificadas | 20–35% | Drift detection + anomaly voting ensemble | Planta con 10–50 sensores, mantenimiento correctivo predominante |
| Reducción de falsos positivos | 40–60% | Hampel filter + InhibitionGate + atenuadores | Operadores actualmente ignoran alarmas por saturación |
| Tiempo de diagnóstico post-incidente | –70% | ReasoningTrace por fase + ExplanationRenderer | Auditoría actual requiere 3+ sistemas y semanas de trabajo |
| Cumplimiento de auditoría técnica | Automatizado | ComplianceExporter HMAC-SHA256 + AuditPort | Cliente o regulador requiere trazabilidad de decisiones ML |
| Costo vs soluciones enterprise (Palantir, AWS Lookout) | –80% infraestructura | Open source + deploy local sin cloud obligatorio | No requiere reemplazo de sensores existentes |

Estos estimados asumen integración con sensores existentes vía MQTT o HTTP. El sistema no requiere reemplazo de infraestructura OT existente.

---

## Caso de Uso 1: Planta con 10–50 Sensores, Mantenimiento Correctivo

### Contexto

Planta manufacturera mediana con sensores de temperatura, vibración, y presión en equipos rotativos. Mantenimiento predominante: correctivo (reparar cuando falla). Sin sistema de CMMS integrado con datos de sensores.

### Dolor actual

- 2–3 paradas no planificadas por mes, 4–8 horas cada una.
- Costo estimado: USD $5,000–$15,000 por parada (materiales + mano de obra + producción perdida).
- Supervisores revisan dashboards a mano; no hay correlación automática entre sensor y orden de trabajo.

### Qué aporta ZENIN

| Capacidad | Impacto |
|-----------|---------|
| `VotingAnomalyDetector` (8 detectores) | Detecta degradación 24–72h antes de falla catastrófica en equipos rotativos. |
| `VelocityZDetector` + `AccelerationZDetector` | Detecta cambios de régimen (arranque/parada anómalos) antes de que la temperatura cruce umbral. |
| `DriftDetectionPhase` | Identifica cuando un sensor "está mintiendo" (concept drift) antes de que el operador lo note. |
| `ContextualDecisionEngine` | Reduce falsos positivos en 40–60% → operadores vuelven a confiar en las alarmas. |
| `ComplianceExporter` | Cada predicción queda auditada; cuando falla algo, se reconstruye en minutos. |

### ROI estimado (conservador)

- Paradas evitadas: 1 por mes × 12 meses = 12 paradas.
- Ahorro: 12 × $8,000 (promedio conservador) = **$96,000/año**.
- Costo de implementación: 1 ingeniero ML × 3 meses + infraestructura (Docker en servidor existente) ≈ **$25,000**.
- **Payback: 3–4 meses.**

---

## Caso de Uso 2: Planta con Reportes de Mantenimiento en Texto

### Contexto

Planta con técnicos que llenan reportes de mantenimiento en texto libre (Word, Excel, o sistema legacy). Los reportes contienen observaciones valiosas: "ruido anómalo en compresor A", "temperatura alta en cojinete B", pero nadie las cruza con los datos numéricos de sensores.

### Qué aporta ZENIN (pipeline documental)

| Capacidad | Impacto |
|-----------|---------|
| `TextCognitiveEngine` | Extrae entidades y urgencia de reportes de texto. |
| `HybridNeuralEngine` | Clasifica severidad de reportes textuales. |
| `SemanticExtractionPort` | Conecta entidades extraídas con series_id de sensores. |
| Pipeline dual (numérico + documental) | Unifica predicción de sensores + análisis de reportes en un solo `DecisionContext`. |

### Ejemplo de flujo

1. Sensor de vibración reporta anomalía moderada (`score=0.55`, `severity=MEDIUM`).
2. Técnico reporta "ruido anómalo en compresor A" (urgencia=HIGH).
3. `TextCognitiveEngine` extrae: entidad="compresor A", urgencia="HIGH".
4. `ContextualDecisionEngine` aplica amplificador por urgencia textual + anomalía numérica.
5. Decisión final: `INVESTIGATE` (prioridad 2) en vez de `MONITOR` (prioridad 3).

**Impacto:** El sistema eleva la prioridad cuando múltiples fuentes (sensor + humano) coinciden, evitando subestimar problemas emergentes.

---

## Caso de Uso 3: Planta con Auditoría Regulatoria Frecuente

### Contexto

Planta alimentaria o farmacéutica sometida a auditorías de calidad (FDA, ANVISA, INVIMA). Requiere demostrar que decisiones automatizadas son trazables, verificables, y no arbitrarias.

### Qué aporta ZENIN

| Capacidad | Impacto |
|-----------|---------|
| `ComplianceExporter` (NDJSON + HMAC) | Cada decisión ML queda firmada criptográficamente; no se puede repudiar. |
| `AuditPort` | Logging estructurado de predicciones, anomalías, y cambios de configuración. |
| `ExplanationRenderer` | Cada predicción incluye reasoning trace por fase — auditable por un inspector. |
| `pipeline_timing` | Latencia por fase demostrable; no hay cajas negras. |

### Comparación de tiempo de auditoría

| Actividad | Sin ZENIN | Con ZENIN |
|-----------|-----------|-----------|
| Reconstruir decisión de parada | 2–3 días (revisar logs de 3 sistemas) | 5 minutos (`jq` sobre NDJSON) |
| Verificar integridad de logs | No posible (logs sin firma) | 10 minutos (`verify_record` en batch) |
| Explicar por qué se paró la línea | "La alarma sonó" | "Anomalía score=0.87 en sensor temp_reactor_01, regime=STABLE, velocity_z detectó rampa de 2.3°C/min, ContextualDecisionEngine aplicó amplificador drift_high×1.20, acción=ESCALATE" |
| Demostrar que no hay manipulación | No posible | HMAC-SHA256 verificable independientemente |

**Impacto:** Reducción de tiempo de auditoría de semanas a horas. Reducción de riesgo regulatorio por trazabilidad demostrable.

---

## Comparación de Costo Total de Ownership (TCO)

| Componente | ZENIN (on-prem) | AWS Lookout | Palantir AIP |
|---|---|---|---|
| Licencia anual | $0 (open source) | ~$50,000–$200,000* | ~$500,000–$2,000,000* |
| Infraestructura | Servidor existente + Docker | Cloud AWS obligatorio | On-prem costoso |
| Ingeniería ML (setup) | 2–3 meses | 1–2 semanas (managed) | 3–6 meses |
| Ingeniería ML (ongoing) | 0.25 FTE | 0.05 FTE | 0.5–1.0 FTE |
| Integración sensores | MQTT/HTTP nativo | AWS IoT Core requerido | Conector custom |
| Customización | Total (código abierto) | Limitada | Alta pero costosa |
| Escalabilidad >1000 sensores | ⚠️ Degrada | ✅ Nativa | ✅ Nativa |
| Audit trail criptográfico | ✅ HMAC-SHA256 | ❌ No nativo | ❌ No nativo |

\* Estimados de mercado; varían por volumen de datos y contrato.

**Conclusión:** ZENIN es competitivo en TCO para plantas de 10–200 sensores con equipo técnico local. Para >1000 sensores o sin equipo técnico, soluciones managed (AWS Lookout) pueden ser más eficientes.

---

## Requisitos Mínimos de Infraestructura para Producción

| Recurso | Mínimo | Recomendado | Notas |
|---------|--------|-------------|-------|
| CPU | 2 cores | 4 cores | PredictPhase usa ThreadPoolExecutor (max 3 workers) |
| RAM | 4 GB | 8 GB | SlidingWindowStore + priors bayesianos en memoria |
| Disco | 20 GB SSD | 50 GB SSD | NDJSON audit logs crecen con el tiempo |
| Redis | 1 instancia | 1 instancia + AOF | Plasticity shared state + sliding windows |
| SQL Server | Express | Standard | Predicciones, anomalías, configuración |
| Red | 100 Mbps LAN | 1 Gbps LAN | Latencia sensor→pipeline < 50ms recomendado |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | Docker y Python 3.10+ nativos |

### Topología sugerida (on-premise)

```
[Sensores MQTT] ──► [MQTT Broker] ──► [ZENIN ML Service] ──► [Redis]
                                          │                      │
                                          ▼                      ▼
                                    [SQL Server]          [NDJSON Sink]
                                          │
                                          ▼
                                    [Dashboard / API]
```

### Alta disponibilidad (fase 2)

- Redis: Sentinel o Cluster para failover.
- SQL Server: Always On Availability Groups.
- ZENIN ML Service: múltiples réplicas con balanceador (HAProxy / nginx).
- NDJSON Sink: replicación rsync o sistema de archivos distribuido (NFS / Ceph).
