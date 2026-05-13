# Cumplimiento y Auditoría

**Última actualización:** 2026-05-12
**Archivo fuente:** `infrastructure/ml/cognitive/compliance/compliance_exporter.py`, `domain/ports/audit_port.py`

---

## ComplianceExporter

### Formato NDJSON

Cada predicción genera exactamente una línea NDJSON (newline-delimited JSON) en el sink configurado (`ML_COMPLIANCE_EXPORT_PATH`).

### Campos del Registro

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `schema_version` | string | `"1.0"` — versión del esquema de registro |
| `record_id` | string | UUID v4 en hex — identificador único del registro |
| `created_at` | string | ISO-8601 UTC con microsegundos (`2026-05-04T14:32:10.123456Z`) |
| `series_id` | string | Identificador de la serie auditada |
| `outcome` | object | Subcampos: `predicted_value`, `confidence`, `trend` (o forma enriquecida si hay explanation interna) |
| `sanitization_flags` | array | Flags de sanitización (ej. `["cusum_ramp_up"]`) |
| `fusion_flags` | array | Flags de fusión (ej. `["hampel_rejected:1"]`) |
| `engine_failures` | array | Lista de fallos de motores (timeout, excepción, cannot_handle) |
| `hampel` | object/null | Diagnóstico del filtro Hampel si aplica |
| `pipeline_timing` | object/null | Tiempos por fase en ms (`{"perceive_ms": 0.5, ...}`) |
| `explanation_digest` | string/null | SHA-256 del explanation canónico |
| `content_hash` | string | SHA-256 del cuerpo canónico del registro |
| `hmac_sha256` | string/null | HMAC-SHA256 del cuerpo canónico (si `hmac_key` configurada) |

### Ejemplo de Línea NDJSON

```json
{"schema_version":"1.0","record_id":"a1b2c3...","created_at":"2026-05-04T14:32:10.123456Z","series_id":"temp_reactor_01","outcome":{"predicted_value":452.3,"confidence":0.87,"trend":"stable"},"sanitization_flags":[],"fusion_flags":["hampel_rejected:1"],"engine_failures":[],"hampel":{"rejected_count":1,"mad":2.1,"threshold":6.3},"pipeline_timing":{"perceive_ms":0.5,"predict_ms":12.3,"adapt_ms":0.2,"inhibit_ms":0.1,"fuse_ms":0.8,"explain_ms":1.2,"total_ms":15.1},"explanation_digest":"e3b0c442...","content_hash":"f6c5d4e3...","hmac_sha256":"aabbccdd..."}
```

### HMAC-SHA256

1. **Cuerpo canónico:** JSON ordenado lexicográficamente por clave, sin espacios, encoding UTF-8.
2. **Content hash:** SHA-256 del cuerpo canónico.
3. **HMAC:** HMAC-SHA256 del cuerpo canónico con clave configurada en `ML_COMPLIANCE_HMAC_KEY`.

### Verificación Independiente

```python
from compliance_exporter import ComplianceExporter

record = ComplianceRecord.from_json_line(line)
is_valid = ComplianceExporter.verify_record(record, key=b"mi-clave-secreta")
```

**Características de seguridad:**
- Comparación constant-time (`hmac.compare_digest`) — no revela información temporal.
- Si el content hash no coincide → detecta tampering de contenido.
- Si el HMAC no coincide → detecta clave incorrecta o tampering.

### Escritura Atómica

- `append` + `flush` + `os.fsync(fileno())` — datos durables en disco antes de retornar.
- `threading.Lock` a nivel de instancia — líneas NDJSON no se entrelazan bajo concurrencia.
- Fallo de escritura → log WARNING + retorna `None` (no interrumpe el pipeline).

---

## AuditPort

### Eventos Auditables

| Evento | Método | Controles ISO 27001 |
|--------|--------|---------------------|
| Predicción generada | `log_prediction` / `log_series_prediction` | A.12.4.1 — Logging de eventos |
| Anomalía detectada | `log_anomaly` / `log_series_anomaly` | A.12.4.1 — Logging de eventos |
| Cambio de configuración | `log_config_change` | A.12.4.3 — Registro de cambios |
| Acceso a datos | `log_event` genérico | A.9.4.2 — Control de acceso |
| Decisión automatizada | `log_event("decision", ...)` | A.12.4.1 — Logging de eventos |

### Dual Interface

- **Legacy:** `log_prediction(sensor_id: int, ...)` — abstracto, debe implementarse.
- **Agnóstico:** `log_series_prediction(series_id: str, ...)` — default bridge que convierte `series_id → int` y delega.

Implementaciones existentes: `FileAuditLogger`, `NullAuditLogger`, `SqlServerStorageAdapter` (con AuditPort).

---

## Estado de Cumplimiento por Estándar

### ISO 13374 (Condition Monitoring & Diagnostics)

| Bloque Funcional | Estado | Notas |
|-----------------|--------|-------|
| Percepción de estado (State Perception) | ✅ Implementado | `PerceivePhase` + `SignalAnalyzer` |
| Indicadores de condición (Condition Indicators) | ✅ Implementado | `noise_ratio`, `stability`, `regime`, `drift_magnitude` |
| Detección de anomalías (Anomaly Detection) | ✅ Implementado | `VotingAnomalyDetector` con 8+ detectores |
| Diagnóstico de causa raíz (Root Cause Diagnosis) | ❌ Faltante | `CausalNarrativeBuilder` es básico; no llega a causa mecánica |
| Pronóstico de vida útil remanente (RUL) | ❌ Faltante | No implementado |
| Recomendación de acción (Action Recommendation) | ⚠️ Parcial | `ContextualDecisionEngine` da acciones, no prioriza tareas de mantenimiento |

### ISO 27001

| Control | Estado | Notas |
|---------|--------|-------|
| A.12.4.1 — Logging de eventos | ✅ Implementado | `AuditPort` con eventos estructurados |
| A.12.4.3 — Registro de cambios | ✅ Implementado | `log_config_change` |
| A.9.4.2 — Control de acceso a datos | ⚠️ Parcial | `safe_series_id_to_int` valida entrada, pero no hay RBAC |
| A.14.2.8 — Pruebas de sistemas | ✅ Implementado | 1,800+ tests unitarios + meta-tests arquitectónicos |
| A.12.3.1 — Gestión de capacidad | ❌ Faltante | No hay monitoreo de capacidad ni auto-scaling |
| A.17.1.1 — Continuidad de negocio | ❌ Faltante | No hay plan de DR documentado |

### IEC 62443 (Seguridad Industrial)

| Control | Estado | Notas |
|---------|--------|-------|
| Segmentación de red | ⚠️ Parcial | Segmentación por `series_id` a nivel de código, no de red |
| Validación de entrada | ✅ Implementado | `safe_series_id_to_int`, bounds checking en `SanitizePhase` |
| Autenticación de dispositivos | ❌ Faltante | No hay autenticación MQTT/HTTP en el pipeline |
| Seguridad por diseño (Security by Design) | ❌ Faltante | No evaluado formalmente |

### NIS2

| Requisito | Estado | Notas |
|-----------|--------|-------|
| Notificación de incidentes | ❌ Faltante | No hay mecanismo de notificación a autoridad |
| Gestión de riesgos de cadena de suministro | ❌ Faltante | No auditado |
| Registro de operadores | ❌ Faltante | No aplica aún (directiva en transposición en LATAM) |

---

## Cómo Usar el Compliance Export para una Auditoría Real

### Paso 1: Verificar integridad

```bash
# Contar registros
gwc -l compliance_export.ndjson

# Verificar que cada línea tenga HMAC
python -c "
import json
for i, line in enumerate(open('compliance_export.ndjson')):
    r = json.loads(line)
    assert 'hmac_sha256' in r and r['hmac_sha256'], f'Línea {i} sin HMAC'
print('Todas las líneas tienen HMAC')
"
```

### Paso 2: Verificar firmas

```python
from infrastructure.ml.cognitive.compliance.compliance_exporter import ComplianceExporter
from infrastructure.ml.cognitive.compliance.compliance_record import ComplianceRecord

key = b"tu-clave-desde-env"
invalid = 0
with open("compliance_export.ndjson") as fh:
    for line in fh:
        record = ComplianceRecord.from_json_line(line.strip())
        if not ComplianceExporter.verify_record(record, key):
            invalid += 1

print(f"Registros inválidos: {invalid}")
```

### Paso 3: Reconstruir timeline de incidente

```bash
# Filtrar por serie y rango de fechas
jq 'select(.series_id == "temp_reactor_01" and .created_at >= "2026-05-01T00:00:00Z")' compliance_export.ndjson
```

### Paso 4: Correlacionar con decisiones

Los campos `engine_failures`, `fusion_flags`, y `pipeline_timing` permiten reconstruir por qué el sistema tomó una decisión en cada punto del tiempo.
