## API Reference

> For complete Pydantic schema definitions, see `ml_service/api/schemas.py`.

### Prediction

**POST** `/api/v1/predict`

```json
// Request
{
  "series_id": "sensor_42",
  "values": [82.1, 83.5, 84.2, 85.0, 85.5],
  "timestamps": [1741234500, 1741234515, 1741234530, 1741234545, 1741234560],
  "threshold": 90.0
}
```

### Anomaly Detection

**POST** `/api/v1/detect-anomalies`

```json
// Request
{
  "series_id": "sensor_42",
  "values": [82.1, 83.5, 84.2, 85.0, 92.5],
  "timestamps": [1741234500, 1741234515, 1741234530, 1741234545, 1741234560]
}
```

### Document Analysis

**POST** `/api/v1/analyze-document`

```json
// Request
{
  "document_id": "crisis_report_001",
  "content": "ALERTA: Rack B-07 temperatura 94°C (límite 80°C). Servidores offline. 77% capacidad perdida.",
  "tenant_id": "acme_corp",
  "content_type": "text"
}
```

**Response:**
```json
{
  "document_id": "crisis_report_001",
  "tenant_id": "acme_corp",
  "classification": "infrastructure",
  "severity": "critical",
  "confidence": 0.65,
  "entities": ["94°C", "80°C"],
  "actions": [
    "Restart affected node immediately",
    "Check sensor readings and thresholds",
    "Reduce system load to prevent cascade failure"
  ],
  "processing_time_ms": 145.2
}
```
