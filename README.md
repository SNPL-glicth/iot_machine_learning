# ZENIN

> Cognitive Decision Engine for Real-Time Data Analysis
> Production-grade, explainable, and self-adapting ML for mission-critical operations.

![Tests](https://img.shields.io/badge/tests-1800%2B-green) ![Python](https://img.shields.io/badge/python-3.12-blue) ![Docker](https://img.shields.io/badge/docker-ready-blue) ![Architecture](https://img.shields.io/badge/arch-hexagonal-success)

---

## The Problem

- **Black-box predictions** — operators can't trust what they don't understand, especially when automated actions shut valves or page engineers.
- **Static models** — a single offline-trained model fails the moment a sensor shifts from stable to volatile or a new regime emerges.
- **No safety boundaries** — predictions trigger actions with no gate to prevent bad calls from executing in production.

## How ZENIN Solves It

| Advantage | Why it matters |
|-----------|---------------|
| **Signal Inhibition** | Noisy engine outputs suppressed *before* they reach the decision layer. |
| **Multi-Engine Fusion** | Four predictors compete per regime; the system learns which wins without retraining. |
| **Explainability Built-In** | Every prediction carries a reasoning trace — auditable by operators and regulators. |
| **Safety Guardrails** | AUTO / ASK / DENY levels for every autonomous action; critical severity executes immediately. |
| **Real-Time by Design** | p99 < 150ms for the full cognitive pipeline, proven at 1,000+ sensors. |

## Numbers That Matter

| Metric | Value | Meaning |
|--------|-------|---------|
| Test coverage | 1,800+ | Unit, integration, architectural meta-tests |
| Latency (p99) | < 150ms | Full pipeline perception → fusion |
| Online learning | Zero retraining | Bayesian weight updates per regime |
| Outlier rejection | Hampel (≈3σ) | Rogue engine outputs removed before fusion |
| Architecture | Hexagonal + clean ports | Domain layer zero external dependencies |

## The Pipeline

```
PERCEIVE → PREDICT → ADAPT → INHIBIT → FUSE → EXPLAIN → DECIDE → ACT
```

1. **Perceive** — classify signal regime and measure noise.
2. **Predict** — run capable engines concurrently with per-engine timeouts.
3. **Adapt** — retrieve per-regime weights learned from past accuracy.
4. **Inhibit** — suppress engines with high recent error.
5. **Fuse** — Hampel-filter outliers, then weighted consensus.
6. **Explain** — build human-readable reasoning trace.
7. **Decide** — map prediction to recommended action with business impact.
8. **Act** — execute through AUTO / ASK / DENY guardrails.

## Real-World Use Cases

| Domain | Application |
|--------|------------|
| **IoT / Infrastructure** | Predictive maintenance, threshold alerting, anomaly detection on sensor streams |
| **Cybersecurity** | Behavioral baselining, regime-shift detection, automated response with guardrails |
| **Operational Monitoring** | Real-time KPI tracking, SLA breach prediction, automated ticket creation |
| **Document Intelligence** | Crisis report analysis, urgency classification, entity extraction from unstructured text |

## Quick Start

```bash
# Dependencies
docker run -d -p 6379:6379 redis:7-alpine
docker run -d -p 1434:1434 -e SA_PASSWORD=YourPassword123 -e ACCEPT_EULA=Y mcr.microsoft.com/mssql/server:2022-latest

# Run service
uvicorn ml_service.main:app --host 0.0.0.0 --port 8002 --reload

# Verify
curl http://localhost:8002/
# {"service": "iot-ml-service", "version": "0.3.0-GOLD", "status": "ok"}
```

See `docs/configuration.md` for full environment setup.

## Documentation

| Topic | File |
|-------|------|
| Architecture rules | `docs/ARCHITECTURE.md` |
| Engine deep-dive | `docs/ENGINES.md` |
| Feature flags & config | `docs/configuration.md` |
| API reference | `docs/api.md` |
| Operations & monitoring | `docs/operations.md` |
| Development guide | `docs/development.md` |
| Plasticity & learning | `docs/plasticity.md` |

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Uvicorn |
| ML / Math | NumPy, scikit-learn |
| State | Redis (streams, cache, sliding windows) |
| Persistence | SQL Server |
| Deployment | Docker, Kubernetes-ready |
| Observability | Prometheus, structured JSON logging |

## License

Built by Sergio Nicolas. Open to collaboration and pilot deployments.

Contact: LinkedIn | [GitHub](https://github.com/SNPL-glicth/ZENIN)
