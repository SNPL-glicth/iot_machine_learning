## Learning & Adaptation

### How Plasticity Works

Plasticity is **regime-contextual weight learning** — the system learns which engines perform best in specific signal regimes.

**Mechanism:**
1. **Regime detection:** SignalAnalyzer classifies into STABLE, TRENDING, VOLATILE, NOISY, TRANSITIONAL
2. **Error tracking:** After each prediction, `record_actual()` computes |predicted - actual|
3. **Bayesian update:** Inverse error updates Gaussian prior for (regime, engine) pair
4. **Weight computation:** Weights = normalized posterior means
5. **TTL decay:** Unused regimes decay to uniform weights over time

### Key parameters (configurables via feature flags)

All plasticity parameters are now runtime-configurable without redeployment.
Set them as environment variables or in your `.env` file:

| Flag | Default | Controls |
|------|---------|----------|
| `ML_PLASTICITY_ALPHA` | `0.15` | Base EMA smoothing factor |
| `ML_PLASTICITY_REGIME_ALPHAS` | JSON string | Per-regime learning rates |
| `ML_PLASTICITY_MIN_WEIGHT` | `0.05` | Floor to prevent total suppression |
| `ML_PLASTICITY_MAX_REGIMES` | `10` | LRU eviction threshold |
| `ML_PLASTICITY_REGIME_TTL_SECONDS` | `86400.0` | Unused regime decay (1 day) |
| `ML_PLASTICITY_NOISE_THRESHOLD` | `0.3` | Noise penalty activation |
| `ML_PLASTICITY_LR_FACTORS` | JSON string | Per-regime LR multipliers |

**JSON flag example** — override regime learning rates at runtime:
```bash
export ML_PLASTICITY_REGIME_ALPHAS='{"STABLE":0.10,"TRENDING":0.20,"VOLATILE":0.35,"NOISY":0.05}'
export ML_PLASTICITY_LR_FACTORS='{"STABLE":1.0,"TRENDING":1.2,"VOLATILE":1.5,"NOISY":0.8}'
```

Hot-reload: every call to `_get_flags()` reads the current singleton.
Changes take effect on the next prediction without restarting the service.

### Why Redis is Used

**Problem:** Multi-worker deployments need shared plasticity state

**Solution:** Redis-backed shared plasticity
- **Writes:** Each `update()` writes to Redis hash `plasticity:{regime}`
- **Reads:** `get_weights()` reads from Redis with 60s local cache
- **Consistency:** All workers see same weights for same regime
- **Performance:** Local cache prevents Redis round-trips

### Redis key registry

All Redis key patterns are centralized in `infrastructure/redis_keys.py`.
Never hardcode key strings — always use `RedisKeys`:

```python
from infrastructure.redis_keys import RedisKeys

# Plasticity weights
key = RedisKeys.plasticity("STABLE")          # → "plasticity:STABLE"

# Error history
key = RedisKeys.error_history("s42", "taylor") # → "error_history:s42:taylor"

# Anomaly tracking
key = RedisKeys.anomaly_track("s42")           # → "anomaly_track:s42"
key = RedisKeys.anomaly_consecutive("s42")     # → "anomaly_consecutive:s42"

# Alert suppression
key = RedisKeys.last_alert("s42")              # → "last_alert:s42"
key = RedisKeys.suppressed("s42")              # → "suppressed:s42"
```

Changing a key pattern requires editing only `RedisKeys` — all consumers
update automatically. This enables full key auditability (ISO 27001 A.12.4.1).

### How the System "Learns" Over Time

**Short-term (within session):**
- In-memory accuracy tracking per regime
- Immediate weight adjustment after each prediction

**Long-term (across restarts):**
- Optional repository persistence (SQL Server)
- Batched writes every 10 updates (performance optimization)
- State reload on initialization

**Example Learning Curve:**
```
Hour 0 (cold start):
  STABLE: {taylor: 0.5, baseline: 0.5}

Hour 1 (after 100 predictions):
  STABLE: {taylor: 0.72, baseline: 0.28}  # Taylor better in stable

Hour 2 (pattern change to TRENDING):
  TRENDING: {taylor: 0.45, statistical: 0.55}  # Statistical better for trends
```
