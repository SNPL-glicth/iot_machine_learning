-- 006_create_calibration_evidence.sql
-- Trading MVP 0.1: evidencia de calibración por predicción (FASE 10.5).
--
-- Responde la pregunta del experimento reproducible:
--   "¿Con qué calibrador estaba trabajando ZENIN a las 14:37?"
--
-- Una fila por señal emitida en vivo (vela más reciente de cada ciclo):
-- qué creyó el modelo crudo (prob_raw), qué corrigió el calibrador
-- (prob_calibrated), con qué versión/nivel de fallback, y qué decidió el
-- evidence gate (paper_action). Append-only; upsert idempotente por
-- prediction_id igual que market_predictions.

CREATE TABLE IF NOT EXISTS calibration_evidence (
    prediction_id          VARCHAR(128) NOT NULL,
    symbol                 VARCHAR(32)  NOT NULL,
    horizon_seconds        INT          NOT NULL,
    observation_timestamp  DOUBLE       NOT NULL,
    regime                 VARCHAR(32)  NULL,

    prob_raw               DOUBLE       NOT NULL,
    prob_calibrated        DOUBLE       NOT NULL,
    fallback_level         VARCHAR(32)  NOT NULL,
    calibrator_version     VARCHAR(64)  NULL,

    paper_action           VARCHAR(16)  NOT NULL,
    gate_reason            VARCHAR(32)  NOT NULL,

    created_at             DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (prediction_id),
    KEY idx_evidence_symbol_obs (symbol, observation_timestamp),
    KEY idx_evidence_version (calibrator_version),
    KEY idx_evidence_action (symbol, paper_action)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
