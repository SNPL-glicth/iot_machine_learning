-- 002_create_market_predictions.sql
-- Tabla de predicciones de ZENIN Market (FASE 7).
--
-- Una fila por predicción: snapshot de la observación (JSON), predicción,
-- y los campos del ciclo de vida Outcome -> Evaluation -> Reward (NULL
-- hasta que se resuelven). Append-only por diseño: FASE 7 NO aprende de
-- estos datos todavía (regla: primero observar, luego adaptar).
--
-- prediction_id es único por (symbol, obs_timestamp, horizon): el upsert
-- hace que re-correr un shadow jamás duplique filas.

CREATE TABLE IF NOT EXISTS market_predictions (
    prediction_id          VARCHAR(128) NOT NULL,
    symbol                 VARCHAR(32)  NOT NULL,
    horizon_seconds        INT          NOT NULL,
    emitted_at             DOUBLE       NOT NULL,
    observation_timestamp  DOUBLE       NOT NULL,
    entry_price            DOUBLE       NOT NULL,
    expected_return        DOUBLE       NOT NULL,
    probability_up         DOUBLE       NOT NULL,
    confidence             DOUBLE       NOT NULL,
    interval_lower         DOUBLE       NULL,
    interval_upper         DOUBLE       NULL,
    interval_confidence    DOUBLE       NULL,
    regime                 VARCHAR(32)  NULL,
    strategy               VARCHAR(64)  NULL,
    data_status            VARCHAR(32)  NULL,
    feature_count          INT          NULL,
    feature_version        VARCHAR(64)  NULL,
    observation            JSON         NOT NULL,
    status                 VARCHAR(32)  NOT NULL,
    invalidation_reason    VARCHAR(64)  NULL,

    outcome_measured_at    DOUBLE       NULL,
    outcome_final_price    DOUBLE       NULL,
    outcome_return_realized DOUBLE      NULL,

    direction_correct      TINYINT(1)   NULL,
    magnitude_error        DOUBLE       NULL,
    within_interval        TINYINT(1)   NULL,
    calibration_error      DOUBLE       NULL,

    reward_direction       DOUBLE       NULL,
    reward_magnitude       DOUBLE       NULL,
    reward_calibration     DOUBLE       NULL,
    reward_execution_costs DOUBLE       NULL,
    reward_total           DOUBLE       NULL,

    created_at             DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at             DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                                       ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (prediction_id),
    KEY idx_symbol_emitted (symbol, emitted_at),
    KEY idx_status (status),
    KEY idx_horizon (horizon_seconds),
    KEY idx_strategy (strategy)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;