-- FASE 10.1 — Calibration Versioning: guardar calibradores con versionado
-- Regla: nunca modificar predicciones históricas, crear nueva versión para cambios

CREATE TABLE IF NOT EXISTS calibrators (
    calibrator_id VARCHAR(64) PRIMARY KEY COMMENT 'ID único: calibrator_v1, calibrator_v2, etc.',
    method VARCHAR(32) NOT NULL COMMENT 'Método: platt, bucket, isotonic, none',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT 'Fecha de creación',
    is_active BOOLEAN DEFAULT TRUE COMMENT 'Si está activo para nuevas predicciones',
    description TEXT COMMENT 'Descripción del calibrador (qué corrige, por qué se creó)',
    train_samples INT COMMENT 'Total de muestras usadas para entrenar',
    train_brier DECIMAL(10,6) COMMENT 'Brier score en entrenamiento',
    train_ece DECIMAL(10,6) COMMENT 'ECE en entrenamiento',
    test_samples INT COMMENT 'Muestras de validación out-of-sample',
    test_brier DECIMAL(10,6) COMMENT 'Brier score en validación',
    test_ece DECIMAL(10,6) COMMENT 'ECE en validación',
    params_json JSON COMMENT 'Parámetros por contexto: {context_key: {method, params, n_train, train_brier, train_ece}}',
    metadata JSON COMMENT 'Metadata adicional: symbols, train_ratio, etc.',
    INDEX idx_created_at (created_at),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Versiones de calibradores - FASE 10.1';

-- Tabla para tracking de qué calibrador se usó en cada predicción
CREATE TABLE IF NOT EXISTS prediction_calibrators (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    prediction_id VARCHAR(128) NOT NULL COMMENT 'Referencia a market_predictions.prediction_id',
    calibrator_id VARCHAR(64) COMMENT 'Calibrador usado (NULL = sin calibración)',
    prob_raw DECIMAL(6,4) NOT NULL COMMENT 'Probabilidad sin calibrar',
    prob_calibrated DECIMAL(6,4) COMMENT 'Probabilidad calibrada (NULL = sin calibración)',
    context_key VARCHAR(128) COMMENT 'Contexto usado: strategy·horizon·regime',
    applied_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6) COMMENT 'Cuándo se aplicó la calibración',
    UNIQUE KEY uk_prediction (prediction_id),
    INDEX idx_calibrator (calibrator_id),
    INDEX idx_context (context_key),
    FOREIGN KEY (prediction_id) REFERENCES market_predictions(prediction_id) ON DELETE CASCADE,
    FOREIGN KEY (calibrator_id) REFERENCES calibrators(calibrator_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Tracking de calibración por predicción - FASE 10.1';