-- FASE 10.5 — Versionado Real: tracking completo de versiones por predicción
-- Regla: cada predicción debe registrar qué versiones estaban activas

-- Agregar columnas de versionado a market_predictions
ALTER TABLE market_predictions 
ADD COLUMN model_version VARCHAR(32) DEFAULT 'baseline_v1' COMMENT 'Versión del modelo usado',
ADD COLUMN calibrator_version VARCHAR(32) COMMENT 'Versión del calibrador usado (NULL = sin calibración)',
ADD COLUMN strategy_version VARCHAR(32) DEFAULT 'v1' COMMENT 'Versión de la estrategia',
ADD COLUMN evidence_policy_version VARCHAR(32) DEFAULT 'v1' COMMENT 'Versión del policy de evidencia',
ADD INDEX idx_model_version (model_version),
ADD INDEX idx_calibrator_version (calibrator_version);

-- Tabla para tracking de rechazos de calibradores
CREATE TABLE IF NOT EXISTS calibrator_rejections (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    calibrator_id VARCHAR(64) NOT NULL COMMENT 'ID del calibrador rechazado',
    rejection_reason TEXT NOT NULL COMMENT 'Razón del rechazo',
    train_brier DECIMAL(10,6) COMMENT 'Brier en training',
    val_brier DECIMAL(10,6) COMMENT 'Brier en validation',
    test_brier DECIMAL(10,6) COMMENT 'Brier en test',
    brier_delta DECIMAL(10,6) COMMENT 'Diferencia de Brier (test - train)',
    train_ece DECIMAL(10,6) COMMENT 'ECE en training',
    val_ece DECIMAL(10,6) COMMENT 'ECE en validation',
    test_ece DECIMAL(10,6) COMMENT 'ECE en test',
    ece_delta DECIMAL(10,6) COMMENT 'Diferencia de ECE (test - train)',
    economic_impact DECIMAL(10,6) COMMENT 'Impacto económico neto',
    rejected_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6) COMMENT 'Cuándo se rechazó',
    metadata JSON COMMENT 'Metadata adicional',
    INDEX idx_calibrator (calibrator_id),
    INDEX idx_rejected_at (rejected_at),
    FOREIGN KEY (calibrator_id) REFERENCES calibrators(calibrator_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Historial de rechazos de calibradores - FASE 10.5';

-- Tabla para métricas de comparación raw vs calibrated
CREATE TABLE IF NOT EXISTS calibration_comparisons (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    calibrator_id VARCHAR(64) NOT NULL,
    context_key VARCHAR(128) NOT NULL COMMENT 'strategy·horizon·regime',
    
    -- Métricas RAW
    raw_brier DECIMAL(10,6),
    raw_ece DECIMAL(10,6),
    raw_log_loss DECIMAL(10,6),
    raw_wilson_lb DECIMAL(10,6),
    raw_economic_edge DECIMAL(10,6),
    
    -- Métricas CALIBRATED
    calibrated_brier DECIMAL(10,6),
    calibrated_ece DECIMAL(10,6),
    calibrated_log_loss DECIMAL(10,6),
    calibrated_wilson_lb DECIMAL(10,6),
    calibrated_economic_edge DECIMAL(10,6),
    
    -- Diferencias
    brier_improvement DECIMAL(10,6),
    ece_improvement DECIMAL(10,6),
    log_loss_improvement DECIMAL(10,6),
    wilson_improvement DECIMAL(10,6),
    economic_impact DECIMAL(10,6),
    
    -- Veredicto
    is_accepted BOOLEAN,
    rejection_reason TEXT,
    
    measured_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
    
    INDEX idx_calibrator (calibrator_id),
    INDEX idx_context (context_key),
    INDEX idx_measured_at (measured_at),
    FOREIGN KEY (calibrator_id) REFERENCES calibrators(calibrator_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Comparación raw vs calibrated por contexto - FASE 10.5';