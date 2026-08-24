-- 003_create_model_versions.sql
-- Versionado auditable del modelo ZENIN (FASE 8).
--
-- model_versions: cada versión es inmutable (append-only); contiene el
-- snapshot completo de pesos por contexto (JSON), la calibración
-- observada al crearla, la razón del cambio y el parent_version_id.
-- Solo una versión es activa (is_active=1); crear una versión nueva
-- desactiva la anterior (mismo transacción en el repo).
--
-- adaptation_proposals: TODO lo que el sistema consideró cambiar se
-- guarda, aceptado o rechazado, con el resultado de cada chequeo del
-- guardrail y el motivo del rechazo. Respuesta a "¿por qué ZENIN le
-- dio más peso a Momentum?" -> está almacenada, no inferida.

CREATE TABLE IF NOT EXISTS model_versions (
    version_id        INT AUTO_INCREMENT PRIMARY KEY,
    created_at        DOUBLE       NOT NULL,
    weights           JSON         NOT NULL,
    calibration       JSON         NOT NULL,
    reason            TEXT         NOT NULL,
    parent_version_id INT          NULL,
    proposal_id       VARCHAR(128) NULL,
    guard_checks      JSON         NULL,
    is_active         TINYINT(1)   NOT NULL DEFAULT 1,
    KEY idx_versions_parent (parent_version_id),
    KEY idx_versions_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS adaptation_proposals (
    proposal_id       VARCHAR(128) NOT NULL PRIMARY KEY,
    version_id        INT          NULL,
    expert            VARCHAR(64)  NOT NULL,
    regime            VARCHAR(32)  NULL,
    horizon_seconds   INT          NOT NULL,
    current_weight    DOUBLE       NOT NULL,
    proposed_weight   DOUBLE       NOT NULL,
    observed_reward   DOUBLE       NULL,
    calibration       DOUBLE       NULL,
    sample_size       INT          NOT NULL,
    reason            TEXT         NOT NULL,
    status            VARCHAR(16)  NOT NULL,
    rejected_reason   VARCHAR(255) NULL,
    guard_checks      JSON         NULL,
    parent_version_id INT          NULL,
    created_at        DOUBLE       NOT NULL,
    KEY idx_proposals_expert (expert),
    KEY idx_proposals_status (status),
    KEY idx_proposals_context (regime, horizon_seconds)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;