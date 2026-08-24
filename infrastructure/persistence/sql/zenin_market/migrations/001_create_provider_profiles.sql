-- 001_create_provider_profiles.sql
-- Tabla mínima de ProviderProfile para el circuito dominio → persistencia → dominio.
-- Idempotente: seguro de re-ejecutar.

CREATE TABLE IF NOT EXISTS provider_profiles (
    provider        VARCHAR(64)  NOT NULL,
    asset_class     VARCHAR(32)  NOT NULL,
    capabilities    JSON         NOT NULL,
    max_ws_symbols  INT          NULL,
    created_at      DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (provider, asset_class)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     VARCHAR(128) NOT NULL,
    applied_at  DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;