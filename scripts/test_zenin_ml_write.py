"""Script de prueba para validar escritura a zenin_ml.predictions.

Ejecutar:
    python scripts/test_zenin_ml_write.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import create_engine, text

from infrastructure.persistence.sql.dual_write_storage import DualWriteStorageAdapter
from domain.entities.prediction import Prediction, PredictionConfidence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Prueba de escritura a zenin_ml.predictions."""
    
    # 1. Conectar a DB
    connection_string = (
        "mssql+pyodbc://sa:Sandevistan2510@localhost:1434/zenin_db"
        "?driver=ODBC+Driver+18+for+SQL+Server"
        "&TrustServerCertificate=yes"
    )
    
    logger.info("Conectando a zenin_db...")
    engine = create_engine(connection_string, echo=False)
    
    with engine.connect() as conn:
        # 2. Verificar tablas
        logger.info("Verificando esquemas...")
        result = conn.execute(text("""
            SELECT 
                SCHEMA_NAME(schema_id) as schema_name,
                name as table_name
            FROM sys.tables
            WHERE SCHEMA_NAME(schema_id) IN ('dbo', 'zenin_ml')
            AND name = 'predictions'
            ORDER BY schema_name
        """))
        
        for row in result:
            logger.info(f"  Tabla encontrada: {row[0]}.{row[1]}")
        
        # 3. Crear predicción de prueba
        logger.info("\nCreando predicción de prueba...")
        prediction = Prediction(
            series_id="999",  # sensor_id legacy
            predicted_value=25.5,
            confidence_score=0.85,
            confidence_level=PredictionConfidence.HIGH,
            trend="increasing",
            engine_name="test_engine",
            horizon_steps=1,
            confidence_interval=(24.0, 27.0),
            feature_contributions={"temperature": 0.6, "humidity": 0.4},
            audit_trace_id=uuid4(),
            metadata={
                "test": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "regime": "stable",
            },
        )
        
        # 4. Escribir con dual-write adapter
        logger.info("Escribiendo con DualWriteStorageAdapter...")
        adapter = DualWriteStorageAdapter(conn, enable_zenin_ml=True)
        
        try:
            legacy_id = adapter.save_prediction(prediction)
            logger.info(f"✓ Escritura exitosa! Legacy ID: {legacy_id}")
            
            # 5. Verificar registros
            logger.info("\nVerificando registros...")
            
            # dbo.predictions
            legacy_count = conn.execute(text("""
                SELECT COUNT(*) FROM dbo.predictions
                WHERE sensor_id = 999
            """)).scalar()
            logger.info(f"  dbo.predictions: {legacy_count} registros con sensor_id=999")
            
            # zenin_ml.predictions
            zenin_count = conn.execute(text("""
                SELECT COUNT(*) FROM zenin_ml.predictions
                WHERE Metadata LIKE '%"sensor_id": 999%'
            """)).scalar()
            logger.info(f"  zenin_ml.predictions: {zenin_count} registros con sensor_id=999")
            
            # Mostrar último registro zenin_ml
            zenin_latest = conn.execute(text("""
                SELECT TOP 1
                    Id, SeriesId, TenantId, PredictedValue,
                    ConfidenceScore, EngineName, PredictedAt
                FROM zenin_ml.predictions
                WHERE Metadata LIKE '%"sensor_id": 999%'
                ORDER BY PredictedAt DESC
            """)).fetchone()
            
            if zenin_latest:
                logger.info(f"\n  Último registro zenin_ml:")
                logger.info(f"    Id: {zenin_latest[0]}")
                logger.info(f"    SeriesId: {zenin_latest[1]}")
                logger.info(f"    TenantId: {zenin_latest[2]}")
                logger.info(f"    PredictedValue: {zenin_latest[3]}")
                logger.info(f"    ConfidenceScore: {zenin_latest[4]}")
                logger.info(f"    EngineName: {zenin_latest[5]}")
                logger.info(f"    PredictedAt: {zenin_latest[6]}")
            
            logger.info("\n✓ PRUEBA EXITOSA: Dual-write funcionando correctamente")
            
        except Exception as exc:
            logger.error(f"✗ Error en dual-write: {exc}", exc_info=True)
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
