"""Production impact assessor service for operational context."""

from __future__ import annotations

import logging
from typing import Tuple

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..models.work_shift import ProductionImpact

logger = logging.getLogger(__name__)


class ProductionImpactAssessor:
    """Assesses production impact of device issues."""
    
    # Tipos de dispositivo por criticidad
    CRITICAL_TYPES = {"server", "datacenter", "production", "critical"}
    HIGH_TYPES = {"hvac", "power", "ups", "network"}
    MEDIUM_TYPES = {"sensor_hub", "gateway", "monitoring"}
    
    def __init__(self, conn: Connection):
        self._conn = conn
    
    def assess_impact(
        self,
        device_id: int,
        base_severity: str,
    ) -> Tuple[ProductionImpact, list[str]]:
        """Evalúa el impacto en producción del dispositivo afectado."""
        affected_processes = []
        
        try:
            row = self._conn.execute(
                text("""
                    SELECT device_type, metadata, name
                    FROM dbo.devices
                    WHERE id = :device_id
                """),
                {"device_id": device_id},
            ).fetchone()
            
            if row:
                device_type = str(row.device_type or "").lower()
                device_name = str(row.name or "")
                
                if any(t in device_type for t in self.CRITICAL_TYPES):
                    affected_processes.append(f"Producción ({device_name})")
                    return ProductionImpact.CRITICAL, affected_processes
                
                if any(t in device_type for t in self.HIGH_TYPES):
                    affected_processes.append(f"Infraestructura ({device_name})")
                    return ProductionImpact.HIGH, affected_processes
                
                if any(t in device_type for t in self.MEDIUM_TYPES):
                    affected_processes.append(f"Monitoreo ({device_name})")
                    return ProductionImpact.MEDIUM, affected_processes
        
        except Exception as e:
            logger.warning("Failed to assess impact: %s", str(e))
        
        # Ajustar por severidad base
        if base_severity.lower() == "critical":
            return ProductionImpact.HIGH, affected_processes
        
        return ProductionImpact.LOW, affected_processes
    
    def get_maintenance_history(self, device_id: int) -> dict:
        """Obtiene historial de mantenimiento del dispositivo."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT 
                        DATEDIFF(day, MAX(resolved_at), GETDATE()) AS days_since_last,
                        COUNT(CASE WHEN status = 'active' THEN 1 END) AS pending
                    FROM dbo.ml_events
                    WHERE device_id = :device_id
                      AND resolved_at IS NOT NULL
                """),
                {"device_id": device_id},
            ).fetchone()
            
            if row:
                return {
                    "days_since_last": int(row.days_since_last) if row.days_since_last else None,
                    "pending_tasks": int(row.pending) if row.pending else 0,
                }
        except Exception as e:
            logger.warning("Failed to get maintenance history: %s", str(e))
        
        return {"days_since_last": None, "pending_tasks": 0}
    
    def get_recent_incidents(self, sensor_id: int, days: int = 7) -> int:
        """Obtiene cantidad de incidentes recientes para el sensor."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT COUNT(*) AS cnt
                    FROM dbo.ml_events
                    WHERE sensor_id = :sensor_id
                      AND created_at >= DATEADD(day, -:days, GETDATE())
                """),
                {"sensor_id": sensor_id, "days": days},
            ).fetchone()
            
            if row:
                return int(row.cnt)
        except Exception as e:
            logger.warning("Failed to get recent incidents: %s", str(e))
        
        return 0
