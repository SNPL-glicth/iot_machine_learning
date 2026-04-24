"""AnalysisResultAdapter — implementación de AnalysisDataPort.

Consulta analysis_results en zenin_db usando ZeninDbConnection existente.
"""

from __future__ import annotations

import json
import logging
from typing import List, Dict, Any, Optional

from sqlalchemy import text

from iot_machine_learning.domain.ports.analysis_data_port import (
    AnalysisDataPort,
    AnalysisRecord,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
    ZeninDbConnection,
)

logger = logging.getLogger(__name__)


class AnalysisResultAdapter(AnalysisDataPort):
    """Adaptador para consultar resultados de análisis."""
    
    def get_critical(self, tenant_id: str, limit: int = 5) -> List[AnalysisRecord]:
        """Documentos con severity critical o warning."""
        query = text("""
            SELECT TOP (:limit) 
                Id, OriginalFilename, Classification, Status,
                MlResult, Conclusion, MaxSeverity, AnalyzedAt
            FROM analysis_results
            WHERE TenantId = :tenant_id 
                AND MaxSeverity IN ('critical', 'warning')
            ORDER BY AnalyzedAt DESC
        """)
        return self._execute_query(query, {"tenant_id": tenant_id, "limit": limit})
    
    def get_recent(self, tenant_id: str, limit: int = 5) -> List[AnalysisRecord]:
        """Documentos más recientes."""
        query = text("""
            SELECT TOP (:limit)
                Id, OriginalFilename, Classification, Status,
                MlResult, Conclusion, MaxSeverity, AnalyzedAt
            FROM analysis_results
            WHERE TenantId = :tenant_id
            ORDER BY AnalyzedAt DESC
        """)
        return self._execute_query(query, {"tenant_id": tenant_id, "limit": limit})
    
    def get_by_domain(self, tenant_id: str, domain: str) -> List[AnalysisRecord]:
        """Documentos por dominio (extraído de MlResult)."""
        query = text("""
            SELECT TOP 50
                Id, OriginalFilename, Classification, Status,
                MlResult, Conclusion, MaxSeverity, AnalyzedAt
            FROM analysis_results
            WHERE TenantId = :tenant_id
                AND MlResult LIKE '%' + :domain + '%'
            ORDER BY AnalyzedAt DESC
        """)
        return self._execute_query(query, {"tenant_id": tenant_id, "domain": domain})
    
    def get_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Resumen: total, por severidad, promedio confianza."""
        try:
            with ZeninDbConnection.get_connection() as conn:
                # Total por severidad
                severity_query = text("""
                    SELECT MaxSeverity, COUNT(*) as count
                    FROM analysis_results
                    WHERE TenantId = :tenant_id
                    GROUP BY MaxSeverity
                """)
                severity_result = conn.execute(severity_query, {"tenant_id": tenant_id})
                by_severity = {row[0]: row[1] for row in severity_result}
                
                # Total general
                total = sum(by_severity.values())
                
                return {
                    "total": total,
                    "critical": by_severity.get("critical", 0),
                    "warning": by_severity.get("warning", 0),
                    "info": by_severity.get("info", 0),
                    "success": by_severity.get("success", 0),
                }
        except Exception as e:
            logger.error("[ANALYSIS_ADAPTER] Error getting summary: %s", e)
            return {"total": 0, "critical": 0, "warning": 0, "info": 0, "success": 0}
    
    def _execute_query(
        self, 
        query, 
        params: Dict[str, Any]
    ) -> List[AnalysisRecord]:
        """Ejecuta query y mapea resultados."""
        records: List[AnalysisRecord] = []
        try:
            with ZeninDbConnection.get_connection() as conn:
                result = conn.execute(query, params)
                for row in result:
                    records.append(self._map_row_to_record(row))
        except Exception as e:
            logger.error("[ANALYSIS_ADAPTER] Query error: %s", e)
        return records
    
    def _map_row_to_record(self, row) -> AnalysisRecord:
        """Mapea fila SQL a AnalysisRecord."""
        # Parsear MlResult JSON
        ml_result = self._parse_ml_result(row.MlResult)
        
        return AnalysisRecord(
            id=str(row.Id),
            filename=row.OriginalFilename or "Sin nombre",
            severity=row.MaxSeverity or "unknown",
            domain=ml_result.get("domain", "general"),
            urgency=ml_result.get("urgency_score", 0.0),
            confidence=ml_result.get("confidence", 0.0),
            conclusion=row.Conclusion or ml_result.get("conclusion", "Sin conclusión"),
            entities=ml_result.get("entities", []),
            actions=ml_result.get("actions", []),
            analyzed_at=row.AnalyzedAt.isoformat() if row.AnalyzedAt else "",
        )
    
    def _parse_ml_result(self, ml_result_json: Optional[str]) -> Dict[str, Any]:
        """Parsea MlResult JSON."""
        if not ml_result_json:
            return {}
        try:
            return json.loads(ml_result_json)
        except json.JSONDecodeError:
            return {}
