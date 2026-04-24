"""DataFormatter — inyecta datos reales en plantillas.

Responsabilidad única: formatear AnalysisRecord en strings.
"""

from __future__ import annotations

from typing import List, Dict, Any

from iot_machine_learning.domain.ports.analysis_data_port import AnalysisRecord


class DataFormatter:
    """Formatea registros de análisis en lenguaje natural."""
    
    def format_summary(self, template: str, summary: Dict[str, Any]) -> str:
        """Inyecta conteos en plantilla."""
        return template.format(
            total=summary.get("total", 0),
            critical=summary.get("critical", 0),
            warning=summary.get("warning", 0),
            info=summary.get("info", 0),
            success=summary.get("success", 0),
        )
    
    def format_records(
        self,
        template: str,
        item_template: str,
        records: List[AnalysisRecord]
    ) -> str:
        """Inyecta lista de documentos en plantilla."""
        if not records:
            return "No hay documentos disponibles."
        
        items = []
        for r in records[:5]:  # Máximo 5
            date = r.analyzed_at[:10] if r.analyzed_at else "fecha desconocida"
            item = item_template.format(
                filename=r.filename,
                severity=r.severity,
                urgency=round(r.urgency, 2),
                date=date
            )
            items.append(item)
        
        return template.format(count=len(records), list="\n".join(items))
    
    def truncate(self, text: str, max_len: int = 50) -> str:
        """Trunca texto largo."""
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."
