"""ResponseBuilder — orquesta construcción de respuestas.

Responsabilidad única: coordinar selector + formatter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from .template_selector import TemplateSelector
from .data_formatter import DataFormatter
from .chat_context_manager import ChatContextManager
from iot_machine_learning.domain.ports.analysis_data_port import AnalysisRecord


@dataclass
class ChatResponse:
    text: str
    response_type: str
    metadata: Dict[str, Any]


class ResponseBuilder:
    """Orquesta la construcción de respuestas con datos reales."""
    
    def __init__(self, selector: TemplateSelector, formatter: DataFormatter) -> None:
        self._selector = selector
        self._formatter = formatter
    
    def build(
        self,
        intent: str,
        records: List[AnalysisRecord],
        summary: Dict[str, Any],
    ) -> ChatResponse:
        """Construye respuesta según intención y datos."""
        if intent == "metrics_query":
            return self._build_summary(summary)
        elif intent in ("document_query", "analysis_query"):
            return self._build_records(records, intent)
        elif intent == "conversational":
            return self._build_conversational()
        else:
            return ChatResponse(
                text=self._selector.select("error"),
                response_type="error",
                metadata={}
            )
    
    def _build_summary(self, summary: Dict[str, Any]) -> ChatResponse:
        template = self._selector.select("summary", ["Resumen: {total} documentos"])
        text = self._formatter.format_summary(template, summary)
        return ChatResponse(text=text, response_type="metrics", metadata=summary)
    
    def _build_records(
        self, records: List[AnalysisRecord], intent: str
    ) -> ChatResponse:
        if not records:
            text = self._selector.select("no_critical", ["No hay documentos"])
            return ChatResponse(text=text, response_type=intent, metadata={"records": 0})
        
        cat = "critical_documents" if intent == "document_query" else "recent_documents"
        template = self._selector.select(cat, ["Documentos:\n{list}"])
        item_tmpl = self._selector.select("document_item", ["• {filename} — {severity}"])
        
        text = self._formatter.format_records(template, item_tmpl, records)
        return ChatResponse(
            text=text, response_type=intent,
            metadata={"records": len(records)}
        )
    
    def _build_conversational(self) -> ChatResponse:
        text = self._selector.select("greeting")
        return ChatResponse(text=text, response_type="conversational", metadata={})
