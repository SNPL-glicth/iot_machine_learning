"""Document Analysis Service.

Universal document analyzer that processes any content type
and generates adaptive conclusions.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """Universal document analyzer."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.logger = logger
    
    def analyze(
        self,
        document_id: str,
        content_type: str,
        normalized_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze document and generate conclusion.
        
        Args:
            document_id: UUID of document
            content_type: Type of content (tabular, text, image, audio, binary)
            normalized_payload: Parsed and normalized data
            
        Returns:
            Analysis result with conclusion, triggers, thresholds
        """
        start_time = time.time()
        
        try:
            if content_type == "tabular":
                result = self._analyze_tabular(normalized_payload)
            elif content_type == "text":
                result = self._analyze_text(normalized_payload)
            elif content_type == "image":
                result = self._analyze_image(normalized_payload)
            elif content_type == "audio":
                result = self._analyze_audio(normalized_payload)
            else:
                result = self._analyze_binary(normalized_payload)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return {
                "document_id": document_id,
                "content_type": content_type,
                "analysis": result["analysis"],
                "adaptive_thresholds": result.get("adaptive_thresholds", {}),
                "conclusion": result["conclusion"],
                "confidence": result.get("confidence", 0.85),
                "processing_time_ms": processing_time_ms,
            }
        except Exception as e:
            logger.exception(f"[DOCUMENT-ANALYZER] Error analyzing document {document_id}: {e}")
            raise
    
    def _analyze_tabular(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tabular data (CSV, Excel)."""
        data = payload.get("data", {})
        row_count = data.get("row_count", 0)
        numeric_columns = data.get("numeric_columns", [])
        headers = data.get("headers", [])
        sample_rows = data.get("sample_rows", [])
        
        analysis = {
            "patterns": [],
            "anomalies": [],
            "predictions": [],
            "triggers_activated": [],
        }
        
        # Analyze numeric columns
        thresholds = {}
        numeric_stats = []
        
        for col in numeric_columns:
            values = [float(row.get(col, 0)) for row in sample_rows if row.get(col)]
            if not values:
                continue
            
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = variance ** 0.5
            
            numeric_stats.append({
                "column": col,
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
            })
            
            # Adaptive thresholds
            thresholds[f"{col}_warning"] = mean + 1.5 * std
            thresholds[f"{col}_critical"] = mean + 2.5 * std
            thresholds[f"{col}_min"] = mean - 2 * std
            
            # Check for triggers
            for value in values:
                if value > mean + 2.5 * std:
                    analysis["triggers_activated"].append({
                        "type": "critical",
                        "field": col,
                        "value": value,
                        "threshold": mean + 2.5 * std,
                        "message": f"{col} superó umbral crítico",
                    })
                elif value > mean + 1.5 * std:
                    analysis["triggers_activated"].append({
                        "type": "warning",
                        "field": col,
                        "value": value,
                        "threshold": mean + 1.5 * std,
                        "message": f"{col} superó umbral de advertencia",
                    })
        
        # Generate conclusion
        conclusion_parts = []
        conclusion_parts.append(f"Documento tabular con {row_count} registros y {len(headers)} columnas.")
        
        if numeric_columns:
            conclusion_parts.append(f"Se detectaron {len(numeric_columns)} columnas numéricas: {', '.join(numeric_columns[:3])}.")
        
        if analysis["triggers_activated"]:
            critical_count = sum(1 for t in analysis["triggers_activated"] if t["type"] == "critical")
            warning_count = sum(1 for t in analysis["triggers_activated"] if t["type"] == "warning")
            
            if critical_count > 0:
                conclusion_parts.append(f"⚠️ Se detectaron {critical_count} valores críticos.")
            if warning_count > 0:
                conclusion_parts.append(f"Se detectaron {warning_count} advertencias.")
            
            conclusion_parts.append("Se recomienda revisar los registros marcados.")
        else:
            conclusion_parts.append("Todos los valores están dentro de rangos normales.")
        
        conclusion = " ".join(conclusion_parts)
        
        return {
            "analysis": {
                **analysis,
                "numeric_stats": numeric_stats,
            },
            "adaptive_thresholds": thresholds,
            "conclusion": conclusion,
            "confidence": 0.87,
        }
    
    def _analyze_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text content."""
        data = payload.get("data", {})
        word_count = data.get("word_count", 0)
        char_count = data.get("char_count", 0)
        paragraph_count = data.get("paragraph_count", 0)
        full_text = data.get("full_text", "")
        
        # Simple urgency detection by keywords
        urgency_keywords_es = ["error", "falla", "crítico", "alerta", "urgente", "caída", "pérdida", "crisis"]
        urgency_keywords_en = ["error", "failure", "critical", "alert", "urgent", "down", "loss", "crisis"]
        
        text_lower = full_text.lower()
        urgency_count = sum(1 for kw in urgency_keywords_es + urgency_keywords_en if kw in text_lower)
        urgency_score = min(1.0, urgency_count / 10.0)
        
        # Sentiment (very basic)
        positive_words = ["bueno", "excelente", "éxito", "mejora", "bien", "good", "excellent", "success"]
        negative_words = ["malo", "error", "problema", "falla", "bad", "error", "problem", "failure"]
        
        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)
        
        if negative_count > positive_count:
            sentiment = "negative"
        elif positive_count > negative_count:
            sentiment = "positive"
        else:
            sentiment = "neutral"
        
        analysis = {
            "sentiment": sentiment,
            "urgency_score": urgency_score,
            "keywords": urgency_keywords_es + urgency_keywords_en,
            "entities": [],
            "triggers_activated": [],
        }
        
        # Triggers based on urgency
        if urgency_score > 0.7:
            analysis["triggers_activated"].append({
                "type": "critical",
                "field": "urgency",
                "value": urgency_score,
                "threshold": 0.7,
                "message": "Urgencia alta detectada en el texto",
            })
        elif urgency_score > 0.4:
            analysis["triggers_activated"].append({
                "type": "warning",
                "field": "urgency",
                "value": urgency_score,
                "threshold": 0.4,
                "message": "Urgencia moderada detectada en el texto",
            })
        
        # Generate conclusion
        conclusion_parts = []
        conclusion_parts.append(f"Documento de texto con {word_count} palabras y {paragraph_count} párrafos.")
        conclusion_parts.append(f"Sentimiento: {sentiment}.")
        
        if urgency_score > 0.7:
            conclusion_parts.append(f"⚠️ Urgencia alta detectada (score: {urgency_score:.2f}).")
            conclusion_parts.append("Se recomienda acción inmediata.")
        elif urgency_score > 0.4:
            conclusion_parts.append(f"Urgencia moderada detectada (score: {urgency_score:.2f}).")
        else:
            conclusion_parts.append("No se detectaron indicadores de urgencia.")
        
        conclusion = " ".join(conclusion_parts)
        
        return {
            "analysis": analysis,
            "adaptive_thresholds": {
                "urgency_warning": 0.4,
                "urgency_critical": 0.7,
            },
            "conclusion": conclusion,
            "confidence": 0.75,
        }
    
    def _analyze_image(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image metadata."""
        data = payload.get("data", {})
        width = data.get("width", 0)
        height = data.get("height", 0)
        format_type = data.get("format", "unknown")
        file_size_kb = data.get("file_size_kb", 0)
        
        analysis = {
            "metadata": data,
            "triggers_activated": [],
        }
        
        conclusion = f"Imagen {format_type} de {width}x{height} píxeles, {file_size_kb} KB. Sin análisis de contenido visual disponible."
        
        return {
            "analysis": analysis,
            "adaptive_thresholds": {},
            "conclusion": conclusion,
            "confidence": 1.0,
        }
    
    def _analyze_audio(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audio metadata."""
        data = payload.get("data", {})
        duration = data.get("duration_seconds", 0)
        bitrate = data.get("bitrate_kbps", 0)
        format_type = data.get("format", "unknown")
        
        analysis = {
            "metadata": data,
            "triggers_activated": [],
        }
        
        conclusion = f"Audio {format_type} de {duration}s, {bitrate} kbps. Sin análisis de contenido de audio disponible."
        
        return {
            "analysis": analysis,
            "adaptive_thresholds": {},
            "conclusion": conclusion,
            "confidence": 1.0,
        }
    
    def _analyze_binary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze binary file."""
        data = payload.get("data", {})
        size_bytes = data.get("size_bytes", 0)
        mime_type = data.get("mime_type", "application/octet-stream")
        
        analysis = {
            "metadata": data,
            "triggers_activated": [],
        }
        
        conclusion = f"Archivo binario ({mime_type}) de {size_bytes} bytes. No se puede analizar el contenido."
        
        return {
            "analysis": analysis,
            "adaptive_thresholds": {},
            "conclusion": conclusion,
            "confidence": 1.0,
        }
