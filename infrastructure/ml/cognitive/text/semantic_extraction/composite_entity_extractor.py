"""RegexEntityExtractor — extractor de entidades basado en regex (sin NLP).

DESIGN NOTE: Diseño nuevo, no reconstrucción.
El módulo semantic_extraction/ original fue eliminado sin rastro en git
(corrupción de objetos). No existe contrato original conocido.

Decisión: implementación regex-only (sin spaCy, sin modelos ML)
siguiendo el enfoque "NER sin modelo" acordado.
"""

from __future__ import annotations

import re
import time
from typing import Any

from iot_machine_learning.domain.entities.semantic_extraction import (
    EntityType,
    MetricAttributes,
    SemanticEnrichmentResult,
    SemanticEntity,
)
from iot_machine_learning.domain.ports.semantic_extraction_port import (
    EntityExtractionResult,
    EntityExtractorPort,
)

_EQUIPMENT_PREFIX = re.compile(
    r"\b(?:COMP|VLV|MOT|PUMP|CMP|BLR|GEN|TX|HV|NODE|SERVER|ROUTER|SWITCH)(-?\w+)\b",
    re.IGNORECASE,
)
_METRIC_VALUE_PATTERN = re.compile(
    r"(?<!\w)(\d+\.?\d*)\s*"
    r"(°[CF]|%|PSI|BAR|°[CF]|k?Pa|k?Wh|MW|kW|m³|m3|l/min|rpm|Hz|mm|cm|kg)\b(?![\w/])",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}-\d{2}-\d{2})\b",
)
_CAPITALIZED_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_ALERT_PATTERN = re.compile(
    r"\b(critical|emergency|warning|danger|alarm|error|failure|shutdown|trip|fault)\b",
    re.IGNORECASE,
)
_OPERATIONAL_PATTERN = re.compile(
    r"\b(startup|maintenance|inspection|calibration|restart|override|bypass)\b",
    re.IGNORECASE,
)


# Located in common English words that happen to be capitalized
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "this", "that", "these", "those", "what", "which", "when", "where",
    "who", "whom", "whose", "why", "how", "all", "each", "every", "both",
    "few", "many", "much", "some", "any", "no", "none", "other", "another",
    "temperature", "pressure", "level", "speed", "rate",
    "voltage", "power", "frequency", "analysis", "report", "system", "data",
    "test", "result", "status", "alert", "alarm", "warning", "error", "fault",
    "time", "date", "value", "range", "mode", "type", "source",
    "during", "after", "before", "while", "since", "until", "within",
    "previous", "following", "initial", "final", "total", "average",
    "minimum", "maximum", "recommended", "required", "estimated", "expected",
    "observed", "measured", "calculated", "network", "storage", "memory",
    "safety", "routine", "immediate", "automatic", "manual",
    "friday", "monday", "tuesday", "wednesday", "thursday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "weather", "wind", "clouds", "rain", "celsius",
    "manufacturers", "industry", "industrial", "transformation",
    "implementation", "deployment", "integration", "replacement",
    "evacuation", "investigation", "inspection", "calibration",
    "operator", "personnel", "management", "maintenance",
    "based", "made", "used", "found", "seen", "taken",
    "last", "next", "first", "second", "third",
    "approximately", "significantly", "previously", "typically",
    "tank", "valve", "pump", "motor", "bearing", "sensor",
    "zone", "area", "line", "floor", "room",
    "gas", "oil", "water", "steam", "air", "heat",
})


def _is_sentence_start(text: str, pos: int) -> bool:
    """Check if position is at the start of a sentence or paragraph."""
    if pos == 0:
        return True
    before = text[max(pos - 5, 0):pos].lstrip()
    if not before:
        return True
    return before[-1] in ".!?\n"


class RegexEntityExtractor(EntityExtractorPort):
    """Extractor que usa exclusivamente regex para detectar entidades.

    Sin dependencias de NLP, sin modelos, sin I/O.
    Detecta: equipos (códigos alfanuméricos con dígitos), métricas numéricas
    con unidades multi-char, fechas ISO, palabras clave de alerta/operación,
    y entidades nombradas capitalizadas (con filtro de stop-words y detección
    de inicio de oración para evitar falsos positivos).
    """

    def __init__(self, domain_hint: str = "general") -> None:
        self._domain = domain_hint

    @property
    def extractor_name(self) -> str:
        return "regex_entity_extractor"

    def supports_domain(self, domain: str) -> bool:
        return domain in ("general", "infrastructure", "industrial", "security", "trading")

    def extract(
        self,
        text: str,
        domain_hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> EntityExtractionResult:
        t0 = time.monotonic()

        if not text or not text.strip():
            return EntityExtractionResult.empty()

        entities: list[SemanticEntity] = []
        seen: set[str] = set()

        for match in _EQUIPMENT_PREFIX.finditer(text):
            raw = match.group()
            suffix = match.group(1).lstrip("-")
            if suffix.isalpha() and len(suffix) > 3:
                continue
            norm = raw.upper().replace("-", "")
            if norm.upper() in seen:
                continue
            seen.add(norm.upper())
            entities.append(SemanticEntity(
                text=raw,
                normalized=norm,
                entity_type=EntityType.EQUIPMENT,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.85,
            ))

        for match in _METRIC_VALUE_PATTERN.finditer(text):
            raw = match.group()
            norm = raw.strip()
            if norm.upper() in seen:
                continue
            seen.add(norm.upper())
            value_str, unit = match.groups()
            unit_lower = unit.lower().replace("°", "").replace("³", "3")
            metric_class = _classify_metric(unit_lower)
            try:
                value_float = float(value_str)
            except ValueError:
                value_float = 0.0
            entities.append(SemanticEntity(
                text=raw,
                normalized=norm,
                entity_type=EntityType.METRIC,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.8,
                attributes=MetricAttributes(
                    value=value_float,
                    unit=unit,
                    metric_class=metric_class,
                ).to_dict(),
            ))

        for match in _DATE_PATTERN.finditer(text):
            raw = match.group()
            if raw in seen:
                continue
            seen.add(raw)
            entities.append(SemanticEntity(
                text=raw,
                normalized=raw,
                entity_type=EntityType.TEMPORAL,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.7,
            ))

        for match in _ALERT_PATTERN.finditer(text):
            raw = match.group()
            key = raw.lower()
            if key in seen:
                continue
            seen.add(key)
            is_crit = key in ("critical", "emergency", "failure", "shutdown", "trip")
            entities.append(SemanticEntity(
                text=raw,
                normalized=raw.lower(),
                entity_type=EntityType.ALERT,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.9 if is_crit else 0.7,
            ))

        for match in _OPERATIONAL_PATTERN.finditer(text):
            raw = match.group()
            key = raw.lower()
            if key in seen:
                continue
            seen.add(key)
            entities.append(SemanticEntity(
                text=raw,
                normalized=raw.lower(),
                entity_type=EntityType.OPERATIONAL,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.7,
            ))

        for match in _CAPITALIZED_PATTERN.finditer(text):
            raw = match.group()
            key = raw.lower()

            if len(raw) < 3:
                continue
            if key in seen:
                continue

            if key in _STOP_WORDS:
                continue
            if _is_sentence_start(text, match.start()):
                continue

            seen.add(key)
            entities.append(SemanticEntity(
                text=raw,
                normalized=raw.strip(),
                entity_type=EntityType.LOCATION,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.4,
            ))

        elapsed = (time.monotonic() - t0) * 1000
        domain = domain_hint or self._domain

        return EntityExtractionResult(
            entities=entities,
            domain_detected=domain,
            confidence_aggregate=0.6 if entities else 0.0,
            extraction_duration_ms=round(elapsed, 2),
        )

    def to_enrichment_result(
        self,
        extraction: EntityExtractionResult,
        urgency_context: float = 0.0,
    ) -> SemanticEnrichmentResult:
        """Convertir EntityExtractionResult → SemanticEnrichmentResult.

        Bridge method para enrich_phase.py — no parte del port.
        """
        from iot_machine_learning.application.semantic_extraction import (
            EntityPrioritizer,
        )
        from iot_machine_learning.domain.entities.semantic_extraction import (
            EnrichmentContext,
        )

        ctx = EnrichmentContext(
            domain=extraction.domain_detected,
            urgency_score=urgency_context,
        )
        prioritizer = EntityPrioritizer()
        prioritized = prioritizer.prioritize(extraction.entities, ctx)
        return prioritized.to_enrichment_result(extraction)


def _classify_metric(unit: str) -> str:
    if unit in ("c", "f"):
        return "temperature"
    if unit in ("psi", "bar", "kpa", "pa"):
        return "pressure"
    if unit in ("%", "percent"):
        return "percentage"
    if unit in ("m3", "m³", "l/min"):
        return "flow"
    if unit in ("rpm", "hz"):
        return "speed"
    if unit in ("mm", "cm", "m"):
        return "dimension"
    if unit in ("kg", "g"):
        return "weight"
    if unit in ("a", "v", "w", "mw", "kw", "kwh"):
        return "electrical"
    return "unknown"
