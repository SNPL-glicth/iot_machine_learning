"""GraphQL endpoint for UniversalAnalysisEngine — resolver semi-delgado.

Design decisions (documented from pre-implementation discussion):
- El cliente declara inputType explícitamente (TEXT o NUMERIC) — el resolver
  NO reimplementa clasificación de tipo, eso duplica al engine interno.
- Para TEXT: computa pre_computed_scores reales (sentiment, urgency, etc.)
  antes de delegar a UniversalAnalysisEngine — así el engine recibe
  datos enriquecidos y la conclusión incluye sentimiento/entidades reales.
- Para NUMERIC/otros: pasa pre_computed_scores vacío — el engine lo maneja
  solo, y build_conclusion() ya no muestra "Sentiment: neutral" engañoso
  (fixeado en commit 5a1565d).
- UniversalAnalysisEngine es stateless (confirmado) — una instancia
  module-level reusable entre requests.
- Feature flag: ML_ENABLE_GRAPHQL_API (default false) controla el montaje
  del router en main.py (try/except).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

from iot_machine_learning.infrastructure.ml.cognitive.universal import (
    UniversalAnalysisEngine,
    UniversalContext,
)

logger = logging.getLogger(__name__)

# ── Module-level engine instance (stateless, reusable) ──────────────
# NOTE: This module only loads when ML_ENABLE_GRAPHQL_API=true (main.py).
# TEXT perception must be active for the pipeline to generate perceptions
# and avoid fallback.  We set it here so the engine instance captures it.
import os as _os
from ml_service.config.feature_flags import reset_feature_flags as _reset_flags
_os.environ.setdefault("ML_ENABLE_TEXT_PERCEPTION", "true")
_reset_flags()

_engine = UniversalAnalysisEngine(
    enable_plasticity=False,
    enable_monte_carlo=False,
    enable_semantic_enrichment=False,
    budget_ms=5000,
)

# ── Schema ──────────────────────────────────────────────────────────

@strawberry.enum
class InputTypeEnum(Enum):
    TEXT = "text"
    NUMERIC = "numeric"


@strawberry.input
class AnalysisInput:
    raw_data: str
    input_type: InputTypeEnum
    domain_hint: Optional[str] = strawberry.UNSET
    tenant_id: str
    document_id: Optional[str] = strawberry.UNSET


@strawberry.type
class AnalysisResult:
    domain: str
    severity: str
    risk_level: str
    confidence: float
    conclusion: str
    entities: list[str]


# ── Resolver ────────────────────────────────────────────────────────

def _extract_entities_regex(text: str) -> list[str]:
    """Fallback regex entity extraction — full match + case-insensitive dedup."""
    import re
    seen: set[str] = set()
    result: list[str] = []

    def _add(raw: str) -> None:
        key = raw.upper().replace("-", "").replace(" ", "")
        if key not in seen:
            seen.add(key)
            result.append(raw)

    for m in re.finditer(r'\b\d+\s*°[CF]\b', text):
        _add(m.group())

    for m in re.finditer(r'\b(NODE|TMP|SERVER|ROUTER|SWITCH)-\w+\b', text):
        _add(m.group())

    _EQUIPMENT = re.compile(
        r'\b(COMP|VLV|MOT|PUMP|CMP|BLR|GEN|TX|HV)[-]?[A-Z0-9]+\b',
        re.IGNORECASE,
    )
    for m in _EQUIPMENT.finditer(text):
        raw = m.group()
        suffix = raw[len(m.group(1)):].lstrip("-")
        if suffix.isalpha() and len(suffix) > 3:
            continue
        _add(raw)

    for m in re.finditer(
        r'\$[\d,]+(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})+\s*(?:USD|EUR|USD\$|\$)\b', text
    ):
        _add(m.group())

    for m in re.finditer(r'\b\d+%\b', text):
        _add(m.group())

    for m in re.finditer(r'\bSLA\s+\d+\.?\d*%?\b', text, re.IGNORECASE):
        _add(m.group())

    return result


def _build_pre_computed_scores(raw_data: str) -> dict:
    """Compute text analysis scores for TEXT inputs.

    Reusa los mismos sub-analyzers que routes_query.py (Paso 1).
    """
    from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers import (
        compute_sentiment,
        compute_urgency,
        compute_readability,
        compute_text_structure,
    )
    from iot_machine_learning.infrastructure.ml.cognitive.text.text_pattern import (
        detect_text_patterns,
    )

    word_count = len(raw_data.split())
    paragraph_count = max(1, raw_data.count("\n\n") + 1)

    sentiment = compute_sentiment(raw_data)
    urgency = compute_urgency(raw_data)
    readability = compute_readability(raw_data, word_count)
    structural = compute_text_structure(readability.sentences if readability else [])
    patterns = detect_text_patterns(readability.sentences if readability else [])

    # Extract entities via regex fallback (same pattern as universal_bridge.py)
    entities = _extract_entities_regex(raw_data)

    return {
        "sentiment_score": sentiment.score if sentiment else 0.0,
        "sentiment_label": sentiment.label if sentiment else "neutral",
        "sentiment_positive_count": sentiment.positive_count if sentiment else 0,
        "sentiment_negative_count": sentiment.negative_count if sentiment else 0,
        "urgency_score": urgency.score if urgency else 0.0,
        "urgency_severity": urgency.severity if urgency else "info",
        "urgency_total_hits": urgency.total_hits if urgency else 0,
        "urgency_hits": urgency.hits if urgency else (),
        "word_count": word_count,
        "paragraph_count": paragraph_count,
        "entities": entities,
        "patterns": {
            "pattern_summary": {
                "urgency_regime": _urgency_regime(urgency.score if urgency else 0.0),
                "n_change_points": 1 if (urgency.score if urgency else 0.0) > 0.5 else 0,
                "n_spikes": 1 if (urgency.score if urgency else 0.0) > 0.7 else 0,
                "has_escalation": (urgency.score if urgency else 0.0) > 0.6,
                "improvement_points": 0 if (urgency.score if urgency else 0.0) > 0.5 else 1,
            }
        },
    }


def _urgency_regime(score: float) -> str:
    if score >= 0.8:
        return "critical"
    if score >= 0.5:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


@strawberry.type
class Query:
    @strawberry.field
    def analyze(self, input: AnalysisInput, info: Info) -> AnalysisResult:
        """Analyze any input via UniversalAnalysisEngine.

        Args:
            input: AnalysisInput with raw_data, input_type, optional domain_hint.

        Returns:
            AnalysisResult with domain, severity, confidence, conclusion, entities.
        """
        domain_hint = input.domain_hint if input.domain_hint is not strawberry.UNSET else ""
        document_id = input.document_id if input.document_id is not strawberry.UNSET else input.tenant_id

        # Build pre_computed_scores based on input type
        if input.input_type == InputTypeEnum.TEXT:
            pre_computed_scores = _build_pre_computed_scores(input.raw_data)
        else:
            pre_computed_scores = {}

        # Build context
        ctx = UniversalContext(
            series_id=document_id,
            tenant_id=input.tenant_id,
            domain_hint=domain_hint,
            budget_ms=5000.0,
        )

        try:
            # Run analysis
            result = _engine.analyze(
                raw_data=input.raw_data,
                ctx=ctx,
                pre_computed_scores=pre_computed_scores,
            )

            # Build human-readable conclusion
            from iot_machine_learning.ml_service.api.services.analysis.result_builder import (
                build_conclusion,
            )
            conclusion = build_conclusion(result)

            # Extract entities safely
            entities = list(result.analysis.get("entities", []))

            # Extract severity info
            severity = getattr(result.severity, "severity", "unknown")
            risk_level = getattr(result.severity, "risk_level", "UNKNOWN")

            return AnalysisResult(
                domain=result.domain,
                severity=severity,
                risk_level=risk_level,
                confidence=result.confidence,
                conclusion=conclusion,
                entities=[str(e) for e in entities],
            )

        except Exception as e:
            logger.error(f"graphql_analyze_failed: {e}", exc_info=True)
            return AnalysisResult(
                domain="general",
                severity="info",
                risk_level="NONE",
                confidence=0.0,
                conclusion="",
                entities=[],
            )


# ── Router ─────────────────────────────────────────────────────────

schema = strawberry.Schema(query=Query)

router = GraphQLRouter(schema, graphql_ide=True)
