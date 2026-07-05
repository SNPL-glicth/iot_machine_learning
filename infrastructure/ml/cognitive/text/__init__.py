"""TextCognitiveEngine — análisis cognitivo para documentos de texto.

DESIGN NOTES: Reconstrucción vs Diseño Nuevo
==============================================

RECONSTRUCCIÓN (contratos confirmados por callers en text_analyzer.py):
  - TextAnalysisInput: 25+ campos → text_analyzer.py:127-154
  - TextAnalysisContext: document_id, tenant_id, filename, weaviate_url → text_analyzer.py:156
  - TextCognitiveEngine.analyze(inp, ctx) → objeto con .analysis (dict), .explanation (Explanation), .confidence (float)
  - .analysis usado como base por text_analyzer.py — agrega triggers_activated, semantic, explanation.to_dict()
  - .explanation.to_dict() → analysis["explanation"]
  - .confidence → round(.confidence, 3) en el retorno final

DISEÑO NUEVO (código original no disponible — git corrupto):
  - Estructura interna de analysis["cognitive"]: engine_weights, engine_perceptions, signal_profile
  - Lógica de fusión: weighted average de 4 engines (urgency 0.45, sentiment 0.20, structural 0.15, readability 0.10, pattern 0.10)
  - Cálculo de confianza: normalized entropy sobre los pesos de engine
  - Feature flag ML_ENABLE_TEXT_ANALYSIS evaluado internamente en analyze()
  - Explicación vía ExplanationBuilder (set_fusion + set_perceptions + build)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from iot_machine_learning.domain.entities.explainability.explanation import Explanation
from iot_machine_learning.infrastructure.ml.cognitive.explanation import ExplanationBuilder

logger = logging.getLogger(__name__)

# Feature flag key
_FLAG_KEY = "ML_ENABLE_TEXT_ANALYSIS"

# Fusion weights — diseño nuevo, no basado en código original
_WEIGHTS: dict[str, float] = {
    "text_urgency": 0.45,
    "text_sentiment": 0.20,
    "text_structural": 0.15,
    "text_readability": 0.10,
    "text_pattern": 0.10,
}


@dataclass(frozen=True)
class TextAnalysisInput:
    """Pre-computed text analysis scores para el motor cognitivo.

    Todos los campos son provistos por text_analyzer.py antes de llamar a analyze().
    """
    full_text: str = ""
    word_count: int = 0
    paragraph_count: int = 0

    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    sentiment_positive_count: int = 0
    sentiment_negative_count: int = 0

    urgency_score: float = 0.0
    urgency_severity: str = "info"
    urgency_total_hits: int = 0
    urgency_hits: tuple[str, ...] = ()

    readability_avg_sentence_length: float = 0.0
    readability_n_sentences: int = 0
    readability_vocabulary_richness: float = 0.0
    readability_embedded_numeric_count: int = 0
    readability_sentences: tuple[str, ...] = ()

    structural_regime: str = "unknown"
    structural_trend: str = "unknown"
    structural_stability: float = 0.0
    structural_noise: float = 0.0
    structural_available: bool = False

    pattern_n_patterns: int = 0
    pattern_change_points: tuple = ()
    pattern_spikes: tuple = ()
    pattern_available: bool = False
    pattern_summary: Any = None


@dataclass(frozen=True)
class TextAnalysisContext:
    """Contexto de ejecución para el análisis de texto."""
    document_id: str = ""
    tenant_id: str = ""
    filename: str = ""
    weaviate_url: str | None = None


@dataclass(frozen=True)
class TextCognitiveResult:
    """Resultado del motor cognitivo de texto."""
    analysis: dict[str, Any] = field(default_factory=dict)
    explanation: Explanation = field(default_factory=lambda: Explanation.minimal(""))
    confidence: float = 0.0


class TextCognitiveEngine:
    """Engine cognitivo para análisis de texto.

    Corre después de los sub-analyzers (sentiment, urgency, readability,
    structure, patterns) y produce un analysis dict, explanation, y confidence.
    """

    def __init__(self) -> None:
        self._enabled: bool | None = None  # lazy check

    def analyze(
        self,
        inp: TextAnalysisInput,
        ctx: TextAnalysisContext,
    ) -> TextCognitiveResult:
        """Ejecutar análisis cognitivo sobre texto pre-analizado.

        Args:
            inp: Scores pre-computados de los sub-analyzers.
            ctx: Contexto de ejecución (document_id, tenant, etc.).

        Returns:
            TextCognitiveResult con analysis dict, Explanation, y confidence.
        """
        if not self._is_enabled():
            logger.info("text_cognitive_engine_disabled")
            return TextCognitiveResult(
                analysis={
                    "cognitive": {
                        "status": "disabled",
                        "reason": f"{_FLAG_KEY}=false",
                    },
                    "word_count": inp.word_count,
                    "paragraph_count": inp.paragraph_count,
                    "sentiment_score": inp.sentiment_score,
                    "sentiment_label": inp.sentiment_label,
                    "urgency_score": inp.urgency_score,
                    "urgency_severity": inp.urgency_severity,
                },
                explanation=Explanation.minimal(ctx.document_id),
                confidence=0.0,
            )

        perceptions = self._build_perceptions(inp)
        fused_value, fused_confidence = self._fuse(perceptions)
        analysis = self._build_analysis_dict(inp, perceptions, fused_value)
        explanation = self._build_explanation(ctx.document_id, perceptions, fused_value, fused_confidence)
        confidence = self._compute_confidence(perceptions)

        return TextCognitiveResult(
            analysis=analysis,
            explanation=explanation,
            confidence=round(confidence, 4),
        )

    # ── Internals ──────────────────────────────────────────────

    @staticmethod
    def _is_enabled() -> bool:
        """Check feature flag ML_ENABLE_TEXT_ANALYSIS via global singleton.

        Usa get_feature_flags() (loader singleton con soporte de env vars)
        en vez de crear una instancia FeatureFlags() directa, para que
        el flag responda a la variable de entorno ML_ENABLE_TEXT_ANALYSIS.
        """
        try:
            from iot_machine_learning.ml_service.config.feature_flags import get_feature_flags
            flags = get_feature_flags()
            return bool(getattr(flags, _FLAG_KEY, False))
        except Exception:
            return False

    @staticmethod
    def _normalize_sentiment(score: float) -> float:
        """Map [-1, 1] sentiment → [0, 1] magnitude."""
        return min(1.0, max(0.0, abs(score)))

    @staticmethod
    def _build_perceptions(inp: TextAnalysisInput) -> list[dict[str, Any]]:
        """Build engine perception dicts from pre-computed scores."""
        return [
            {
                "engine_name": "text_urgency",
                "value": inp.urgency_score,
                "metadata": {
                    "severity": inp.urgency_severity,
                    "total_hits": inp.urgency_total_hits,
                    "hits": list(inp.urgency_hits),
                },
            },
            {
                "engine_name": "text_sentiment",
                "value": TextCognitiveEngine._normalize_sentiment(inp.sentiment_score),
                "metadata": {
                    "label": inp.sentiment_label,
                    "score": inp.sentiment_score,
                    "positive_count": inp.sentiment_positive_count,
                    "negative_count": inp.sentiment_negative_count,
                },
            },
            {
                "engine_name": "text_structural",
                "value": inp.structural_stability if inp.structural_available else 0.0,
                "metadata": {
                    "regime": inp.structural_regime,
                    "trend": inp.structural_trend,
                    "stability": inp.structural_stability,
                    "noise": inp.structural_noise,
                    "available": inp.structural_available,
                },
            },
            {
                "engine_name": "text_readability",
                "value": min(1.0, inp.readability_avg_sentence_length / 50.0) if inp.readability_n_sentences else 0.0,
                "metadata": {
                    "avg_sentence_length": inp.readability_avg_sentence_length,
                    "n_sentences": inp.readability_n_sentences,
                    "vocabulary_richness": inp.readability_vocabulary_richness,
                    "embedded_numeric_count": inp.readability_embedded_numeric_count,
                },
            },
            {
                "engine_name": "text_pattern",
                "value": min(1.0, inp.pattern_n_patterns / 10.0) if inp.pattern_available else 0.0,
                "metadata": {
                    "n_patterns": inp.pattern_n_patterns,
                    "change_points": list(inp.pattern_change_points),
                    "spikes": list(inp.pattern_spikes),
                    "available": inp.pattern_available,
                    "summary": inp.pattern_summary,
                },
            },
        ]

    def _fuse(self, perceptions: list[dict[str, Any]]) -> tuple[float, float]:
        """Weighted average fusion of engine perceptions.

        Returns:
            (fused_value, fused_confidence)
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for p in perceptions:
            name = p["engine_name"]
            weight = _WEIGHTS.get(name, 0.0)
            if weight <= 0:
                continue
            weighted_sum += p["value"] * weight
            total_weight += weight

        if total_weight <= 0:
            return (0.0, 0.0)

        fused_value = weighted_sum / total_weight
        fused_confidence = total_weight / sum(_WEIGHTS.values())
        return (min(1.0, fused_value), min(1.0, fused_confidence))

    def _compute_confidence(self, perceptions: list[dict[str, Any]]) -> float:
        """Confianza basada en agreement entre engines (normalized entropy)."""
        values = [p["value"] for p in perceptions]
        total = sum(values) + 1e-12
        probs = [v / total for v in values]
        entropy = -sum(p * math.log(p) for p in probs if p > 1e-12)
        n = len(probs)
        h_max = math.log(n) if n > 1 else 1.0
        normalized_entropy = entropy / h_max if h_max > 0 else 0.0
        consensus = 1.0 - normalized_entropy
        return 0.30 + consensus * 0.65

    def _build_analysis_dict(
        self,
        inp: TextAnalysisInput,
        perceptions: list[dict[str, Any]],
        fused_value: float,
    ) -> dict[str, Any]:
        """Build backward-compatible analysis dict with cognitive section."""
        engine_weights = {
            p["engine_name"]: _WEIGHTS.get(p["engine_name"], 0.0)
            for p in perceptions
        }
        return {
            "word_count": inp.word_count,
            "paragraph_count": inp.paragraph_count,
            "sentiment_score": inp.sentiment_score,
            "sentiment_label": inp.sentiment_label,
            "sentiment_positive_count": inp.sentiment_positive_count,
            "sentiment_negative_count": inp.sentiment_negative_count,
            "urgency_score": inp.urgency_score,
            "urgency_severity": inp.urgency_severity,
            "urgency_total_hits": inp.urgency_total_hits,
            "urgency_hits": list(inp.urgency_hits),
            "readability_avg_sentence_length": inp.readability_avg_sentence_length,
            "readability_n_sentences": inp.readability_n_sentences,
            "readability_vocabulary_richness": inp.readability_vocabulary_richness,
            "readability_embedded_numeric_count": inp.readability_embedded_numeric_count,
            "readability_sentences": list(inp.readability_sentences),
            "structural_regime": inp.structural_regime,
            "structural_trend": inp.structural_trend,
            "structural_stability": inp.structural_stability,
            "structural_noise": inp.structural_noise,
            "structural_available": inp.structural_available,
            "pattern_available": inp.pattern_available,
            "cognitive": {
                "engine_weights": engine_weights,
                "engine_perceptions": perceptions,
                "fused_value": round(fused_value, 4),
                "signal_profile": {
                    "n_engines": len(perceptions),
                    "word_count": inp.word_count,
                    "n_sentences": inp.readability_n_sentences,
                },
            },
        }

    def _build_explanation(
        self,
        series_id: str,
        perceptions: list[dict[str, Any]],
        fused_value: float,
        fused_confidence: float,
    ) -> Explanation:
        """Build Explanation domain object via ExplanationBuilder."""
        builder = ExplanationBuilder(series_id=series_id)

        builder.set_fusion(
            fused_value=fused_value,
            fused_confidence=fused_confidence,
            fused_trend="stable",
            final_weights={p["engine_name"]: _WEIGHTS.get(p["engine_name"], 0.0) for p in perceptions},
            selected_engine="text_urgency",
            selection_reason="weighted_fusion",
            fusion_method="weighted_average",
        )

        builder.set_fallback(fused_value, "text_cognitive_engine" if fused_value > 0 else "no_signals")

        for p in perceptions:
            if p["engine_name"] == "text_urgency" and p["value"] > 0.6:
                builder.set_filter("urgency_threshold", {"severity": p["metadata"]["severity"]})
                break

        return builder.build()
