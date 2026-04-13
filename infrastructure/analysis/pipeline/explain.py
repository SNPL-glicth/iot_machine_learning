"""Fase 4: Explicación — construye narrativa legible."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from iot_machine_learning.domain.ports.analysis import (
    AnalysisContext,
    Decision,
    Explanation,
    Signal,
)

# Import format_conclusion for rich narrative output
try:
    from ...ml_service.api.services.analysis.conclusion_formatter import format_conclusion
    _HAS_FORMATTER = True
except ImportError:
    _HAS_FORMATTER = False

logger = logging.getLogger(__name__)


class ExplainPhase:
    """Fase 4: Construcción de explicación.
    
    Responsabilidad: construir explicación legible del análisis.
    Consolida los DOS sistemas de explicación actuales:
    - domain/entities/explainability/ (domain objects)
    - infrastructure/ml/cognitive/explanation/ (builders)
    
    Args:
        narrative_builder: Constructor de narrativa (inyectado, opcional)
    """
    
    def __init__(
        self,
        narrative_builder=None,
    ) -> None:
        """Inicializa fase con builder opcional."""
        self._narrative_builder = narrative_builder
    
    def execute(
        self,
        decision: Decision,
        signal: Signal,
        context: AnalysisContext,
        timing: Dict[str, float],
    ) -> Explanation:
        """Ejecuta construcción de explicación.
        
        Args:
            decision: Decisión del razonamiento
            signal: Señal percibida
            context: Contexto de análisis
            timing: Dict para registrar tiempos
        
        Returns:
            Explanation con narrativa y trazas
        """
        t0 = time.monotonic()
        
        # 1. Construir narrativa
        narrative = self._build_narrative(decision, signal, context)
        
        # 2. Construir contributions
        contributions = self._build_contributions(decision)
        
        # 3. Construir reasoning trace
        reasoning_trace = self._build_reasoning_trace(decision, timing)
        
        explanation = Explanation(
            narrative=narrative,
            confidence=decision.confidence,
            contributions=contributions,
            reasoning_trace=reasoning_trace,
            domain=signal.domain,
            severity=decision.severity,
            metadata={
                "series_id": context.series_id,
                "tenant_id": context.tenant_id,
                "input_type": signal.input_type.value,
            },
        )
        
        timing["explain"] = (time.monotonic() - t0) * 1000
        
        logger.debug(
            "explain_complete",
            extra={
                "narrative_length": len(narrative),
                "n_contributions": len(contributions),
                "ms": round(timing["explain"], 2),
            },
        )
        
        return explanation
    
    def _build_narrative(
        self,
        decision: Decision,
        signal: Signal,
        context: AnalysisContext,
    ) -> str:
        """Construye narrativa legible con formato rico."""
        logger.warning(f"[EXPLAIN_PHASE] _build_narrative called: domain={signal.domain}, has_formatter={_HAS_FORMATTER}")
        
        if self._narrative_builder is not None:
            try:
                return self._narrative_builder.build(decision, signal, context)
            except Exception as e:
                logger.warning(f"narrative_builder_failed: {e}")
        
        # Build rich narrative using format_conclusion
        # Create a mock result object with the data we have
        class MockResult:
            def __init__(self, signal, decision):
                self.signal = signal
                self.decision = decision
                self.domain = signal.domain
                self.confidence = decision.confidence
                # Build analysis dict from signal features
                self.analysis = getattr(signal, 'features', {})
                self.explanation = None
        
        mock_result = MockResult(signal, decision)
        
        if _HAS_FORMATTER:
            try:
                logger.warning(f"[EXPLAIN_PHASE] Calling format_conclusion with domain={mock_result.domain}, words={mock_result.analysis.get('word_count', 0)}")
                narrative = format_conclusion(mock_result)
                logger.warning(f"[EXPLAIN_PHASE] format_conclusion returned: {narrative[:100]}...")
                return narrative
            except Exception as e:
                logger.warning(f"[EXPLAIN_PHASE] format_conclusion_failed: {e}", exc_info=True)
        
        # Fallback: narrativa simple
        parts = [
            f"Análisis de {signal.input_type.value} en dominio {signal.domain}.",
            f"Severidad: {decision.severity}.",
            f"Confianza: {decision.confidence:.2%}.",
        ]
        
        if decision.perceptions:
            top_perception = max(decision.perceptions, key=lambda p: p.confidence)
            parts.append(
                f"Perspectiva principal: {top_perception.perspective} "
                f"(score: {top_perception.score:.2f})."
            )
        
        if decision.selected_engine:
            parts.append(f"Motor seleccionado: {decision.selected_engine}.")
        
        return " ".join(parts)
    
    def _build_contributions(
        self,
        decision: Decision,
    ) -> Dict[str, float]:
        """Construye dict de contribuciones por perspectiva."""
        contributions = {}
        
        for perception in decision.perceptions:
            weight = decision.weights.get(perception.perspective, 0.0)
            contributions[perception.perspective] = weight * perception.score
        
        return contributions
    
    def _build_reasoning_trace(
        self,
        decision: Decision,
        timing: Dict[str, float],
    ) -> List[str]:
        """Construye traza de razonamiento."""
        trace = []
        
        # Fase 1: Percepción
        if "perceive" in timing:
            trace.append(
                f"PERCEIVE: Detectado tipo y dominio ({timing['perceive']:.1f}ms)"
            )
        
        # Fase 2: Análisis
        if "analyze" in timing:
            trace.append(
                f"ANALYZE: Colectadas {len(decision.perceptions)} percepciones "
                f"({timing['analyze']:.1f}ms)"
            )
        
        # Fase 3: Razonamiento
        if "reason" in timing:
            n_inhibited = decision.metadata.get("n_inhibited", 0)
            trace.append(
                f"REASON: Inhibidas {n_inhibited}, fusionadas {len(decision.perceptions)} "
                f"({timing['reason']:.1f}ms)"
            )
        
        # Fase 4: Explicación
        if "explain" in timing:
            trace.append(
                f"EXPLAIN: Narrativa construida ({timing['explain']:.1f}ms)"
            )
        
        # Decisión final
        trace.append(
            f"DECISION: {decision.severity} con confianza {decision.confidence:.2%}"
        )
        
        return trace
