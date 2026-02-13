"""Renderer de explicaciones cognitivas.

Transforma ``Explanation`` (dominio) en formatos consumibles.
NO calcula lógica nueva.  NO recalcula métricas.  NO interpreta
diferente a como el dominio ya decidió.

El renderer no piensa.  Solo comunica.

Tres salidas:
- ``render_summary``          — resumen corto para dashboards / alertas.
- ``render_technical_report`` — reporte técnico con clasificaciones metacognitivas.
- ``render_structured_json``  — JSON extendido con clasificaciones.

Clasificaciones metacognitivas (solo lectura de propiedades del dominio):
- Nivel de certeza
- Nivel de desacuerdo interno
- Estabilidad cognitiva
- Riesgo de sobreajuste
- Nivel de conflicto entre engines

Dependencias: solo ``domain.entities.explainability``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from iot_machine_learning.domain.entities.explainability.explanation import Explanation


# ── Clasificaciones metacognitivas ──────────────────────────────
#
# Cada función lee UNA propiedad ya computada del dominio y la
# mapea a un nivel discreto.  Sin cálculos nuevos.


def _classify_certainty(confidence: float) -> str:
    """Clasifica nivel de certeza a partir de confidence ya computada."""
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.6:
        return "moderate"
    if confidence >= 0.35:
        return "low"
    return "very_low"


def _classify_disagreement(consensus_spread: float, n_engines: int) -> str:
    """Clasifica desacuerdo interno entre engines.

    Lee ``contributions.consensus_spread`` (max - min de predicciones).
    """
    if n_engines < 2:
        return "none"
    if consensus_spread < 0.5:
        return "consensus"
    if consensus_spread < 2.0:
        return "mild"
    if consensus_spread < 5.0:
        return "significant"
    return "severe"


def _classify_cognitive_stability(
    n_inhibited: int,
    n_engines: int,
    has_adaptation: bool,
) -> str:
    """Clasifica estabilidad cognitiva del sistema.

    Lee ``contributions.n_inhibited``, ``trace.has_adaptation``.
    """
    if n_engines == 0:
        return "degraded"
    inhibition_ratio = n_inhibited / n_engines
    if inhibition_ratio == 0.0 and not has_adaptation:
        return "stable"
    if inhibition_ratio <= 0.3:
        return "adapting"
    if inhibition_ratio <= 0.6:
        return "stressed"
    return "degraded"


def _classify_overfit_risk(
    dominant_weight_ratio: float,
    n_engines: int,
) -> str:
    """Clasifica riesgo de sobreajuste a un solo engine.

    Lee ``contributions.dominant_weight_ratio``.
    """
    if n_engines < 2:
        return "not_applicable"
    if dominant_weight_ratio > 0.9:
        return "high"
    if dominant_weight_ratio > 0.7:
        return "moderate"
    return "low"


def _classify_engine_conflict(
    contributions: list,
) -> str:
    """Clasifica conflicto entre engines por tendencias opuestas.

    Lee ``EngineContribution.trend`` de cada engine activo.
    """
    active = [c for c in contributions if c.final_weight > 0.01]
    if len(active) < 2:
        return "none"
    trends = {c.trend for c in active}
    if "up" in trends and "down" in trends:
        return "directional_conflict"
    if len(trends) > 1:
        return "mild_divergence"
    return "aligned"


# ── Renderer ────────────────────────────────────────────────────


class ExplanationRenderer:
    """Transforma Explanation en formatos consumibles.

    Reglas:
    - No importa nada de infrastructure.
    - No modifica Explanation.
    - No calcula lógica nueva — solo lee propiedades del dominio.
    - Solo transforma y clasifica.
    """

    def render_summary(self, explanation: Explanation) -> str:
        """Resumen corto (1-3 líneas) para dashboards / alertas.

        Args:
            explanation: Explanation del dominio.

        Returns:
            String con resumen legible.
        """
        parts: List[str] = []
        o = explanation.outcome
        c = explanation.contributions
        s = explanation.signal

        # Línea 1: resultado
        if o.predicted_value is not None:
            parts.append(
                f"Predicción: {o.predicted_value:.4f} "
                f"(confianza: {_classify_certainty(o.confidence)}, "
                f"tendencia: {o.trend})"
            )

        # Línea 2: motor y consenso
        if c.n_engines > 0:
            disagreement = _classify_disagreement(
                c.consensus_spread, c.n_engines
            )
            parts.append(
                f"Motor: {c.selected_engine} "
                f"({c.n_engines} engines, "
                f"desacuerdo: {disagreement})"
            )
        elif c.fallback_used:
            parts.append(f"Fallback: {c.fallback_reason}")

        # Línea 3: régimen
        if s.regime != "unknown":
            parts.append(f"Régimen: {s.regime} (ruido: {s.noise_ratio:.3f})")

        return "\n".join(parts)

    def render_technical_report(self, explanation: Explanation) -> str:
        """Reporte técnico con clasificaciones metacognitivas.

        Args:
            explanation: Explanation del dominio.

        Returns:
            String multi-línea con secciones.
        """
        sections: List[str] = []
        o = explanation.outcome
        c = explanation.contributions
        s = explanation.signal
        t = explanation.trace
        f = explanation.filter

        # ── Encabezado
        sections.append(
            f"=== Reporte Cognitivo: {explanation.series_id} ==="
        )

        # ── Señal
        sections.append(
            f"\n[Señal] n={s.n_points}, μ={s.mean:.4f}, σ={s.std:.4f}, "
            f"ruido={s.noise_ratio:.4f}, régimen={s.regime}"
        )
        if s.slope != 0.0:
            sections.append(
                f"  pendiente={s.slope:.6f}, curvatura={s.curvature:.6f}"
            )

        # ── Filtrado
        if explanation.has_filter_data:
            sections.append(
                f"\n[Filtrado] {f.filter_name}, "
                f"reducción_ruido={f.noise_reduction_ratio:.4f}, "
                f"efectivo={f.is_effective}"
            )

        # ── Engines
        if c.n_engines > 0:
            sections.append(f"\n[Engines] {c.n_engines} participaron:")
            for ec in c.contributions:
                marker = " [INHIBIDO]" if ec.inhibited else ""
                sections.append(
                    f"  {ec.engine_name}: pred={ec.predicted_value:.4f}, "
                    f"peso={ec.final_weight:.4f}, "
                    f"confianza={ec.confidence:.3f}{marker}"
                )

        # ── Clasificaciones metacognitivas
        meta = self._compute_metacognitive_labels(explanation)
        sections.append("\n[Metacognición]")
        sections.append(f"  certeza:       {meta['certainty']}")
        sections.append(f"  desacuerdo:    {meta['disagreement']}")
        sections.append(f"  estabilidad:   {meta['cognitive_stability']}")
        sections.append(f"  sobreajuste:   {meta['overfit_risk']}")
        sections.append(f"  conflicto:     {meta['engine_conflict']}")

        # ── Resultado
        sections.append(f"\n[Resultado] {o.kind}")
        if o.predicted_value is not None:
            sections.append(f"  valor={o.predicted_value:.6f}")
        sections.append(
            f"  confianza={o.confidence:.4f}, tendencia={o.trend}"
        )
        if o.is_anomaly:
            sections.append(f"  anomalía=sí, score={o.anomaly_score}")

        # ── Traza
        if explanation.has_trace:
            sections.append(
                f"\n[Traza] fases={t.phase_kinds}"
            )

        return "\n".join(sections)

    def render_structured_json(self, explanation: Explanation) -> Dict[str, Any]:
        """JSON extendido con clasificaciones metacognitivas.

        Incluye todo lo de ``Explanation.to_dict()`` más un bloque
        ``metacognitive`` con las clasificaciones.

        Args:
            explanation: Explanation del dominio.

        Returns:
            Dict serializable a JSON.
        """
        base = explanation.to_dict()
        base["metacognitive"] = self._compute_metacognitive_labels(explanation)
        return base

    # ── Interno ─────────────────────────────────────────────────

    def _compute_metacognitive_labels(
        self,
        explanation: Explanation,
    ) -> Dict[str, str]:
        """Lee propiedades del dominio y las clasifica.

        No calcula nada nuevo.  Solo mapea valores ya existentes
        a niveles discretos.
        """
        c = explanation.contributions
        t = explanation.trace

        return {
            "certainty": _classify_certainty(explanation.outcome.confidence),
            "disagreement": _classify_disagreement(
                c.consensus_spread, c.n_engines,
            ),
            "cognitive_stability": _classify_cognitive_stability(
                c.n_inhibited, c.n_engines, t.has_adaptation,
            ),
            "overfit_risk": _classify_overfit_risk(
                c.dominant_weight_ratio, c.n_engines,
            ),
            "engine_conflict": _classify_engine_conflict(
                c.contributions,
            ),
        }
