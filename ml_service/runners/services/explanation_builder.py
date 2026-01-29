"""Explanation builder service for ML online processing.

Extraído de ml_stream_runner.py para modularidad.
Responsabilidad: Construir explicaciones y determinar severidad.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..models.online_analysis import OnlineAnalysis
from ..models.sensor_state import SensorState

logger = logging.getLogger(__name__)


class ExplanationBuilder:
    """Construye explicaciones y determina severidad de eventos ML.
    
    Responsabilidades:
    - Construir explicación en lenguaje humano
    - Determinar severidad basada en patrón
    - Determinar acción recomendada
    - Decidir si emitir evento
    """
    
    def build_explanation(
        self,
        *,
        sensor_type: str,
        analysis: OnlineAnalysis,
        value_within_warning_range: bool = False,
    ) -> tuple:
        """Devuelve (severity_label, recommended_action, explanation).
        
        FIX CRÍTICO: Si el valor está dentro del rango WARNING del usuario,
        el ML NO puede generar severidad WARNING/CRITICAL. El usuario definió
        ese rango como "normal" y el ML debe respetarlo.
        """
        pattern = analysis.behavior_pattern

        if value_within_warning_range:
            severity = "NORMAL"
            action = "none"
            human = (
                "Valor dentro del rango definido por el usuario. "
                f"Patrón detectado: {pattern} (informativo, sin acción requerida)."
            )
        elif pattern == "STABLE":
            severity = "NORMAL"
            action = "none"
            human = "Comportamiento estable del sensor, dentro de su patrón histórico."
        elif pattern in {"OSCILLATING", "DRIFTING"}:
            severity = "WARN"
            action = "check_soon"
            human = (
                "El sensor muestra oscilaciones o una deriva sostenida; "
                "revisar calibración y condiciones ambientales a la brevedad."
            )
        else:
            severity = "CRITICAL"
            action = "immediate_intervention"
            human = (
                "Comportamiento fuertemente anómalo (spikes o crecimiento/caída exponencial); "
                "se recomienda intervención inmediata sobre el dispositivo."
            )

        trend_desc: str
        if analysis.slope_long > 0.1:
            trend_desc = "en aumento"
        elif analysis.slope_long < -0.1:
            trend_desc = "en descenso"
        else:
            trend_desc = "estable"

        parts: list = []
        parts.append(
            f"Sensor tipo {sensor_type or 'desconocido'} con severidad actual {severity} "
            f"y patrón {pattern}. Acción recomendada: {action}. "
        )
        parts.append(
            "Ventanas 1s/5s/10s: "
            f"media_base={analysis.baseline_mean:.4f}, "
            f"último_valor={analysis.last_value:.4f}, "
            f"z_score={analysis.z_score_last:.2f}, "
            f"tendencia_larga={trend_desc}. "
        )

        explanation = human + " " + "".join(parts)
        return severity, action, explanation

    def should_emit_state_event(
        self,
        *,
        prev_state: Optional[SensorState],
        new_state: SensorState,
        analysis: OnlineAnalysis,
        value_within_warning_range: bool = False,
    ) -> tuple:
        """Decide si se debe insertar un evento de estado y qué event_code usar.
        
        REGLA DE DOMINIO:
        - Si valor dentro del rango del usuario → NO emitir eventos
        - Solo emitir para cambios SIGNIFICATIVOS de severidad
        - NO generar eventos masivos por oscilaciones normales
        - Un evento por transición, no por lectura
        """
        # REGLA 1: Si valor dentro del rango del usuario → NO emitir eventos
        if value_within_warning_range:
            return False, prev_state.last_event_code if prev_state else "ML_BEHAVIOR_STATE"

        # REGLA 2: Primer evento para el sensor → NO emitir (esperar baseline)
        if prev_state is None:
            return False, "ML_BEHAVIOR_STATE"

        # REGLA 3: Solo emitir si hay cambio SIGNIFICATIVO de severidad
        severity_changed = prev_state.severity != new_state.severity
        
        if not severity_changed:
            return False, prev_state.last_event_code or "ML_BEHAVIOR_STATE"

        # REGLA 4: Determinar código de evento según el tipo de anomalía
        if analysis.new_transient_anomaly:
            code = "TRANSIENT_ANOMALY"
        elif analysis.is_curve_anomalous:
            code = "CURVE_ANOMALY"
        elif analysis.has_microvariation:
            code = "MICRO_VARIATION"
        else:
            code = "ML_BEHAVIOR_STATE"

        # REGLA 5: No repetir el mismo código de evento
        if prev_state.last_event_code == code:
            return False, code

        return True, code

    def map_severity_to_event_type(self, severity: str) -> str:
        """Mapea severidad lógica (NORMAL/WARN/CRITICAL) a event_type."""
        sev = severity.upper()
        if sev == "CRITICAL":
            return "critical"
        if sev == "WARN":
            return "warning"
        return "notice"

    def build_title(self, event_code: str, severity: str, pattern: str) -> str:
        """Construye título para el evento ML."""
        return f"ML online: {event_code} (estado {severity}, patrón {pattern})"
