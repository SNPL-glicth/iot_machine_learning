"""Phrase bank for embedding-based narrative generation.

No f-strings.  No if/elif at runtime.  Phrases are static text; selection
happens via cosine similarity against the 8-dim situation embedding.

Dimensions (8-dim target vector, all in [0, 1]):
    0: criticality / urgency
    1: warning / concern level
    2: stability / calm
    3: positive trend strength
    4: negative trend strength
    5: anomaly presence
    6: high model confidence
    7: uncertainty / low confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class PhraseEntry:
    text: str
    target: List[float]  # 8-dim, in [0, 1]
    domain: Optional[str] = None  # None = domain-neutral (always eligible)


_PHRASES: List[PhraseEntry] = [
    # ── Critical / urgent (domain-neutral) ──
    PhraseEntry(
        "Condición crítica detectada. Requiere atención inmediata.",
        [0.95, 0.05, 0.05, 0.05, 0.05, 0.90, 0.80, 0.10],
    ),
    PhraseEntry(
        "Situación de emergencia identificada. Alertar al equipo responsable.",
        [0.90, 0.10, 0.05, 0.05, 0.05, 0.85, 0.75, 0.15],
    ),
    PhraseEntry(
        "Umbral crítico superado. Intervención manual recomendada sin demora.",
        [0.92, 0.08, 0.05, 0.05, 0.70, 0.80, 0.85, 0.10],
    ),
    PhraseEntry(
        "Anomalía severa confirmada con alta confianza. Escalar a nivel operativo.",
        [0.88, 0.12, 0.10, 0.10, 0.10, 0.90, 0.90, 0.05],
    ),

    # ── Warning / concern (domain-neutral) ──
    PhraseEntry(
        "Patrón inusual detectado. Supervisar de cerca en próximas horas.",
        [0.15, 0.85, 0.20, 0.10, 0.40, 0.60, 0.70, 0.20],
    ),
    PhraseEntry(
        "Tendencia preocupante observada. Revisar indicadores secundarios.",
        [0.20, 0.80, 0.15, 0.30, 0.50, 0.40, 0.65, 0.25],
    ),
    PhraseEntry(
        "Comportamiento atípico identificado. Evaluación recomendada.",
        [0.25, 0.75, 0.20, 0.15, 0.35, 0.55, 0.60, 0.30],
    ),
    PhraseEntry(
        "Indicadores de riesgo moderado. Monitoreo intensificado sugerido.",
        [0.30, 0.70, 0.25, 0.20, 0.30, 0.45, 0.55, 0.35],
    ),
    PhraseEntry(
        "Desviación significativa respecto a línea base. Verificar condiciones operativas.",
        [0.35, 0.65, 0.20, 0.25, 0.40, 0.50, 0.60, 0.30],
    ),

    # ── Stable / info (domain-neutral) ──
    PhraseEntry(
        "Operación dentro de parámetros normales. Continuar monitoreo estándar.",
        [0.05, 0.10, 0.90, 0.10, 0.10, 0.05, 0.75, 0.15],
    ),
    PhraseEntry(
        "Comportamiento estable y predecible. Sin acción requerida.",
        [0.05, 0.08, 0.92, 0.05, 0.05, 0.03, 0.80, 0.12],
    ),
    PhraseEntry(
        "Valores dentro de rangos esperados. Estado saludable confirmado.",
        [0.03, 0.05, 0.95, 0.05, 0.05, 0.02, 0.85, 0.10],
    ),
    PhraseEntry(
        "Tendencia positiva sostenida. Evolución favorable detectada.",
        [0.05, 0.05, 0.70, 0.90, 0.05, 0.05, 0.65, 0.20],
    ),
    PhraseEntry(
        "Recuperación gradual observada. Parámetros retornando a línea base.",
        [0.10, 0.15, 0.60, 0.80, 0.10, 0.10, 0.60, 0.25],
    ),

    # ── Declining / negative trend (domain-neutral) ──
    PhraseEntry(
        "Tendencia descendente identificada. Evaluar causas subyacentes.",
        [0.20, 0.40, 0.30, 0.05, 0.85, 0.30, 0.55, 0.30],
    ),
    PhraseEntry(
        "Deterioro progresivo detectado. Revisar correlaciones con variables externas.",
        [0.40, 0.50, 0.15, 0.05, 0.80, 0.50, 0.60, 0.30],
    ),
    PhraseEntry(
        "Degradación sostenida en métricas clave. Análisis profundo recomendado.",
        [0.50, 0.55, 0.10, 0.05, 0.85, 0.55, 0.65, 0.25],
    ),

    # ── Noisy / uncertain (domain-neutral) ──
    PhraseEntry(
        "Señal altamente ruidosa. Interpretar resultados con cautela.",
        [0.30, 0.30, 0.20, 0.20, 0.20, 0.40, 0.20, 0.85],
    ),
    PhraseEntry(
        "Baja confianza en predicción debido a varianza elevada. Ampliar ventana de datos.",
        [0.25, 0.25, 0.20, 0.15, 0.15, 0.30, 0.15, 0.90],
    ),
    PhraseEntry(
        "Datos contradictorios entre motores. Requiere verificación manual.",
        [0.35, 0.45, 0.15, 0.25, 0.25, 0.40, 0.25, 0.80],
    ),
    PhraseEntry(
        "Modelo con incertidumbre alta. Resultado preliminar; no tomar decisiones drásticas.",
        [0.20, 0.30, 0.25, 0.15, 0.15, 0.25, 0.10, 0.95],
    ),

    # ── Anomaly-specific (domain-neutral) ──
    PhraseEntry(
        "Anomalía estadística confirmada. Patrón no observado históricamente.",
        [0.60, 0.30, 0.10, 0.10, 0.10, 0.95, 0.70, 0.20],
    ),
    PhraseEntry(
        "Cambio abrupto detectado. Correlación con eventos externos sugerida.",
        [0.55, 0.35, 0.15, 0.50, 0.40, 0.80, 0.65, 0.25],
    ),
    PhraseEntry(
        "Outlier multivariado identificado. Revisar sensores relacionados.",
        [0.50, 0.40, 0.15, 0.20, 0.20, 0.85, 0.60, 0.25],
    ),

    # ── Fusion / multi-engine (domain-neutral) ──
    PhraseEntry(
        "Fusión multi-motor indica consenso fuerte. Predicción robusta.",
        [0.15, 0.15, 0.70, 0.30, 0.30, 0.10, 0.90, 0.08],
    ),
    PhraseEntry(
        "Discrepancia entre motores detectada. Peso adaptativo aplicado.",
        [0.30, 0.40, 0.20, 0.25, 0.25, 0.35, 0.50, 0.55],
    ),
    PhraseEntry(
        "Dominancia de motor especializado. Resultado influenciado por experto principal.",
        [0.20, 0.20, 0.50, 0.20, 0.20, 0.20, 0.75, 0.20],
    ),

    # ── Cybersecurity / seguridad informática (domain=security) ──
    PhraseEntry(
        "Intentos de autenticación fallidos detectados. Revisar política de acceso.",
        [0.30, 0.80, 0.10, 0.05, 0.20, 0.70, 0.75, 0.25],
        domain="security",
    ),
    PhraseEntry(
        "Escaneo de puertos detectado. Posible reconocimiento de red.",
        [0.35, 0.75, 0.10, 0.05, 0.15, 0.75, 0.70, 0.30],
        domain="security",
    ),
    PhraseEntry(
        "Comportamiento anómalo de proceso identificado. Verificar integridad del sistema.",
        [0.40, 0.65, 0.15, 0.05, 0.20, 0.80, 0.65, 0.35],
        domain="security",
    ),
    PhraseEntry(
        "Acceso fuera de horario normal. Revisar logs de sesión.",
        [0.30, 0.70, 0.20, 0.05, 0.15, 0.60, 0.70, 0.30],
        domain="security",
    ),
    PhraseEntry(
        "Escalación de privilegios detectada. Intervención inmediata requerida.",
        [0.90, 0.10, 0.05, 0.05, 0.10, 0.90, 0.85, 0.15],
        domain="security",
    ),
    PhraseEntry(
        "Tráfico de red inusual. Analizar patrones de comunicación.",
        [0.35, 0.70, 0.15, 0.05, 0.20, 0.65, 0.70, 0.30],
        domain="security",
    ),
    PhraseEntry(
        "Cambio de régimen crítico en logs. Posible indicador de compromiso.",
        [0.85, 0.15, 0.05, 0.05, 0.30, 0.85, 0.80, 0.20],
        domain="security",
    ),
    PhraseEntry(
        "Ataque de fuerza bruta detectado. Bloquear origen y auditar cuentas.",
        [0.85, 0.15, 0.05, 0.05, 0.10, 0.90, 0.80, 0.20],
        domain="security",
    ),
    PhraseEntry(
        "Inyección SQL identificada en logs. Revisar aplicaciones expuestas.",
        [0.90, 0.10, 0.05, 0.05, 0.10, 0.95, 0.85, 0.15],
        domain="security",
    ),
    PhraseEntry(
        "Movimiento lateral detectado en red. Aislar segmento afectado.",
        [0.88, 0.12, 0.05, 0.05, 0.15, 0.92, 0.80, 0.20],
        domain="security",
    ),
    PhraseEntry(
        "Exfiltración de datos sospechada. Activar respuesta a incidentes.",
        [0.92, 0.08, 0.05, 0.05, 0.10, 0.90, 0.82, 0.18],
        domain="security",
    ),
    PhraseEntry(
        "Malware identificado en endpoint. Ejecutar contención inmediata.",
        [0.87, 0.13, 0.05, 0.05, 0.10, 0.88, 0.85, 0.15],
        domain="security",
    ),
    PhraseEntry(
        "Denegación de servicio (DoS) detectada. Mitigar tráfico malicioso.",
        [0.82, 0.18, 0.05, 0.05, 0.20, 0.85, 0.78, 0.22],
        domain="security",
    ),
    PhraseEntry(
        "Acceso a datos sensibles sin autorización. Escalar a equipo de seguridad.",
        [0.88, 0.12, 0.05, 0.05, 0.10, 0.90, 0.82, 0.18],
        domain="security",
    ),
    PhraseEntry(
        "Credenciales comprometidas detectadas. Forzar rotación inmediata.",
        [0.86, 0.14, 0.05, 0.05, 0.10, 0.88, 0.80, 0.20],
        domain="security",
    ),
    PhraseEntry(
        "Política de seguridad violada. Revisar configuraciones y permisos.",
        [0.40, 0.60, 0.15, 0.05, 0.20, 0.55, 0.70, 0.30],
        domain="security",
    ),
    PhraseEntry(
        "Actividad de ransomware detectada. Aislar sistema y preservar evidencia.",
        [0.95, 0.05, 0.05, 0.05, 0.10, 0.95, 0.90, 0.10],
        domain="security",
    ),
    PhraseEntry(
        "Acción recomendada: aislar sistema afectado de la red.",
        [0.80, 0.20, 0.10, 0.05, 0.05, 0.70, 0.75, 0.25],
        domain="security",
    ),
    PhraseEntry(
        "Acción recomendada: alertar equipo SOC para respuesta coordinada.",
        [0.50, 0.50, 0.15, 0.05, 0.10, 0.60, 0.70, 0.30],
        domain="security",
    ),
    PhraseEntry(
        "Acción recomendada: revisar logs completos para análisis forense.",
        [0.20, 0.40, 0.30, 0.05, 0.10, 0.40, 0.65, 0.35],
        domain="security",
    ),

    # ── General / fallback (domain-neutral) ──
    PhraseEntry(
        "Evaluación cognitiva completada. Resumen de hallazgos disponible.",
        [0.10, 0.10, 0.60, 0.10, 0.10, 0.10, 0.70, 0.20],
    ),
    PhraseEntry(
        "Pipeline de inferencia ejecutado. Métricas de calidad dentro de umbrales aceptables.",
        [0.05, 0.05, 0.75, 0.05, 0.05, 0.05, 0.80, 0.10],
    ),
    PhraseEntry(
        "Análisis contextual integrado. Contexto histórico incorporado en predicción.",
        [0.08, 0.08, 0.65, 0.15, 0.15, 0.08, 0.75, 0.15],
    ),
]


def get_phrase_bank() -> List[PhraseEntry]:
    """Return the full phrase bank (seed set, ~30 phrases).

    Designed to grow to ~200 phrases via offline training on historical logs.
    Each new phrase must have an 8-dim target vector assigned by an oracle
    (human annotator or autoencoder projection).
    """
    return list(_PHRASES)
