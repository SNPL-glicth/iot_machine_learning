"""Centralized keyword configuration for text analysis.

All keyword lists live here.  Logic modules import from this file
instead of defining their own constants.  To add/remove keywords,
edit ONLY this file.
"""

from __future__ import annotations

from typing import List

# ---------------------------------------------------------------------------
# Urgency keywords (Spanish)
# ---------------------------------------------------------------------------

URGENCY_KEYWORDS_ES: List[str] = [
    "error", "falla", "crítico", "alerta", "urgente", "caída",
    "pérdida", "crisis", "incidente", "interrupción", "degradación",
    "timeout", "excepción", "fatal", "pánico", "sobrecarga",
]

# ---------------------------------------------------------------------------
# Urgency keywords (English)
# ---------------------------------------------------------------------------

URGENCY_KEYWORDS_EN: List[str] = [
    "error", "failure", "critical", "alert", "urgent", "down",
    "loss", "crisis", "incident", "outage", "degradation",
    "timeout", "exception", "fatal", "panic", "overload",
]

# ---------------------------------------------------------------------------
# Sentiment — positive words (ES + EN)
# ---------------------------------------------------------------------------

POSITIVE_WORDS: List[str] = [
    "bueno", "excelente", "éxito", "mejora", "bien", "estable",
    "óptimo", "resuelto", "correcto", "recuperado",
    "good", "excellent", "success", "improved", "stable",
    "optimal", "resolved", "correct", "recovered",
]

# ---------------------------------------------------------------------------
# Sentiment — negative words (ES + EN)
# ---------------------------------------------------------------------------

NEGATIVE_WORDS: List[str] = [
    "malo", "error", "problema", "falla", "inestable", "degradado",
    "lento", "fallido", "rechazado", "perdido",
    "bad", "problem", "failure", "unstable", "degraded",
    "slow", "failed", "rejected", "lost",
]
