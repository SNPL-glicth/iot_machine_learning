"""ImpactSignalDetector — scans text for hard impact signals.

Goes beyond keyword-counting urgency to detect **real-world impact**:

    1. **Critical markers** — explicit severity tags (CRÍTICO, CRITICAL, FATAL).
    2. **SLA/KPI breaches** — contract/metric violations (SLA 78% vs 99.5%).
    3. **Extreme metrics** — dangerous numeric values (temperatura 89°C, CPU 95%).
    4. **Temporal risk** — future failure language ("riesgo de caída en 72h").
    5. **Cascade/total failure** — systemic collapse language.

Returns ``ImpactSignalResult`` with a composite score [0, 1] and
individual signal hits for explainability.

No imports from ml_service — pure infrastructure.
Single entry point: ``detect_impact_signals()``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Signal categories with weights ──

# Category 1: Explicit critical markers (ES + EN)
_CRITICAL_MARKERS: List[str] = [
    "crítico", "critical", "fatal", "emergencia", "emergency",
    "catastrófico", "catastrophic", "colapso", "collapse",
    "caída total", "total failure", "total outage",
    "interrupción total", "complete outage", "system down",
    "fuera de servicio", "out of service",
]

# Category 2: SLA/KPI breach language
_SLA_BREACH_PATTERNS: List[str] = [
    r"sla\b", r"kpi\b", r"breach", r"brecha",
    r"incumplimiento", r"violation", r"violación",
    r"por debajo del?\s*\d", r"below\s+\d",
    r"no cumple", r"does not meet", r"failed to meet",
    r"fuera de rango", r"out of range",
    r"\d+\.?\d*\s*%\s*vs\s*\d+\.?\d*\s*%",  # "78% vs 99.5%"
]

# Category 3: Extreme metric patterns (captures the value)
_EXTREME_METRIC_PATTERNS: List[re.Pattern] = [
    # Temperature: >= 80°C or >= 176°F
    # Allows intervening words: "Temperatura del rack: 89°C"
    re.compile(
        r"(?:temperatura|temperature|temp)[^0-9]{0,40}?(\d+\.?\d*)\s*°?\s*[cC]",
        re.IGNORECASE,
    ),
    # Standalone temperature values: "94°C", "94 °C", "94C" (no prefix needed)
    re.compile(
        r"(\d{2,3})\s*°?\s*[cC]\b",
        re.IGNORECASE,
    ),
    # CPU/Memory/Disk: >= 90%
    # Allows intervening words: "CPU al 95%", "uso de memoria: 92%"
    re.compile(
        r"(?:cpu|memoria|memory|disco|disk|uso|usage|load)[^0-9]{0,30}?(\d+\.?\d*)\s*%",
        re.IGNORECASE,
    ),
    # Availability/uptime below threshold (captures low percentages)
    re.compile(
        r"(?:disponibilidad|availability|uptime)[^0-9]{0,30}?(\d+\.?\d*)\s*%",
        re.IGNORECASE,
    ),
]

# Category 4: Temporal risk / future failure language
_TEMPORAL_RISK_PATTERNS: List[str] = [
    r"riesgo de (?:caída|falla|interrupción|colapso)",
    r"risk of (?:failure|outage|collapse|downtime|crash)",
    r"en\s+\d+\s*(?:horas|hours|días|days|minutos|minutes)\b",
    r"within\s+\d+\s*(?:hours|days|minutes)\b",
    r"antes de\s+\d", r"before\s+\d",
    r"inminente", r"imminent",
    r"podría fallar", r"could fail", r"will fail",
    r"punto de quiebre", r"breaking point",
    r"capacidad máxima", r"at capacity", r"max capacity",
]

# Category 5: Cascade / systemic failure
_CASCADE_PATTERNS: List[str] = [
    r"cascada", r"cascade", r"efecto dominó", r"domino effect",
    r"múltiples (?:sistemas|servicios|servidores|nodos)",
    r"multiple (?:systems|services|servers|nodes)",
    r"propagación", r"propagation", r"spread(?:ing)?",
    r"todos los", r"all (?:systems|services|servers)",
    r"infraestructura completa", r"entire infrastructure",
]

# Thresholds for extreme metrics
_TEMP_CRITICAL_C = 80.0
_RESOURCE_CRITICAL_PCT = 90.0
_AVAILABILITY_CRITICAL_PCT = 95.0  # below this = breach


@dataclass(frozen=True)
class ImpactSignal:
    """A single detected impact signal."""

    category: str
    pattern: str
    matched_text: str
    value: Optional[float] = None


@dataclass(frozen=True)
class ImpactSignalResult:
    """Result of impact signal detection.

    Attributes:
        score: Composite impact score [0, 1].
        signals: Individual detected signals.
        has_critical_markers: At least one explicit critical tag.
        has_sla_breach: SLA/KPI breach language detected.
        has_extreme_metrics: Dangerous numeric values found.
        has_temporal_risk: Future failure risk language.
        has_cascade_risk: Systemic/cascade failure language.
        summary: Human-readable summary of detected impact.
    """

    score: float
    signals: List[ImpactSignal] = field(default_factory=list)
    has_critical_markers: bool = False
    has_sla_breach: bool = False
    has_extreme_metrics: bool = False
    has_temporal_risk: bool = False
    has_cascade_risk: bool = False
    summary: str = ""

    @property
    def n_signals(self) -> int:
        return len(self.signals)

    @property
    def n_categories_hit(self) -> int:
        return sum([
            self.has_critical_markers,
            self.has_sla_breach,
            self.has_extreme_metrics,
            self.has_temporal_risk,
            self.has_cascade_risk,
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "n_signals": self.n_signals,
            "n_categories_hit": self.n_categories_hit,
            "has_critical_markers": self.has_critical_markers,
            "has_sla_breach": self.has_sla_breach,
            "has_extreme_metrics": self.has_extreme_metrics,
            "has_temporal_risk": self.has_temporal_risk,
            "has_cascade_risk": self.has_cascade_risk,
            "signals": [
                {"category": s.category, "matched_text": s.matched_text}
                for s in self.signals
            ],
            "summary": self.summary,
        }


def detect_impact_signals(text: str) -> ImpactSignalResult:
    """Scan text for hard impact signals that indicate real-world severity.

    Args:
        text: Full document text.

    Returns:
        ``ImpactSignalResult`` with composite score and individual signals.
    """
    text_lower = text.lower()
    signals: List[ImpactSignal] = []

    # ── 1. Critical markers ──
    critical_hits = _scan_keywords(text_lower, _CRITICAL_MARKERS, "critical_marker")
    signals.extend(critical_hits)

    # ── 2. SLA/KPI breach ──
    sla_hits = _scan_patterns(text_lower, _SLA_BREACH_PATTERNS, "sla_breach")
    signals.extend(sla_hits)

    # ── 3. Extreme metrics ──
    metric_hits = _scan_extreme_metrics(text)
    signals.extend(metric_hits)

    # ── 4. Temporal risk ──
    temporal_hits = _scan_patterns(text_lower, _TEMPORAL_RISK_PATTERNS, "temporal_risk")
    signals.extend(temporal_hits)

    # ── 5. Cascade risk ──
    cascade_hits = _scan_patterns(text_lower, _CASCADE_PATTERNS, "cascade_risk")
    signals.extend(cascade_hits)

    # ── Compute composite score ──
    has_critical = len(critical_hits) > 0
    has_sla = len(sla_hits) > 0
    has_extreme = len(metric_hits) > 0
    has_temporal = len(temporal_hits) > 0
    has_cascade = len(cascade_hits) > 0

    # Each category contributes to the score with diminishing returns
    # per-hit within category but additive across categories
    category_scores = {
        "critical_marker": _category_score(len(critical_hits), weight=0.35),
        "sla_breach": _category_score(len(sla_hits), weight=0.25),
        "extreme_metric": _category_score(len(metric_hits), weight=0.20),
        "temporal_risk": _category_score(len(temporal_hits), weight=0.12),
        "cascade_risk": _category_score(len(cascade_hits), weight=0.08),
    }

    raw_score = sum(category_scores.values())

    # Multi-category bonus: correlated impact is worse than isolated
    n_categories = sum([has_critical, has_sla, has_extreme, has_temporal, has_cascade])
    if n_categories >= 3:
        raw_score = min(1.0, raw_score * 1.3)
    elif n_categories >= 2:
        raw_score = min(1.0, raw_score * 1.15)

    score = min(1.0, raw_score)

    # ── Summary ──
    summary = _build_summary(
        has_critical, has_sla, has_extreme, has_temporal, has_cascade, score,
    )

    return ImpactSignalResult(
        score=round(score, 4),
        signals=signals,
        has_critical_markers=has_critical,
        has_sla_breach=has_sla,
        has_extreme_metrics=has_extreme,
        has_temporal_risk=has_temporal,
        has_cascade_risk=has_cascade,
        summary=summary,
    )


# ── Private helpers ──

def _scan_keywords(
    text_lower: str, keywords: List[str], category: str,
) -> List[ImpactSignal]:
    """Scan for exact keyword matches."""
    hits: List[ImpactSignal] = []
    for kw in keywords:
        if kw in text_lower:
            hits.append(ImpactSignal(
                category=category, pattern=kw, matched_text=kw,
            ))
    return hits


def _scan_patterns(
    text_lower: str, patterns: List[str], category: str,
) -> List[ImpactSignal]:
    """Scan for regex pattern matches."""
    hits: List[ImpactSignal] = []
    for pat in patterns:
        matches = re.findall(pat, text_lower)
        for m in matches:
            matched = m if isinstance(m, str) else str(m)
            hits.append(ImpactSignal(
                category=category, pattern=pat, matched_text=matched,
            ))
    return hits


def _scan_extreme_metrics(text: str) -> List[ImpactSignal]:
    """Scan for numeric values that indicate dangerous conditions."""
    hits: List[ImpactSignal] = []

    for pattern in _EXTREME_METRIC_PATTERNS:
        for match in pattern.finditer(text):
            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            full_match = match.group(0).lower()
            is_extreme = False
            metric_type = "unknown"

            # Temperature check - explicit prefix
            if any(t in full_match for t in ("temp", "temperatura", "temperature")):
                metric_type = "temperature"
                if value >= _TEMP_CRITICAL_C:
                    is_extreme = True

            # Standalone temperature (°C suffix only, 2-3 digits)
            elif re.search(r'\d{2,3}\s*°?\s*[cC]\b', match.group(0)):
                metric_type = "temperature"
                if value >= _TEMP_CRITICAL_C:
                    is_extreme = True

            # Resource usage check (CPU, memory, disk)
            elif any(r in full_match for r in ("cpu", "memoria", "memory", "disco", "disk", "uso", "usage", "load")):
                metric_type = "resource"
                if value >= _RESOURCE_CRITICAL_PCT:
                    is_extreme = True

            # Availability check (low = bad)
            elif any(a in full_match for a in ("disponibilidad", "availability", "uptime")):
                metric_type = "availability"
                if value < _AVAILABILITY_CRITICAL_PCT:
                    is_extreme = True

            if is_extreme:
                hits.append(ImpactSignal(
                    category="extreme_metric",
                    pattern=pattern.pattern,
                    matched_text=match.group(0),
                    value=value,
                ))

    return hits


def _category_score(n_hits: int, weight: float) -> float:
    """Compute category contribution with diminishing returns per hit."""
    if n_hits == 0:
        return 0.0
    # First hit = full weight, subsequent hits add 30% each (diminishing)
    return weight * min(1.0, 0.7 + 0.3 * min(n_hits, 3) / 3)


def _build_summary(
    has_critical: bool,
    has_sla: bool,
    has_extreme: bool,
    has_temporal: bool,
    has_cascade: bool,
    score: float,
) -> str:
    """Build human-readable summary of detected impact."""
    if score < 0.1:
        return ""

    parts: List[str] = []
    if has_critical:
        parts.append("explicit critical severity markers")
    if has_sla:
        parts.append("SLA/KPI breach indicators")
    if has_extreme:
        parts.append("extreme metric values")
    if has_temporal:
        parts.append("future failure risk language")
    if has_cascade:
        parts.append("cascade/systemic failure indicators")

    if not parts:
        return ""

    joined = ", ".join(parts)
    return f"Impact signals detected: {joined}. Impact score: {score:.2f}."
