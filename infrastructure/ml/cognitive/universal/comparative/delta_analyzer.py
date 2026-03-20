"""Build human-readable delta conclusion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def build_delta_conclusion(
    severity_delta_pct: float,
    urgency_delta_pct: float,
    topic_overlap_pct: float,
    top_similar: List[Dict[str, Any]],
    domain: str,
) -> str:
    """Build human-readable comparison conclusion.

    Args:
        severity_delta_pct: % change in severity
        urgency_delta_pct: % change in urgency
        topic_overlap_pct: % of common topics
        top_similar: Top 3 similar past incidents
        domain: Document domain

    Returns:
        Multi-sentence comparison conclusion
    """
    if not top_similar:
        return "No similar past incidents found for comparison."
    
    lines: List[str] = []
    
    domain_label = domain.capitalize() if domain != "general" else ""
    severity_direction = "more severe" if severity_delta_pct > 0 else "less severe"
    severity_abs = abs(severity_delta_pct)
    
    most_similar = top_similar[0]
    
    if domain_label:
        lines.append(
            f"This {domain_label} incident is {severity_abs:.0f}% {severity_direction} "
            f"than the most similar past incident (ID: {most_similar['doc_id'][:8]}, "
            f"similarity: {most_similar['score']:.0%})."
        )
    else:
        lines.append(
            f"This incident is {severity_abs:.0f}% {severity_direction} "
            f"than the most similar past incident."
        )
    
    if most_similar.get("resolution_time"):
        lines.append(
            f"That incident resolved in {most_similar['resolution_time']:.1f} hours."
        )
    
    if topic_overlap_pct > 50:
        lines.append(
            f"Topic overlap: {topic_overlap_pct:.0f}% "
            f"(shared themes detected across {len(top_similar)} similar incidents)."
        )
    
    resolution_prob, time_est = estimate_resolution(top_similar)
    
    if resolution_prob is not None:
        lines.append(
            f"Estimated resolution probability: {resolution_prob:.0%} "
            f"based on {len(top_similar)} similar past incidents."
        )
    
    if time_est:
        lines.append(f"Estimated resolution time: {time_est}.")
    
    return " ".join(lines)


def estimate_resolution(
    top_similar: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[str]]:
    """Estimate resolution probability and time from historical data.

    Args:
        top_similar: Top 3 similar past incidents with resolution metadata

    Returns:
        Tuple of (probability: float | None, time_estimate: str | None)
    """
    if not top_similar:
        return None, None
    
    resolved = []
    total_weight = 0.0
    
    for incident in top_similar:
        score = incident.get("score", 0.0)
        res_time = incident.get("resolution_time")
        
        if res_time is not None and res_time > 0:
            resolved.append((res_time, score))
        
        total_weight += score
    
    if not resolved:
        return None, None
    
    probability = sum(score for _, score in resolved) / max(total_weight, 0.01)
    
    resolution_times = [rt for rt, _ in resolved]
    
    resolution_times_sorted = sorted(resolution_times)
    n = len(resolution_times_sorted)
    
    if n == 1:
        median_hours = resolution_times_sorted[0]
    elif n % 2 == 0:
        median_hours = (resolution_times_sorted[n//2 - 1] + resolution_times_sorted[n//2]) / 2
    else:
        median_hours = resolution_times_sorted[n//2]
    
    time_estimate = _format_time_estimate(median_hours)
    
    return probability, time_estimate


def _format_time_estimate(hours: float) -> str:
    """Format hours as human-readable range."""
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes}-{minutes + 15} minutes"
    
    if hours < 2:
        return "1-2 hours"
    
    if hours < 4:
        return "2-4 hours"
    
    if hours < 8:
        return "4-8 hours"
    
    if hours < 24:
        return "8-24 hours"
    
    days = int(hours / 24)
    return f"{days}-{days + 1} days"
