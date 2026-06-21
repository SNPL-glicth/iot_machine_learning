"""
HistoricalContextAggregator for aggregating historical similar events.

Aggregates similar events from memory to provide historical context.
"""

from typing import List, Dict, Any
from collections import Counter

from domain.entities.memory import MemoryEvent


class HistoricalContextAggregator:
    """Aggregator for historical context from similar events."""
    
    def aggregate(self, similar_events: List[MemoryEvent]) -> Dict[str, Any]:
        """
        Aggregate historical context from similar events.
        
        Args:
            similar_events: List of similar MemoryEvents
        
        Returns:
            Dictionary with aggregated historical context
        """
        if not similar_events:
            return {
                "similar_event_count": 0,
                "historical_context": "No historical similar events found.",
                "historical_patterns": [],
                "regime_distribution": {},
                "anomaly_score_distribution": {"mean": 0.0, "min": 0.0, "max": 0.0},
            }
        
        # Count regimes
        regime_counts = Counter(event.regime for event in similar_events)
        
        # Calculate anomaly score statistics
        anomaly_scores = [event.anomaly_score for event in similar_events]
        anomaly_stats = {
            "mean": sum(anomaly_scores) / len(anomaly_scores),
            "min": min(anomaly_scores),
            "max": max(anomaly_scores),
        }
        
        # Generate historical context text
        historical_context = self._generate_context_text(
            similar_events, regime_counts, anomaly_stats
        )
        
        # Identify patterns
        patterns = self._identify_patterns(similar_events, regime_counts)
        
        return {
            "similar_event_count": len(similar_events),
            "historical_context": historical_context,
            "historical_patterns": patterns,
            "regime_distribution": dict(regime_counts),
            "anomaly_score_distribution": anomaly_stats,
        }
    
    def _generate_context_text(
        self,
        similar_events: List[MemoryEvent],
        regime_counts: Counter,
        anomaly_stats: Dict[str, float],
    ) -> str:
        """Generate historical context text."""
        count = len(similar_events)
        
        if count == 1:
            text = f"1 evento similar encontrado."
        else:
            text = f"{count} eventos similares encontrados."
        
        # Add regime information
        if regime_counts:
            most_common_regime = regime_counts.most_common(1)[0]
            text += f" La mayoría ocurrieron durante {most_common_regime[0]} ({most_common_regime[1]} eventos)."
        
        # Add anomaly score information
        text += f" Score de anomalía promedio: {anomaly_stats['mean']:.2f} (rango: {anomaly_stats['min']:.2f} - {anomaly_stats['max']:.2f})."
        
        return text
    
    def _identify_patterns(
        self,
        similar_events: List[MemoryEvent],
        regime_counts: Counter,
    ) -> List[str]:
        """Identify common operational patterns."""
        patterns = []
        
        # Regime pattern
        if regime_counts:
            most_common_regime = regime_counts.most_common(1)[0]
            if most_common_regime[1] > len(similar_events) / 2:
                patterns.append(f"Patrones recurrentes en régimen {most_common_regime[0]}")
        
        # Event type pattern
        event_type_counts = Counter(event.event_type for event in similar_events)
        if event_type_counts:
            most_common_type = event_type_counts.most_common(1)[0]
            if most_common_type[1] > len(similar_events) / 2:
                patterns.append(f"Mayoría de eventos tipo {most_common_type[0]}")
        
        return patterns
