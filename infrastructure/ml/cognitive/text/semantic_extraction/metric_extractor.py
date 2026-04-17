"""MetricExtractor — extract numeric metrics with units."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from iot_machine_learning.domain.entities.semantic_extraction import (
    EntityType,
    MetricAttributes,
    SemanticEntity,
)


class MetricExtractor:
    """Extract metric entities (value + unit) from text.
    
    Examples: 3401 PSI, 441.2°C, 85%, 1200 rpm
    """
    
    # Unit patterns with metric classes
    UNIT_PATTERNS = [
        # Pressure
        (r'\b(\d{1,4}(?:\.\d{1,2})?)\s*(PSI|BAR|KPA|MPA|PA|ATM|PSIG|BARG)\b', 'pressure'),
        (r'\b(\d{1,4}(?:\.\d{1,2})?)\s*(PSI|BAR|KPA|MPA|PA|ATM|PSIG|BARG)\b', 'pressure'),
        # Temperature  
        (r'\b(\d{1,3}(?:\.\d{1,2})?)\s*°?\s*(C|F|K|°C|°F|°K|CELCIUS|FAHRENHEIT)\b', 'temperature'),
        # Flow
        (r'\b(\d{1,5}(?:\.\d{1,2})?)\s*(GPM|LPM|L\/MIN|M3\/H|CFM|SCFM)\b', 'flow'),
        # Speed/RPM
        (r'\b(\d{1,5}(?:\.\d{1,2})?)\s*(RPM|RPH|HZ|HZ)\b', 'speed'),
        # Level
        (r'\b(\d{1,3}(?:\.\d{1,2})?)\s*(M|CM|MM|FT|IN|%|PERCENT)\s*(level|Level)?\b', 'level'),
        # Power
        (r'\b(\d{1,5}(?:\.\d{1,2})?)\s*(KW|MW|HP|W|WATT)\b', 'power'),
        # Vibration
        (r'\b(\d{1,3}(?:\.\d{1,2})?)\s*(MM\/S|IPS|G|MICRON)\b', 'vibration'),
        # Current/Voltage
        (r'\b(\d{1,4}(?:\.\d{1,2})?)\s*(A|AMP|AMPS|V|VOLT|VOLTS|KV|MA)\b', 'electrical'),
        # Percentage
        (r'\b(\d{1,3}(?:\.\d{1,2})?)\s*%\b', 'percentage'),
        # Generic numbers with units
        (r'\b(\d{1,6}(?:\.\d{1,3})?)\s*(KG|G|MG|LB|TON|L|GAL|ML|OZ)\b', 'mass_volume'),
    ]
    
    # Normal operating ranges (for anomaly detection)
    REFERENCE_RANGES = {
        'pressure': (0, 5000),      # PSI
        'temperature': (-40, 500),  # °C
        'flow': (0, 10000),         # GPM
        'speed': (0, 5000),         # RPM
        'level': (0, 100),          # %
        'power': (0, 10000),        # kW
        'vibration': (0, 25),       # mm/s
        'electrical': (0, 1000),    # A/V
        'percentage': (0, 100),     # %
    }
    
    # Unit normalization
    UNIT_NORMALIZATION = {
        'PSI': 'PSI', 'PSIG': 'PSI', 'BAR': 'BAR', 'BARG': 'BAR',
        'KPA': 'KPA', 'MPA': 'MPA', 'PA': 'PA', 'ATM': 'ATM',
        'C': '°C', '°C': '°C', 'F': '°F', '°F': '°F', 'K': 'K',
        'GPM': 'GPM', 'LPM': 'LPM', 'L/MIN': 'L/min', 'M3/H': 'm³/h',
        'RPM': 'RPM', 'HZ': 'Hz',
        '%': '%', 'PERCENT': '%',
    }
    
    def extract(
        self,
        text: str,
        domain_hint: Optional[str] = None,
    ) -> List[SemanticEntity]:
        """Extract metric entities from text."""
        entities = []
        seen_positions = set()
        
        for pattern, metric_class in self.UNIT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                
                # Skip overlaps
                if any(start <= p < end or start < p <= end for p in seen_positions):
                    continue
                
                try:
                    value = float(match.group(1))
                except (ValueError, IndexError):
                    continue
                
                # Extract unit
                try:
                    unit_raw = match.group(2).upper() if len(match.groups()) > 1 else ''
                    unit = self.UNIT_NORMALIZATION.get(unit_raw, unit_raw)
                except IndexError:
                    unit = ''
                
                # Check if out of normal range
                is_out_of_range, deviation = self._check_range(value, metric_class)
                
                # Get context window
                context_start = max(0, start - 35)
                context_end = min(len(text), end + 35)
                context_window = text[context_start:context_end]
                
                # Build attributes
                ref_range = self.REFERENCE_RANGES.get(metric_class)
                attrs = MetricAttributes(
                    value=value,
                    unit=unit,
                    metric_class=metric_class,
                    is_out_of_range=is_out_of_range,
                    reference_range=ref_range,
                    deviation_percent=deviation,
                )
                
                entity = SemanticEntity(
                    text=match.group(0),
                    normalized=f"{value} {unit}",
                    entity_type=EntityType.METRIC,
                    start_pos=start,
                    end_pos=end,
                    confidence=0.9 if is_out_of_range else 0.8,
                    context_window=context_window,
                    attributes=attrs.to_dict(),
                    relations=[],
                )
                
                entities.append(entity)
                seen_positions.update(range(start, end))
        
        return entities
    
    def _check_range(self, value: float, metric_class: str) -> Tuple[bool, float]:
        """Check if value is outside normal operating range."""
        ref = self.REFERENCE_RANGES.get(metric_class)
        if not ref:
            return False, 0.0
        
        min_val, max_val = ref
        
        if value < min_val:
            deviation = abs(min_val - value) / abs(min_val) * 100 if min_val != 0 else 100
            return True, deviation
        elif value > max_val:
            deviation = (value - max_val) / max_val * 100
            return True, deviation
        
        return False, 0.0
    
    def supports_domain(self, domain: str) -> bool:
        """Works for all domains with physical measurements."""
        return True
