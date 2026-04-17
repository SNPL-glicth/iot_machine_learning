"""EquipmentExtractor — regex-based equipment identification."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from iot_machine_learning.domain.entities.semantic_extraction import (
    EntityType,
    EquipmentAttributes,
    SemanticEntity,
)


class EquipmentExtractor:
    """Extract equipment identifiers using regex patterns.
    
    Examples: C-12, V-23, PUMP-A, COMP-01, MOTOR-B3
    """
    
    # Equipment patterns: prefix + separator + alphanumeric ID
    EQUIPMENT_PATTERNS = [
        # C-12, V-23, P-5 (single letter + dash + number)
        (r'\b([A-Z])[-\s]?(\d{1,3})\b', 'single_letter'),
        # COMP-01, PUMP-A, MOTOR-B3 (abbreviation + dash + alphanum)
        (r'\b(COMP|PUMP|MOTOR|VALVE|VLV|FAN|BLR|GEN|TX|HV|CS|CT|EXH)[-\s]?([A-Z0-9]{1,4})\b', 'abbreviation'),
        # NODE-123, TMP-45, DEV-789
        (r'\b(NODE|TMP|TEMP|SERVER|ROUTER|SWITCH|DEV|DEVICE)[-\s]?(\w{1,6})\b', 'infrastructure'),
        # C12, V23 (letter directly followed by number, no separator)
        (r'\b([A-Z])(\d{2,3})\b', 'compact'),
    ]
    
    # Equipment class mapping from prefix
    EQUIPMENT_CLASSES = {
        'C': 'compressor', 'COMP': 'compressor',
        'P': 'pump', 'PUMP': 'pump',
        'V': 'valve', 'VLV': 'valve', 'VALVE': 'valve',
        'M': 'motor', 'MOTOR': 'motor',
        'F': 'fan', 'FAN': 'fan',
        'BLR': 'boiler', 'B': 'boiler',
        'GEN': 'generator', 'G': 'generator',
        'TX': 'transformer', 'HV': 'high_voltage',
        'NODE': 'node', 'TMP': 'temperature_sensor',
        'SERVER': 'server', 'ROUTER': 'router',
    }
    
    def extract(
        self,
        text: str,
        domain_hint: Optional[str] = None,
    ) -> List[SemanticEntity]:
        """Extract equipment entities from text."""
        entities = []
        seen_positions = set()
        
        for pattern, pattern_type in self.EQUIPMENT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                
                # Avoid overlapping matches
                if any(start <= p < end or start < p <= end for p in seen_positions):
                    continue
                
                prefix = match.group(1).upper()
                equip_id = match.group(2).upper()
                
                # Skip if looks like a common word (heuristic)
                if self._is_false_positive(prefix, equip_id, pattern_type):
                    continue
                
                equipment_class = self.EQUIPMENT_CLASSES.get(prefix, 'equipment')
                normalized = f"{prefix}{equip_id}"
                
                # Get context window
                context_start = max(0, start - 30)
                context_end = min(len(text), end + 30)
                context_window = text[context_start:context_end]
                
                attrs = EquipmentAttributes(
                    equipment_class=equipment_class,
                    equipment_id=equip_id,
                    parent_system=None,
                    is_critical_path=self._is_critical_equipment(equipment_class),
                )
                
                entity = SemanticEntity(
                    text=match.group(0),
                    normalized=normalized,
                    entity_type=EntityType.EQUIPMENT,
                    start_pos=start,
                    end_pos=end,
                    confidence=0.85 if pattern_type == 'abbreviation' else 0.75,
                    context_window=context_window,
                    attributes=attrs.to_dict(),
                    relations=[],
                )
                
                entities.append(entity)
                seen_positions.update(range(start, end))
        
        return entities
    
    def _is_false_positive(self, prefix: str, equip_id: str, pattern_type: str) -> bool:
        """Check if match is likely a false positive."""
        # Common words to exclude
        common_words = ['A', 'I', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'HE',
                       'IF', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'OF', 'ON', 'OR',
                       'OX', 'PM', 'SO', 'TO', 'TV', 'UP', 'US', 'WE']
        
        if pattern_type == 'single_letter' and prefix in common_words:
            return True
        
        # Check if ID looks like a year (1950-2050)
        if equip_id.isdigit() and 1950 <= int(equip_id) <= 2050:
            return True
        
        return False
    
    def _is_critical_equipment(self, equipment_class: str) -> bool:
        """Check if equipment type is typically on critical path."""
        critical_types = ['compressor', 'turbine', 'reactor', 'boiler', 'generator']
        return equipment_class.lower() in critical_types
    
    def supports_domain(self, domain: str) -> bool:
        """Works for all domains."""
        return True
