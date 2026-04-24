"""FinancialEntityExtractor — extract financial entities from text.

Examples: $68,450, USD 45,000, BTC, -12.5%
"""

from __future__ import annotations

import re
from typing import List, Set


class FinancialEntityExtractor:
    """Extract financial entities using regex patterns.
    
    Examples: $2,347,891, USD 45,000, BTC, AAPL, -12.5%
    """
    
    # Financial patterns: ordered from most specific to least specific
    PATTERNS = [
        # Montos con símbolo $: $68,450 | $2,347,891.50
        (r'\$[\d,]+(?:\.\d{2})?', 'amount_usd'),
        # Montos con prefijo USD: USD 45,000 | USD 2,347,891
        (r'USD\s*[\d,]+(?:\.\d{2})?', 'amount_usd'),
        # Montos con prefijo EUR: EUR 45,000 | EUR 2,347,891
        (r'EUR\s*[\d,]+(?:\.\d{2})?', 'amount_eur'),
        # Porcentajes con signo: -12.5% | +34% | +2.5%
        (r'[+-]?\d+(?:\.\d+)?%', 'percentage_change'),
        # Códigos de activos (2-5 mayúsculas) con contexto financiero
        (r'\b[A-Z]{2,5}\b(?=\s+(?:price|value|trading|at|\$|USD|EUR|\d))', 'asset_code'),
        # Criptomonedas comunes
        (r'\b(BTC|ETH|XRP|LTC|ADA|DOT|SOL|AVAX|MATIC|LINK|UNI|AAVE|COMP|MKR|SNX|YFI|CRV|BAL|LRC|ZRX|KNC)\b', 'crypto'),
    ]
    
    def extract(self, text: str) -> List[str]:
        """Extract financial entities from text.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            List of extracted financial entity strings.
        """
        if not text or not isinstance(text, str):
            return []
        
        entities: List[str] = []
        seen: Set[str] = set()
        
        for pattern, entity_type in self.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity_text = match.group(0)
                # Normalizar: uppercase para cripto, mantener original para montos
                normalized = entity_text.upper() if entity_type == 'crypto' else entity_text
                
                # Deduplicar case-insensitive
                key = normalized.lower()
                if key not in seen:
                    entities.append(normalized)
                    seen.add(key)
        
        return entities
    
    def supports_domain(self, domain: str) -> bool:
        """Works for finance and related domains."""
        return domain.lower() in ('finance', 'financial', 'trading', 'investment', 'crypto', 'general')
    
    def is_available(self) -> bool:
        """Always available (regex-based)."""
        return True
