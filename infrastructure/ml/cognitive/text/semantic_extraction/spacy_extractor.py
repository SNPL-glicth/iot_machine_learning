"""SpacyEntityExtractor — NLP-based entity extraction using spaCy.

Provides a layer of real NLP on top of regex extractors.
Graceful degradation if spaCy is not installed.
"""

from __future__ import annotations

from typing import List, Set


class SpacyEntityExtractor:
    """Extract entities using spaCy NLP.
    
    Graceful degradation: returns empty list if spaCy not available.
    
    Relevant entity labels for industrial/financial documents:
    - ORG: Organizations (companies, manufacturers)
    - MONEY: Monetary values
    - PERCENT: Percentages
    - LOC: Locations (plants, facilities)
    - PRODUCT: Products (equipment models)
    - DATE: Dates (relevant for maintenance logs)
    """
    
    RELEVANT_LABELS = {"ORG", "MONEY", "PERCENT", "LOC", "PRODUCT", "DATE"}
    
    def __init__(self):
        self._nlp = None
        self._available = False
        try:
            import spacy
            self._nlp = spacy.load("es_core_news_sm")
            self._available = True
        except Exception:
            # Graceful degradation if spaCy not installed
            pass
    
    def is_available(self) -> bool:
        """Check if spaCy is available and loaded."""
        return self._available
    
    def extract(self, text: str) -> List[str]:
        """Extract entities using spaCy NER.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            List of extracted entity texts.
        """
        if not self._available or not text or not isinstance(text, str):
            return []
        
        doc = self._nlp(text)
        entities: List[str] = []
        seen: Set[str] = set()
        
        for ent in doc.ents:
            if ent.label_ in self.RELEVANT_LABELS:
                entity_text = ent.text.strip()
                # Filtrar entidades muy cortas o vacías
                if len(entity_text) > 2:
                    # Deduplicar case-insensitive
                    key = entity_text.lower()
                    if key not in seen:
                        entities.append(entity_text)
                        seen.add(key)
        
        return entities
