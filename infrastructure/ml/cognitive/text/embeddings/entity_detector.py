"""Entity detector using hybrid embeddings — replaces regex extraction.

Detects entities by semantic vector magnitude, not pattern matching.
Uses hybrid character+word+phrase embeddings for semantic understanding.
Pure numpy, no external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from iot_machine_learning.infrastructure.ml.cognitive.plasticity.base import PlasticityTracker
from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers.keyword_config import (
    DOMAIN_KEYWORDS,
)

from .char_encoder import CharacterEncoder
from .entropy_filter import EntropyFilter
from .hybrid_embedder import HybridEmbedder
from .phrase_encoder import PhraseEncoder
from .word_encoder import WordEncoder


@dataclass(frozen=True)
class EntityResult:
    """Result of entity extraction."""
    entities: List[str]
    entity_types: Dict[str, str]
    confidence_scores: Dict[str, float]
    semantic_vectors: Dict[str, List[float]]
    
    def to_list(self) -> List[str]:
        """Return entities as simple list."""
        return self.entities


class HybridEntityDetector:
    """Entity detector using hybrid semantic embeddings.
    
    Replaces hardcoded regex with vector-based entity detection:
    - Token is entity if semantic_vector magnitude > threshold
    - Entity type classified by domain similarity of vector
    - No hardcoded patterns — pure vector similarity
    
    Args:
        domain_hint: Domain for adaptive behavior
        magnitude_threshold: Min vector magnitude to be entity (default 0.3)
        plasticity_tracker: Optional tracker for adaptive learning
    """
    
    def __init__(
        self,
        domain_hint: str = "general",
        magnitude_threshold: float = 0.3,
        plasticity_tracker: Optional[PlasticityTracker] = None,
    ) -> None:
        self.domain_hint = domain_hint
        self.magnitude_threshold = magnitude_threshold
        
        # Build vocabulary from domain keywords
        self._vocab = self._build_domain_vocab(domain_hint)
        
        # Initialize encoders
        self._entropy_filter = EntropyFilter(threshold=0.4)
        self._char_encoder = CharacterEncoder(
            output_dim=32,
            entropy_filter=self._entropy_filter,
        )
        self._word_encoder = WordEncoder(
            vocab=self._vocab,
            domain_hint=domain_hint,
            output_dim=64,
        )
        self._phrase_encoder = PhraseEncoder(
            word_encoder=self._word_encoder,
            domain_hint=domain_hint,
            output_dim=64,
        )
        self._hybrid_embedder = HybridEmbedder(
            char_encoder=self._char_encoder,
            word_encoder=self._word_encoder,
            phrase_encoder=self._phrase_encoder,
            domain_hint=domain_hint,
            output_dim=128,
        )
        
        # Domain reference vectors for classification
        self._domain_refs = self._build_domain_references()
    
    def _build_domain_vocab(self, domain_hint: str) -> Dict[str, int]:
        """Build vocabulary from domain keywords."""
        keywords = DOMAIN_KEYWORDS.get(domain_hint, DOMAIN_KEYWORDS.get("general", []))
        # Flatten all domain keywords
        all_words = set()
        for domain, words in DOMAIN_KEYWORDS.items():
            all_words.update(words)
        all_words.update(keywords)
        return {w: i for i, w in enumerate(sorted(all_words))}
    
    def _build_domain_references(self) -> Dict[str, np.ndarray]:
        """Build reference vectors for each domain."""
        refs = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if keywords:
                # Average word vectors for domain
                vecs = [self._word_encoder.encode_word(w) for w in keywords[:20]]
                refs[domain] = np.mean(vecs, axis=0)
        return refs
    
    def _is_entity_candidate(self, token: str) -> bool:
        """Check if token could be an entity (basic filters)."""
        if len(token) < 3:
            return False
        if token.isdigit():
            return False
        # Filter common stop words
        stop_words = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy", "did", "she", "use", "her", "way", "many", "oil", "sit", "set", "run", "eat", "far", "sea", "eye", "ago", "off", "too", "any", "say", "man", "try", "ask", "end", "why", "let", "put", "say", "she", "try", "way", "own", "say"}
        if token.lower() in stop_words:
            return False
        return True
    
    def _classify_entity_type(self, token: str, vector: np.ndarray) -> str:
        """Classify entity type by domain vector similarity."""
        if not self._domain_refs:
            return "general"
        
        # Compute cosine similarity to each domain reference
        best_domain = "general"
        best_sim = -1
        
        vec_norm = np.linalg.norm(vector)
        if vec_norm == 0:
            return "general"
        vec_unit = vector / vec_norm
        
        for domain, ref_vec in self._domain_refs.items():
            ref_norm = np.linalg.norm(ref_vec)
            if ref_norm == 0:
                continue
            ref_unit = ref_vec / ref_norm
            
            # Cosine similarity (dot product of unit vectors)
            sim = np.dot(vec_unit[:len(ref_unit)], ref_unit[:len(vec_unit)])
            
            if sim > best_sim:
                best_sim = sim
                best_domain = domain
        
        return best_domain if best_sim > 0.3 else "general"
    
    def extract_entities(self, text: str) -> EntityResult:
        """Extract entities from text using hybrid embeddings.
        
        Args:
            text: Input text string
            
        Returns:
            EntityResult with entities, types, and confidence
        """
        entities = []
        entity_types = {}
        confidence_scores = {}
        semantic_vectors = {}
        
        if not text:
            return EntityResult([], {}, {}, {})
        
        # Get hybrid embeddings
        vectors, tokens = self._hybrid_embedder.embed_text(text)
        
        if len(vectors) == 0:
            return EntityResult([], {}, {}, {})
        
        # Detect entities by vector magnitude
        for i, (token, vec) in enumerate(zip(tokens, vectors)):
            if not self._is_entity_candidate(token):
                continue
            
            magnitude = float(np.linalg.norm(vec))
            
            # Entity if magnitude exceeds threshold
            if magnitude >= self.magnitude_threshold:
                # Boost confidence for capitalized or special-pattern tokens
                confidence = min(1.0, magnitude)
                if token[0].isupper():
                    confidence = min(1.0, confidence * 1.2)
                if any(c.isdigit() for c in token):
                    confidence = min(1.0, confidence * 1.1)
                
                entities.append(token)
                entity_types[token] = self._classify_entity_type(token, vec)
                confidence_scores[token] = confidence
                semantic_vectors[token] = vec.tolist()
        
        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            lower_e = e.lower()
            if lower_e not in seen:
                seen.add(lower_e)
                unique_entities.append(e)
        
        return EntityResult(
            entities=unique_entities,
            entity_types={e: entity_types[e] for e in unique_entities},
            confidence_scores={e: confidence_scores[e] for e in unique_entities},
            semantic_vectors={e: semantic_vectors[e] for e in unique_entities},
        )
    
    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "domain": self.domain_hint,
            "vocab_size": len(self._vocab),
            "threshold": self.magnitude_threshold,
            "fusion_weights": self._hybrid_embedder.get_fusion_weights(),
        }
