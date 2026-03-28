"""Phrase-level encoder for n-grams with persistence tracking.

Encodes multi-word phrases (2-3 grams) with adaptive importance learning.
Uses Bayesian updates for phrase significance tracking.
Pure numpy, no external dependencies.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from iot_machine_learning.infrastructure.ml.inference.bayesian.posterior import BayesianUpdater
from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import GaussianPrior

from .word_encoder import WordEncoder


class PhraseEncoder:
    """Phrase-level encoder for n-grams (2-3 words).
    
    Combines word vectors into phrase representations with:
    - N-gram composition (2-3 word phrases)
    - Persistence tracking (phrase seen N times gets higher weight)
    - Bayesian significance learning per domain
    
    Args:
        word_encoder: WordEncoder for word-level vectors
        domain_hint: Domain identifier for adaptive learning
        output_dim: Target embedding dimension
        min_persistence: Minimum occurrences to retain phrase (default 2)
    """
    
    def __init__(
        self,
        word_encoder: WordEncoder,
        domain_hint: str = "general",
        output_dim: int = 64,
        min_persistence: int = 2,
    ) -> None:
        self.word_encoder = word_encoder
        self.domain_hint = domain_hint
        self.output_dim = output_dim
        self.min_persistence = min_persistence
        
        # Phrase persistence tracking
        self._phrase_counts: Dict[str, int] = {}
        self._phrase_significance: Dict[str, GaussianPrior] = {}
        self._bayesian = BayesianUpdater()
        
        # Active phrase registry per domain
        self._domain_phrases: Dict[str, set] = {}
    
    def _extract_ngrams(self, words: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        """Extract n-grams from word list."""
        if len(words) < n:
            return []
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def _phrase_key(self, phrase: Tuple[str, ...]) -> str:
        """Convert phrase tuple to canonical key."""
        return "|".join(phrase)
    
    def _update_persistence(self, phrases: List[Tuple[str, ...]]) -> None:
        """Update phrase persistence counts and significance."""
        for phrase in phrases:
            key = self._phrase_key(phrase)
            
            # Increment count
            if key not in self._phrase_counts:
                self._phrase_counts[key] = 0
            self._phrase_counts[key] += 1
            
            count = self._phrase_counts[key]
            
            # Bayesian update for significance
            if key not in self._phrase_significance:
                self._phrase_significance[key] = GaussianPrior(mu_0=0.5, sigma2_0=1.0)
            
            # Higher count = higher significance belief
            observation = np.array([min(1.0, count / 5.0)])  # Saturate at 5
            posterior = self._bayesian.update(self._phrase_significance[key], observation)
            self._phrase_significance[key] = posterior.to_prior()
            
            # Track domain association
            if self.domain_hint not in self._domain_phrases:
                self._domain_phrases[self.domain_hint] = set()
            if count >= self.min_persistence:
                self._domain_phrases[self.domain_hint].add(key)
    
    def _compose_phrase_vector(self, phrase: Tuple[str, ...]) -> np.ndarray:
        """Compose phrase vector from word vectors.
        
        Strategy:
        - 2-gram: weighted average with position bias (second word gets more weight)
        - 3-gram: centroid with trigram-specific blending
        """
        if not phrase:
            return np.zeros(self.output_dim, dtype=np.float32)
        
        word_vecs = [self.word_encoder.encode_word(w) for w in phrase]
        
        if len(phrase) == 2:
            # Bigram: position-weighted (0.4, 0.6) - second word often more informative
            vec = 0.4 * word_vecs[0] + 0.6 * word_vecs[1]
        elif len(phrase) == 3:
            # Trigram: smooth blending with center emphasis
            vec = 0.25 * word_vecs[0] + 0.5 * word_vecs[1] + 0.25 * word_vecs[2]
        else:
            # Fallback: simple mean
            vec = np.mean(word_vecs, axis=0)
        
        # Apply significance boost from persistence
        key = self._phrase_key(phrase)
        if key in self._phrase_significance:
            sig = self._phrase_significance[key].mu_0
            vec = vec * (0.5 + sig)  # Boost by significance
        
        return vec.astype(np.float32)
    
    def encode_phrase(self, phrase: Tuple[str, ...]) -> np.ndarray:
        """Encode single phrase tuple to vector.
        
        Args:
            phrase: Tuple of words forming phrase
            
        Returns:
            Phrase-level vector [output_dim]
        """
        return self._compose_phrase_vector(phrase)
    
    def encode_phrases(self, words: List[str]) -> Tuple[np.ndarray, List[Tuple[str, ...]]]:
        """Encode all phrases from word list.
        
        Args:
            words: List of words
            
        Returns:
            Tuple of (phrase_matrix [n_phrases, output_dim], phrase_list)
        """
        if len(words) < 2:
            return np.zeros((0, self.output_dim), dtype=np.float32), []
        
        # Extract bigrams and trigrams
        bigrams = self._extract_ngrams(words, n=2)
        trigrams = self._extract_ngrams(words, n=3) if len(words) >= 3 else []
        all_phrases = bigrams + trigrams
        
        if not all_phrases:
            return np.zeros((0, self.output_dim), dtype=np.float32), []
        
        # Update persistence tracking
        self._update_persistence(all_phrases)
        
        # Filter by persistence threshold
        persistent_phrases = [
            p for p in all_phrases
            if self._phrase_counts.get(self._phrase_key(p), 0) >= self.min_persistence
        ]
        
        # If too few persistent, include all
        if len(persistent_phrases) < 2 and all_phrases:
            persistent_phrases = all_phrases[:10]  # Cap at 10
        
        # Encode phrases
        phrase_vecs = np.array([
            self.encode_phrase(p) for p in persistent_phrases
        ], dtype=np.float32)
        
        return phrase_vecs, persistent_phrases
    
    def get_phrase_importance(self, phrase: Tuple[str, ...]) -> float:
        """Get learned importance score for phrase."""
        key = self._phrase_key(phrase)
        if key in self._phrase_significance:
            return float(self._phrase_significance[key].mu_0)
        return 0.5
    
    def get_stats(self) -> dict:
        """Get encoder statistics."""
        domain_phrases = self._domain_phrases.get(self.domain_hint, set())
        sig_values = [
            self._phrase_significance[k].mu_0
            for k in self._phrase_significance
        ] if self._phrase_significance else []
        
        return {
            "domain": self.domain_hint,
            "total_phrases": len(self._phrase_counts),
            "persistent_phrases": len(domain_phrases),
            "min_persistence": self.min_persistence,
            "avg_significance": float(np.mean(sig_values)) if sig_values else 0.0,
        }
