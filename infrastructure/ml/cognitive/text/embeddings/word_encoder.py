"""Word-level encoder with TF-IDF base + adaptive domain weights.

Extends LightweightEmbedding with PlasticityTracker for domain-specific learning.
Pure numpy, no external dependencies beyond existing codebase.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

from iot_machine_learning.infrastructure.ml.cognitive.neural.attention.attention_embedding import (
    LightweightEmbedding,
)
from iot_machine_learning.infrastructure.ml.cognitive.plasticity.base import PlasticityTracker


class WordEncoder:
    """Word-level encoder with adaptive domain weights.
    
    Combines:
    - TF-IDF base weights (fixed, corpus-level)
    - Domain-specific adaptive weights (learned per domain via PlasticityTracker)
    - Entropy filtering for low-information tokens
    
    Args:
        vocab: Word vocabulary dict (word -> index)
        domain_hint: Domain identifier for adaptive weights
        output_dim: Target embedding dimension
        plasticity_tracker: Optional PlasticityTracker for adaptive learning
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        domain_hint: str = "general",
        output_dim: int = 64,
        plasticity_tracker: Optional[PlasticityTracker] = None,
    ) -> None:
        self.vocab = vocab
        self.domain_hint = domain_hint
        self.output_dim = output_dim
        self.base_embedder = LightweightEmbedding(vocab, len(vocab))
        
        # Adaptive weight tracker per domain
        self._plasticity = plasticity_tracker or PlasticityTracker(
            alpha=0.15,
            min_weight=0.05,
            max_regimes=10,
        )
        
        # Domain-specific word importance (learned online)
        self._domain_weights: Dict[str, float] = {}
        self._word_freq: Dict[str, int] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9áéíóúüñ\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]
    
    def _update_domain_weights(self, tokens: List[str]) -> None:
        """Update domain-specific word importance scores."""
        for token in tokens:
            if token not in self._word_freq:
                self._word_freq[token] = 0
            self._word_freq[token] += 1
            
            # Compute importance as frequency-based signal
            freq = self._word_freq[token]
            if freq >= 2:  # Seen at least twice
                # Use plasticity to track word importance
                self._plasticity.update(
                    regime=self.domain_hint,
                    engine_name=f"word_{token}",
                    prediction_error=1.0 / (freq + 1),
                )
        
        # Extract learned weights from plasticity
        engines = [f"word_{t}" for t in tokens if f"word_{t}" in self._plasticity._accuracy.get(self.domain_hint, {})]
        if engines:
            weights = self._plasticity.get_weights(self.domain_hint, engines)
            for eng_name, weight in weights.items():
                word = eng_name.replace("word_", "")
                self._domain_weights[word] = weight
    
    def encode_word(self, word: str) -> np.ndarray:
        """Encode single word to vector.
        
        Args:
            word: Input word string
            
        Returns:
            Word-level vector [output_dim]
        """
        if word not in self.vocab:
            return np.zeros(self.output_dim, dtype=np.float32)
        
        # Get base TF-IDF embedding
        base_vec = self.base_embedder.embed_sentence(word)
        
        # Pad or truncate to output_dim
        if len(base_vec) < self.output_dim:
            base_vec = base_vec + [0.0] * (self.output_dim - len(base_vec))
        else:
            base_vec = base_vec[:self.output_dim]
        
        # Apply adaptive domain weight
        adaptive_mult = self._domain_weights.get(word, 0.5)
        
        # Boost vector magnitude by adaptive weight
        vec = np.array(base_vec, dtype=np.float32) * (0.5 + adaptive_mult)
        
        return vec
    
    def encode_words(self, words: List[str]) -> np.ndarray:
        """Encode multiple words to matrix.
        
        Args:
            words: List of word strings
            
        Returns:
            Matrix [n_words, output_dim]
        """
        if not words:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        
        # Update domain weights based on current batch
        self._update_domain_weights(words)
        
        return np.array([self.encode_word(w) for w in words], dtype=np.float32)
    
    def get_word_importance(self, word: str) -> float:
        """Get learned importance score for word."""
        return self._domain_weights.get(word, 0.5)
    
    def get_stats(self) -> dict:
        """Get encoder statistics."""
        return {
            "vocab_size": len(self.vocab),
            "domain": self.domain_hint,
            "learned_words": len(self._domain_weights),
            "tracked_words": len(self._word_freq),
            "avg_importance": float(np.mean(list(self._domain_weights.values()))) if self._domain_weights else 0.0,
        }
    
    @classmethod
    def from_corpus(
        cls,
        texts: List[str],
        domain_hint: str = "general",
        max_vocab: int = 1000,
        output_dim: int = 64,
    ) -> "WordEncoder":
        """Factory: build encoder from text corpus.
        
        Args:
            texts: List of text documents
            domain_hint: Domain identifier
            max_vocab: Maximum vocabulary size
            output_dim: Output dimension
            
        Returns:
            Configured WordEncoder
        """
        # Build vocabulary from corpus
        word_freq = {}
        for text in texts:
            tokens = text.lower().split()
            for t in tokens:
                t = re.sub(r'[^a-z0-9áéíóúüñ]', '', t)
                if len(t) > 2:
                    word_freq[t] = word_freq.get(t, 0) + 1
        
        # Take top words up to max_vocab
        top_words = sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)[:max_vocab]
        vocab = {w: i for i, w in enumerate(top_words)}
        
        return cls(vocab=vocab, domain_hint=domain_hint, output_dim=output_dim)
