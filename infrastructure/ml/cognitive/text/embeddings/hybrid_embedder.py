"""Hybrid embedder — fuses character, word, and phrase vectors.

Combines three encoding levels with learned fusion weights per domain.
Uses PlasticityTracker for adaptive α, β, γ learning.
Pure numpy, no external dependencies.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from iot_machine_learning.infrastructure.ml.cognitive.plasticity.base import PlasticityTracker

from .char_encoder import CharacterEncoder
from .word_encoder import WordEncoder
from .phrase_encoder import PhraseEncoder


class HybridEmbedder:
    """Hybrid embedder fusing char + word + phrase vectors.
    
    Fusion: hybrid_vector = α*char_vector + β*word_vector + γ*phrase_vector
    
    α, β, γ are learned per domain via PlasticityTracker.
    
    Args:
        char_encoder: CharacterEncoder instance
        word_encoder: WordEncoder instance
        phrase_encoder: PhraseEncoder instance
        domain_hint: Domain identifier for fusion weight learning
        fusion_tracker: Optional PlasticityTracker for α,β,γ learning
        output_dim: Target output dimension
    """
    
    def __init__(
        self,
        char_encoder: CharacterEncoder,
        word_encoder: WordEncoder,
        phrase_encoder: PhraseEncoder,
        domain_hint: str = "general",
        fusion_tracker: Optional[PlasticityTracker] = None,
        output_dim: int = 128,
    ) -> None:
        self.char_encoder = char_encoder
        self.word_encoder = word_encoder
        self.phrase_encoder = phrase_encoder
        self.domain_hint = domain_hint
        self.output_dim = output_dim
        
        # Fusion weight tracker (α, β, γ per domain)
        self._fusion_tracker = fusion_tracker or PlasticityTracker(
            alpha=0.1,
            min_weight=0.1,
            max_regimes=5,
        )
        
        # Static fusion weights (fallback)
        self._static_weights = {"alpha": 0.2, "beta": 0.5, "gamma": 0.3}
        
        # Token-to-semantic vector cache
        self._vector_cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 1000
    
    def _get_fusion_weights(self) -> Tuple[float, float, float]:
        """Get current fusion weights (α, β, γ) for domain."""
        engines = ["char_level", "word_level", "phrase_level"]
        weights = self._fusion_tracker.get_weights(self.domain_hint, engines)
        
        if self._fusion_tracker.has_history(self.domain_hint):
            # Use learned weights, normalized to sum to 1
            total = sum(weights.values())
            if total > 0:
                alpha = weights["char_level"] / total
                beta = weights["word_level"] / total
                gamma = weights["phrase_level"] / total
                return (alpha, beta, gamma)
        
        # Fallback to static weights
        return (
            self._static_weights["alpha"],
            self._static_weights["beta"],
            self._static_weights["gamma"],
        )
    
    def _update_fusion_weights(
        self,
        char_quality: float,
        word_quality: float,
        phrase_quality: float,
    ) -> None:
        """Update fusion weights based on encoding quality feedback."""
        # Lower error = better quality = higher weight
        self._fusion_tracker.update(
            regime=self.domain_hint,
            engine_name="char_level",
            prediction_error=1.0 - char_quality,
        )
        self._fusion_tracker.update(
            regime=self.domain_hint,
            engine_name="word_level",
            prediction_error=1.0 - word_quality,
        )
        self._fusion_tracker.update(
            regime=self.domain_hint,
            engine_name="phrase_level",
            prediction_error=1.0 - phrase_quality,
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text to words."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9áéíóúüñ\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]
    
    def embed_token(self, token: str, context_words: List[str] = None) -> np.ndarray:
        """Embed single token using hybrid approach.
        
        Args:
            token: Token string to embed
            context_words: Optional surrounding words for phrase context
            
        Returns:
            Semantic vector [output_dim]
        """
        # Check cache
        cache_key = f"{token}:{self.domain_hint}"
        if cache_key in self._vector_cache:
            return self._vector_cache[cache_key]
        
        # 1. Character-level encoding
        char_vec = self.char_encoder.encode_token(token)
        
        # 2. Word-level encoding
        word_vec = self.word_encoder.encode_word(token)
        
        # 3. Phrase-level encoding (if context available)
        phrase_vec = np.zeros(self.phrase_encoder.output_dim, dtype=np.float32)
        if context_words and len(context_words) >= 2:
            # Find token position in context
            if token in context_words:
                idx = context_words.index(token)
                # Extract local phrase around token
                start = max(0, idx - 1)
                end = min(len(context_words), idx + 2)
                local_words = context_words[start:end]
                if len(local_words) >= 2:
                    phrase_vecs, phrases = self.phrase_encoder.encode_phrases(local_words)
                    if len(phrase_vecs) > 0:
                        phrase_vec = np.mean(phrase_vecs, axis=0)
        
        # 4. Project all to output_dim
        char_vec = self._project_vector(char_vec, self.output_dim)
        word_vec = self._project_vector(word_vec, self.output_dim)
        phrase_vec = self._project_vector(phrase_vec, self.output_dim)
        
        # 5. Fuse with learned weights
        alpha, beta, gamma = self._get_fusion_weights()
        hybrid_vec = alpha * char_vec + beta * word_vec + gamma * phrase_vec
        
        # Update cache
        if len(self._vector_cache) < self._cache_max_size:
            self._vector_cache[cache_key] = hybrid_vec
        
        return hybrid_vec.astype(np.float32)
    
    def _project_vector(self, vec: np.ndarray, target_dim: int) -> np.ndarray:
        """Project vector to target dimension."""
        if len(vec) == target_dim:
            return vec
        
        if len(vec) < target_dim:
            # Pad with zeros
            return np.pad(vec, (0, target_dim - len(vec)), mode='constant')
        else:
            # Truncate
            return vec[:target_dim]
    
    def embed_text(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """Embed full text to semantic vectors.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (token_vectors [n_tokens, output_dim], tokens)
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros((0, self.output_dim), dtype=np.float32), []
        
        # Encode each token with context
        vectors = []
        for i, token in enumerate(tokens):
            vec = self.embed_token(token, tokens)
            vectors.append(vec)
        
        # Update fusion weights based on vector quality
        if vectors:
            char_qual = np.mean([np.linalg.norm(v) for v in vectors])
            word_qual = self.word_encoder.get_stats().get("avg_importance", 0.5)
            phrase_qual = self.phrase_encoder.get_stats().get("avg_significance", 0.5)
            self._update_fusion_weights(char_qual, word_qual, phrase_qual)
        
        return np.array(vectors, dtype=np.float32), tokens
    
    def get_fusion_weights(self) -> Dict[str, float]:
        """Get current fusion weights."""
        alpha, beta, gamma = self._get_fusion_weights()
        return {"alpha": alpha, "beta": beta, "gamma": gamma}
    
    def get_stats(self) -> dict:
        """Get embedder statistics."""
        return {
            "domain": self.domain_hint,
            "output_dim": self.output_dim,
            "fusion_weights": self.get_fusion_weights(),
            "cache_size": len(self._vector_cache),
        }
