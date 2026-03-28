"""Character-level encoder for hybrid embeddings.

Encodes characters, bigrams, trigrams with feature flags.
Pure numpy, no external dependencies beyond entropy_filter.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np

from .entropy_filter import EntropyFilter


class CharacterEncoder:
    """Character-level encoder with n-gram support.
    
    Produces fixed-size character vectors per token:
    - ASCII values normalized [0, 1]
    - Bigram/trigram frequency encoding
    - Feature flags: digits, punctuation, uppercase
    
    Args:
        output_dim: Output vector dimension (default 32)
        max_chars: Maximum characters to process per token (default 20)
        entropy_filter: Optional entropy filter for token pre-filtering
    """
    
    def __init__(
        self,
        output_dim: int = 32,
        max_chars: int = 20,
        entropy_filter: EntropyFilter = None,
    ) -> None:
        self.output_dim = output_dim
        self.max_chars = max_chars
        self.entropy_filter = entropy_filter or EntropyFilter(threshold=0.3)
        
        # Character type feature indices
        self._digit_idx = output_dim - 4
        self._punct_idx = output_dim - 3
        self._upper_idx = output_dim - 2
        self._len_idx = output_dim - 1
    
    def _normalize_char(self, c: str) -> float:
        """Normalize single char to [0, 1] using ASCII/Unicode value."""
        if not c:
            return 0.0
        code = ord(c[0])
        # Normalize to [0, 1] using typical ASCII/Extended ASCII range
        return min(1.0, code / 255.0)
    
    def _compute_bigrams(self, text: str) -> Dict[str, int]:
        """Compute bigram frequencies."""
        bigrams = {}
        text_lower = text.lower()
        for i in range(len(text_lower) - 1):
            bg = text_lower[i:i+2]
            bigrams[bg] = bigrams.get(bg, 0) + 1
        return bigrams
    
    def _compute_trigrams(self, text: str) -> Dict[str, int]:
        """Compute trigram frequencies."""
        trigrams = {}
        text_lower = text.lower()
        for i in range(len(text_lower) - 2):
            tg = text_lower[i:i+3]
            trigrams[tg] = trigrams.get(tg, 0) + 1
        return trigrams
    
    def _extract_features(self, text: str) -> Tuple[float, float, float, float]:
        """Extract feature flags and normalized length.
        
        Returns:
            Tuple of (digit_ratio, punct_ratio, upper_ratio, norm_length)
        """
        if not text:
            return (0.0, 0.0, 0.0, 0.0)
        
        n = len(text)
        digits = sum(1 for c in text if c.isdigit())
        punct = sum(1 for c in text if not c.isalnum() and not c.isspace())
        upper = sum(1 for c in text if c.isupper())
        
        return (
            digits / n,
            punct / n,
            upper / n,
            min(1.0, n / self.max_chars),
        )
    
    def encode_token(self, token: str) -> np.ndarray:
        """Encode single token to character vector.
        
        Args:
            token: Input string token
            
        Returns:
            Character-level vector [output_dim]
        """
        vector = np.zeros(self.output_dim, dtype=np.float32)
        
        if not token:
            return vector
        
        # Truncate to max_chars
        text = token[:self.max_chars]
        
        # 1. Character-level encoding (first output_dim-4 positions)
        char_dim = self.output_dim - 4
        for i, c in enumerate(text):
            if i >= char_dim:
                break
            vector[i] = self._normalize_char(c)
        
        # 2. Feature flags (last 4 positions)
        digit_ratio, punct_ratio, upper_ratio, norm_len = self._extract_features(token)
        vector[self._digit_idx] = digit_ratio
        vector[self._punct_idx] = punct_ratio
        vector[self._upper_idx] = upper_ratio
        vector[self._len_idx] = norm_len
        
        # 3. Bigram/trigram augmentation (blended into char positions 8-15)
        if len(token) >= 2:
            bigrams = self._compute_bigrams(token)
            if bigrams:
                top_bg = max(bigrams.values())
                bg_score = min(1.0, len(bigrams) / max(1, len(token) - 1))
                # Blend into positions 8-11
                for i in range(4):
                    idx = 8 + i
                    if idx < char_dim:
                        vector[idx] = (vector[idx] + bg_score) / 2.0
        
        if len(token) >= 3:
            trigrams = self._compute_trigrams(token)
            if trigrams:
                tg_score = min(1.0, len(trigrams) / max(1, len(token) - 2))
                # Blend into positions 12-15
                for i in range(4):
                    idx = 12 + i
                    if idx < char_dim:
                        vector[idx] = (vector[idx] + tg_score) / 2.0
        
        return vector
    
    def encode_tokens(self, tokens: List[str]) -> np.ndarray:
        """Encode multiple tokens to matrix.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Matrix [n_tokens, output_dim]
        """
        if not tokens:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        
        # Apply entropy filter if configured
        if self.entropy_filter:
            filtered, _ = self.entropy_filter.filter_tokens(tokens)
            tokens = filtered if filtered else tokens
        
        return np.array([self.encode_token(t) for t in tokens], dtype=np.float32)
    
    def get_char_stats(self, token: str) -> dict:
        """Get character statistics for token."""
        return {
            "length": len(token),
            "entropy": self.entropy_filter.compute_entropy(token) if self.entropy_filter else 0.0,
            "bigrams": len(self._compute_bigrams(token)),
            "trigrams": len(self._compute_trigrams(token)),
            "features": self._extract_features(token),
        }
