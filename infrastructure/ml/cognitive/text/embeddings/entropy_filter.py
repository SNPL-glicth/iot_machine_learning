"""Shannon entropy filter for token information content.

Filters low-information tokens using entropy thresholding.
Pure numpy, no external dependencies.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


class EntropyFilter:
    """Filters tokens by Shannon entropy H(w) = -Σ p(c) log p(c).
    
    High entropy = more information content = keep.
    Low entropy = repetitive/predictable = discard.
    
    Args:
        threshold: Minimum entropy [0, 1] to retain token (default 0.5)
        min_token_len: Minimum token length to evaluate (default 3)
    """
    
    def __init__(self, threshold: float = 0.5, min_token_len: int = 3) -> None:
        self.threshold = threshold
        self.min_token_len = min_token_len
    
    def compute_entropy(self, token: str) -> float:
        """Compute normalized Shannon entropy of token characters.
        
        Args:
            token: Input string token
            
        Returns:
            Normalized entropy [0, 1] where 1 = maximum randomness
        """
        if len(token) < self.min_token_len:
            return 0.0
        
        # Character frequency counts
        char_counts = {}
        for c in token.lower():
            char_counts[c] = char_counts.get(c, 0) + 1
        
        n = len(token)
        
        # Shannon entropy: H = -Σ p(c) * log2(p(c))
        entropy = 0.0
        for count in char_counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normalize by max possible entropy (log2 of unique char count)
        max_entropy = math.log2(len(char_counts))
        if max_entropy == 0:
            return 0.0
        
        normalized = entropy / max_entropy
        return min(1.0, max(0.0, normalized))
    
    def filter_tokens(self, tokens: List[str]) -> Tuple[List[str], List[float]]:
        """Filter tokens by entropy threshold.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Tuple of (retained_tokens, entropy_scores)
        """
        retained = []
        scores = []
        
        for token in tokens:
            entropy = self.compute_entropy(token)
            scores.append(entropy)
            
            if entropy >= self.threshold:
                retained.append(token)
        
        return retained, scores
    
    def batch_filter(self, token_lists: List[List[str]]) -> List[Tuple[List[str], List[float]]]:
        """Filter multiple token lists efficiently.
        
        Args:
            token_lists: List of token lists
            
        Returns:
            List of (retained_tokens, scores) tuples
        """
        return [self.filter_tokens(tokens) for tokens in token_lists]
    
    def get_stats(self, tokens: List[str]) -> dict:
        """Compute entropy statistics for token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dict with mean, min, max entropy and filter ratio
        """
        if not tokens:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "filter_ratio": 0.0}
        
        scores = [self.compute_entropy(t) for t in tokens]
        retained = sum(1 for s in scores if s >= self.threshold)
        
        return {
            "mean": float(np.mean(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "filter_ratio": 1.0 - (retained / len(tokens)) if tokens else 0.0,
        }
