"""Positional encodings for attention — sinusoidal, no learnable parameters.

Pure numpy implementation following the original Transformer paper
("Attention Is All You Need", Vaswani et al., 2017).
Position encodings allow the model to understand word order.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class PositionalEncoder:
    """Sinusoidal positional encodings.
    
    Precomputes position vectors using sine/cosine waves of different
    frequencies. This allows the model to learn relative positions.
    
    Args:
        d_model: Dimension of embeddings (must be even for sin/cos pairs)
        max_len: Maximum sequence length to precompute
    """
    
    def __init__(self, d_model: int = 64, max_len: int = 100) -> None:
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        
        self.d_model = d_model
        self.max_len = max_len
        self._encoding_matrix: Optional[np.ndarray] = None
        self._precompute()
    
    def _precompute(self) -> None:
        """Precompute position encoding matrix [max_len, d_model]."""
        position = np.arange(self.max_len)[:, np.newaxis]  # [max_len, 1]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * 
            -(np.log(10000.0) / self.d_model)
        )  # [d_model/2]
        
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)  # Even indices: sine
        pe[:, 1::2] = np.cos(position * div_term)  # Odd indices: cosine
        
        self._encoding_matrix = pe
    
    def encode(self, seq_len: int) -> np.ndarray:
        """Get positional encodings for sequence of given length.
        
        Args:
            seq_len: Length of sequence (must be <= max_len)
            
        Returns:
            Position encoding matrix [seq_len, d_model]
        """
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_len {self.max_len}"
            )
        
        return self._encoding_matrix[:seq_len, :].copy()
    
    def add_to_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Add positional encodings to input embeddings.
        
        Args:
            embeddings: Input embeddings [seq_len, d_model]
            
        Returns:
            Embeddings with position info added [seq_len, d_model]
        """
        seq_len, d_model = embeddings.shape
        if d_model != self.d_model:
            raise ValueError(
                f"d_model mismatch: encoder has {self.d_model}, "
                f"embeddings have {d_model}"
            )
        
        pos_enc = self.encode(seq_len)
        return embeddings + pos_enc
    
    def get_relative_weights(self, pos_i: int, pos_j: int) -> float:
        """Compute attention weight bias for relative positions.
        
        In self-attention, positions i and j can attend to each other
        with a learned bias proportional to their distance.
        
        Args:
            pos_i: Position of query
            pos_j: Position of key
            
        Returns:
            Relative position weight (higher for closer positions)
        """
        distance = abs(pos_i - pos_j)
        return np.exp(-distance / 10.0)  # Exponential decay with distance
