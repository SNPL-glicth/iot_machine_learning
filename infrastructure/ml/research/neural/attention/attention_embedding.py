"""Lightweight sentence embeddings — TF-IDF weighted word vectors.

No external ML models (no transformers, torch, tensorflow).
Builds vocabulary from domain keywords and computes TF-IDF vectors.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple


class LightweightEmbedding:
    """TF-IDF sentence embedder using domain keyword vocabulary.
    
    Args:
        vocab: Dictionary of word -> index mapping
        d_model: Embedding dimension (vocab size)
    """
    
    def __init__(self, vocab: Dict[str, int], d_model: int) -> None:
        self.vocab = vocab
        self.d_model = d_model
        self._idf: Dict[str, float] = {}
        self._compute_idf()
    
    def _compute_idf(self) -> None:
        """Compute IDF weights (simplified: all terms equally important)."""
        # All terms have IDF = 1.0 (uniform weighting)
        # Could be enhanced with corpus statistics
        for word in self.vocab:
            self._idf[word] = 1.0
    
    def tokenize(self, text: str) -> List[str]:
        """Simple word tokenization (lowercase, alphanumeric)."""
        text = text.lower()
        # Keep alphanumeric and spaces, remove punctuation
        text = re.sub(r'[^a-z0-9áéíóúüñ\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]  # Filter short tokens
    
    def embed_sentence(self, sentence: str) -> List[float]:
        """Compute TF-IDF weighted embedding for single sentence.
        
        Args:
            sentence: Input sentence string
            
        Returns:
            Embedding vector [d_model] as list
        """
        tokens = self.tokenize(sentence)
        
        if not tokens:
            return [0.0] * self.d_model
        
        # Compute term frequencies
        tf: Dict[str, float] = {}
        for token in tokens:
            if token in self.vocab:
                tf[token] = tf.get(token, 0) + 1.0
        
        # Normalize TF
        max_tf = max(tf.values()) if tf else 1.0
        for token in tf:
            tf[token] /= max_tf
        
        # Build TF-IDF vector
        embedding = [0.0] * self.d_model
        for token, tf_val in tf.items():
            idx = self.vocab[token]
            idf_val = self._idf.get(token, 1.0)
            embedding[idx] = tf_val * idf_val
        
        # L2 normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def embed_sentences(
        self, 
        sentences: List[str], 
        target_dim: int = 64
    ) -> List[List[float]]:
        """Embed multiple sentences, optionally projecting to target dimension.
        
        Args:
            sentences: List of sentence strings
            target_dim: Target embedding dimension (for projection)
            
        Returns:
            List of embedding vectors [n_sentences, target_dim]
        """
        raw_embeddings = [self.embed_sentence(s) for s in sentences]
        
        # Project to target dimension if needed
        if self.d_model != target_dim:
            return self._project_embeddings(raw_embeddings, target_dim)
        
        return raw_embeddings
    
    def _project_embeddings(
        self, 
        embeddings: List[List[float]], 
        target_dim: int
    ) -> List[List[float]]:
        """Project embeddings to lower dimension via random projection.
        
        Uses random Gaussian projection matrix.
        """
        import random
        random.seed(42)  # Reproducible
        
        # Generate random projection matrix [d_model, target_dim]
        proj = [[random.gauss(0, 0.1) for _ in range(target_dim)] 
                for _ in range(self.d_model)]
        
        projected = []
        for emb in embeddings:
            # Matrix multiply: emb [1, d_model] @ proj [d_model, target_dim]
            new_emb = [0.0] * target_dim
            for j in range(target_dim):
                for i in range(self.d_model):
                    new_emb[j] += emb[i] * proj[i][j]
            projected.append(new_emb)
        
        return projected
    
    @classmethod
    def from_keywords(
        cls, 
        keywords: List[str], 
        d_model: int = 64
    ) -> "LightweightEmbedding":
        """Factory: build embedder from keyword list.
        
        Args:
            keywords: List of domain keywords
            d_model: Target dimension (will use min(len(keywords), d_model))
            
        Returns:
            Configured LightweightEmbedding instance
        """
        # Take unique keywords up to d_model
        unique_keywords = list(dict.fromkeys(keywords))[:d_model]
        
        vocab = {kw: i for i, kw in enumerate(unique_keywords)}
        actual_dim = len(vocab)
        
        return cls(vocab=vocab, d_model=actual_dim)
