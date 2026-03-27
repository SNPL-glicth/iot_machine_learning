"""Attention context collector — bridge from text to attended vectors.

Integrates embeddings + positional encoding + multi-head attention
to produce context-aware document representations.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .positional_encoding import PositionalEncoder
from .attention_embedding import LightweightEmbedding
from .multi_head_attention import MultiHeadAttention


@dataclass(frozen=True)
class AttentionContext:
    """Output of attention-based context analysis.
    
    Attributes:
        attended_sentences: List of sentences with attention weights
        temporal_markers: Detected urgency markers with scores
        negation_context: Map of negations found and their scope
        multi_domain_scores: Confidence per domain (not just winner)
        attention_weights: Raw attention matrix for debugging
        confidence: Overall confidence in attention analysis [0, 1]
    """
    attended_sentences: List[str]
    temporal_markers: Dict[str, float]
    negation_context: Dict[str, List[int]]
    multi_domain_scores: Dict[str, float]
    attention_weights: List[List[float]]
    confidence: float


class AttentionContextCollector:
    """Collect contextual understanding via multi-head attention.
    
    Pipeline:
        1. Sentence tokenization
        2. TF-IDF embeddings
        3. Positional encoding
        4. Multi-head self-attention
        5. Extract enhanced features
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        n_heads: int = 4,
        d_model: int = 64,
        max_seq_len: int = 100,
    ) -> None:
        self.embedder = LightweightEmbedding(vocab, len(vocab))
        self.pos_encoder = PositionalEncoder(d_model, max_seq_len)
        self.attention = MultiHeadAttention(n_heads, d_model)
        self.d_model = d_model
        
        # Temporal markers for urgency detection
        self.temporal_keywords = {
            "expires": 1.0, "expira": 1.0, "vence": 0.9,
            "hours": 0.8, "horas": 0.8, "minutes": 0.9, "minutos": 0.9,
            "days": 0.5, "días": 0.5, "weeks": 0.3, "semanas": 0.3,
            "months": 0.2, "meses": 0.2, "immediately": 1.0, "inmediatamente": 1.0,
        }
        
        # Negation markers
        self.negation_words = {"not", "no", "sin", "never", "nunca", "nadie"}
    
    def collect_context(self, text: str, budget_ms: float = 100.0) -> Optional[AttentionContext]:
        """Run attention pipeline on text with time budget.
        
        Args:
            text: Input document text
            budget_ms: Maximum time allowed (graceful fallback if exceeded)
            
        Returns:
            AttentionContext or None if timeout/error
        """
        start = time.monotonic()
        
        try:
            # 1. Sentence splitting
            sentences = self._split_sentences(text)
            if len(sentences) < 2:
                return None  # Need at least 2 sentences for attention
            
            # 2. Generate embeddings
            embeddings_list = self.embedder.embed_sentences(sentences, self.d_model)
            
            # Check budget
            if (time.monotonic() - start) * 1000 > budget_ms * 0.5:
                return None
            
            # 3. Add positional encoding
            seq_len = len(sentences)
            pos_enc = self.pos_encoder.encode(seq_len)
            
            # Convert to format for attention (add pos encoding)
            Q = [[embeddings_list[i][j] + pos_enc[i][j] 
                  for j in range(min(self.d_model, len(embeddings_list[i])))]
                 for i in range(seq_len)]
            
            # Ensure correct dimensionality
            Q = [row + [0.0] * (self.d_model - len(row)) if len(row) < self.d_model else row[:self.d_model] 
                 for row in Q]
            
            # 4. Self-attention (Q=K=V for self-attention)
            attended = self.attention.forward(Q, Q, Q)
            
            # Check budget
            elapsed_ms = (time.monotonic() - start) * 1000
            if elapsed_ms > budget_ms:
                return None
            
            # 5. Extract features
            temporal_markers = self._detect_temporal_markers(text)
            negations = self._detect_negations(text, sentences)
            domain_scores = self._compute_domain_scores(attended, sentences)
            
            # Get attention weights from last forward pass
            attn_weights = self.attention.get_attention_weights()
            avg_attention = self._average_attention_weights(attn_weights) if attn_weights else [[0.0]]
            
            # Compute confidence based on attention entropy
            confidence = self._compute_confidence(avg_attention)
            
            return AttentionContext(
                attended_sentences=sentences,
                temporal_markers=temporal_markers,
                negation_context=negations,
                multi_domain_scores=domain_scores,
                attention_weights=avg_attention,
                confidence=confidence,
            )
            
        except Exception:
            return None  # Graceful fallback
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on punctuation
        sentences = re.split(r'[.!?\n]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10][:50]  # Limit to 50 sentences
    
    def _detect_temporal_markers(self, text: str) -> Dict[str, float]:
        """Detect temporal urgency markers."""
        text_lower = text.lower()
        markers = {}
        for word, score in self.temporal_keywords.items():
            if word in text_lower:
                markers[word] = score
        return markers
    
    def _detect_negations(self, text: str, sentences: List[str]) -> Dict[str, List[int]]:
        """Detect negation words and which sentences they affect."""
        text_lower = text.lower()
        negations = {}
        for neg in self.negation_words:
            indices = [i for i, sent in enumerate(sentences) if neg in sent.lower()]
            if indices:
                negations[neg] = indices
        return negations
    
    def _compute_domain_scores(self, attended: List, sentences: List[str]) -> Dict[str, float]:
        """Compute domain confidence scores from attended vectors."""
        # Simple: use sentence presence to estimate domain distribution
        # Could be enhanced with learned classifiers
        infra_score = sum(1 for s in sentences if any(kw in s.lower() for kw in 
            ["server", "servidor", "cpu", "memory", "network", "red", "database"])) / max(len(sentences), 1)
        sec_score = sum(1 for s in sentences if any(kw in s.lower() for kw in 
            ["breach", "vulnerability", "attack", "malware", "security", "unauthorized"])) / max(len(sentences), 1)
        biz_score = sum(1 for s in sentences if any(kw in s.lower() for kw in 
            ["revenue", "cost", "budget", "sla", "contract", "cliente", "business"])) / max(len(sentences), 1)
        
        return {
            "infrastructure": min(1.0, infra_score * 2),  # Amplify signal
            "security": min(1.0, sec_score * 2),
            "business": min(1.0, biz_score * 2),
        }
    
    def _average_attention_weights(self, weights: List) -> List[List[float]]:
        """Average attention weights across heads."""
        if not weights:
            return [[0.0]]
        # weights is list of heads, each [seq_len, seq_len]
        n_heads = len(weights)
        seq_len = len(weights[0])
        averaged = [[0.0] * seq_len for _ in range(seq_len)]
        for h in range(n_heads):
            for i in range(seq_len):
                for j in range(seq_len):
                    averaged[i][j] += weights[h][i][j] / n_heads
        return averaged
    
    def _compute_confidence(self, attention: List[List[float]]) -> float:
        """Compute confidence from attention entropy."""
        if not attention or not attention[0]:
            return 0.5
        # High entropy = diffused attention = low confidence
        # Low entropy = focused attention = high confidence
        entropies = []
        for row in attention:
            entropy = sum(-p * (p > 0 and p < 1 and __import__('math').log(p + 1e-10) or 0) for p in row)
            entropies.append(entropy)
        avg_entropy = sum(entropies) / len(entropies)
        # Map entropy to confidence (lower entropy = higher confidence)
        return max(0.3, min(0.9, 1.0 - avg_entropy / 2.0))
