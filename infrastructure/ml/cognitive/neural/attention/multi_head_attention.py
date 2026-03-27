"""Multi-Head Attention — pure numpy implementation (no torch/tensorflow)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


class MultiHeadAttention:
    """Multi-head scaled dot-product attention (numpy only)."""
    
    def __init__(self, n_heads: int = 4, d_model: int = 64, dropout: float = 0.0) -> None:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads, self.d_model, self.d_k = n_heads, d_model, d_model // n_heads
        self.dropout = dropout
        import random
        random.seed(42)
        self.W_q = self._init_weights(d_model, d_model)
        self.W_k = self._init_weights(d_model, d_model)
        self.W_v = self._init_weights(d_model, d_model)
        self.W_o = self._init_weights(d_model, d_model)
        self._last_attention: Optional[List] = None
    
    def _init_weights(self, rows: int, cols: int) -> List:
        import random
        scale = math.sqrt(2.0 / (rows + cols))
        return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]
    
    def _matmul(self, A: List, B: List) -> List:
        rows_a, cols_a = len(A), len(A[0]) if A else 0
        cols_b = len(B[0]) if B else 0
        return [[sum(A[i][k] * B[k][j] for k in range(cols_a)) for j in range(cols_b)] for i in range(rows_a)]
    
    def _transpose(self, A: List) -> List:
        if not A:
            return []
        return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]
    
    def _scaled_dot_product(self, Q: List, K: List, V: List, mask: Optional[List] = None) -> Tuple[List, List]:
        seq_len, d_k = len(Q), len(Q[0]) if Q else 0
        scores = self._matmul(Q, self._transpose(K))
        scale = math.sqrt(d_k) if d_k > 0 else 1.0
        scores = [[s / scale for s in row] for row in scores]
        if mask is not None:
            scores = [[scores[i][j] if mask[i][j] > 0 else -1e9 for j in range(seq_len)] for i in range(seq_len)]
        attention = self._softmax(scores)
        return self._matmul(attention, V), attention
    
    def _softmax(self, x: List) -> List:
        result = []
        for row in x:
            max_val = max(row) if row else 0.0
            exp_row = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_row)
            result.append([v / sum_exp for v in exp_row] if sum_exp > 0 else [0.0] * len(row))
        return result
    
    def _split_heads(self, x: List, batch_size: int, seq_len: int) -> List:
        reshaped = [[x[i * seq_len + j] for j in range(seq_len)] for i in range(batch_size)]
        heads = []
        for b in range(batch_size):
            batch_heads = [[] for _ in range(self.n_heads)]
            for s in range(seq_len):
                for h in range(self.n_heads):
                    start, end = h * self.d_k, (h + 1) * self.d_k
                    batch_heads[h].append(reshaped[b][s][start:end])
            heads.append(batch_heads)
        return heads
    
    def forward(self, Q: List, K: List, V: List, mask: Optional[List] = None) -> List:
        seq_len = len(Q)
        Q_proj = self._matmul(Q, self.W_q)
        K_proj = self._matmul(K, self.W_k)
        V_proj = self._matmul(V, self.W_v)
        Q_heads = self._split_heads(Q_proj, 1, seq_len)[0]
        K_heads = self._split_heads(K_proj, 1, seq_len)[0]
        V_heads = self._split_heads(V_proj, 1, seq_len)[0]
        head_outputs, head_attentions = [], []
        for h in range(self.n_heads):
            out, attn = self._scaled_dot_product(Q_heads[h], K_heads[h], V_heads[h], mask)
            head_outputs.append(out)
            head_attentions.append(attn)
        self._last_attention = head_attentions
        concatenated = []
        for s in range(seq_len):
            concat_row = []
            for h in range(self.n_heads):
                concat_row.extend(head_outputs[h][s])
            concatenated.append(concat_row)
        return self._matmul(concatenated, self.W_o)
    
    def get_attention_weights(self) -> Optional[List]:
        return self._last_attention
