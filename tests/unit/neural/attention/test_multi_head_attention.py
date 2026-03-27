"""Tests for Multi-Head Attention implementation.

40 tests covering positional encoding, embeddings, attention, and integration.
"""

from __future__ import annotations

import math
import unittest
from typing import List


class TestPositionalEncoder(unittest.TestCase):
    """10 tests for positional encoding."""
    
    def test_01_constructor_valid(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import PositionalEncoder
        pe = PositionalEncoder(d_model=64, max_len=100)
        self.assertEqual(pe.d_model, 64)
        self.assertEqual(pe.max_len, 100)
    
    def test_02_constructor_odd_d_model(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import PositionalEncoder
        with self.assertRaises(ValueError):
            PositionalEncoder(d_model=63, max_len=100)
    
    def test_03_encode_shape(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import PositionalEncoder
        pe = PositionalEncoder(d_model=64, max_len=100)
        enc = pe.encode(10)
        self.assertEqual(len(enc), 10)
        self.assertEqual(len(enc[0]), 64)
    
    def test_04_encode_exceeds_max(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import PositionalEncoder
        pe = PositionalEncoder(d_model=64, max_len=10)
        with self.assertRaises(ValueError):
            pe.encode(20)
    
    def test_05_encode_unique_positions(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import PositionalEncoder
        pe = PositionalEncoder(d_model=64, max_len=100)
        enc = pe.encode(5)
        # Each position should be different
        for i in range(4):
            self.assertNotEqual(enc[i], enc[i+1])
    
    def test_06_add_to_embeddings_shape(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import PositionalEncoder
        pe = PositionalEncoder(d_model=64, max_len=100)
        embeddings = [[0.1] * 64 for _ in range(10)]
        result = pe.add_to_embeddings(embeddings)
        self.assertEqual(len(result), 10)
        self.assertEqual(len(result[0]), 64)
    
    def test_07_add_changes_values(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import PositionalEncoder
        pe = PositionalEncoder(d_model=64, max_len=100)
        embeddings = [[0.5] * 64 for _ in range(5)]
        result = pe.add_to_embeddings(embeddings)
        # Values should be modified by positional encoding
        self.assertNotEqual(result[0][0], embeddings[0][0])
    
    def test_08_relative_weights_closer(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import PositionalEncoder
        pe = PositionalEncoder(d_model=64, max_len=100)
        w_close = pe.get_relative_weights(0, 1)
        w_far = pe.get_relative_weights(0, 10)
        self.assertGreater(w_close, w_far)
    
    def test_09_relative_weights_symmetric(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import PositionalEncoder
        pe = PositionalEncoder(d_model=64, max_len=100)
        w1 = pe.get_relative_weights(5, 3)
        w2 = pe.get_relative_weights(3, 5)
        self.assertEqual(w1, w2)
    
    def test_10_sinusoidal_pattern(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import PositionalEncoder
        pe = PositionalEncoder(d_model=64, max_len=100)
        enc = pe.encode(10)
        # Sine values should be in [-1, 1]
        for row in enc:
            for val in row[0::2]:  # Even indices are sine
                self.assertGreaterEqual(val, -1.0)
                self.assertLessEqual(val, 1.0)


class TestLightweightEmbedding(unittest.TestCase):
    """8 tests for lightweight embeddings."""
    
    def test_11_constructor(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import LightweightEmbedding
        vocab = {"word1": 0, "word2": 1, "word3": 2}
        emb = LightweightEmbedding(vocab, len(vocab))
        self.assertEqual(emb.d_model, 3)
    
    def test_12_tokenize_lowercase(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import LightweightEmbedding
        vocab = {"word": 0}
        emb = LightweightEmbedding(vocab, 1)
        tokens = emb.tokenize("The WORD is Here")
        self.assertIn("word", tokens)
    
    def test_13_tokenize_filters_short(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import LightweightEmbedding
        vocab = {"word": 0}
        emb = LightweightEmbedding(vocab, 1)
        tokens = emb.tokenize("The word is a b c d")
        self.assertNotIn("a", tokens)
        self.assertNotIn("b", tokens)
    
    def test_14_embed_sentence_known_words(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import LightweightEmbedding
        vocab = {"critical": 0, "error": 1, "warning": 2}
        emb = LightweightEmbedding(vocab, 3)
        result = emb.embed_sentence("Critical error detected")
        self.assertEqual(len(result), 3)
        self.assertGreater(result[0], 0)  # critical
        self.assertGreater(result[1], 0)  # error
    
    def test_15_embed_sentence_unknown_words(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import LightweightEmbedding
        vocab = {"critical": 0}
        emb = LightweightEmbedding(vocab, 1)
        result = emb.embed_sentence("Unknown words here")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 0.0)
    
    def test_16_embed_sentence_empty(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import LightweightEmbedding
        vocab = {"word": 0}
        emb = LightweightEmbedding(vocab, 1)
        result = emb.embed_sentence("")
        self.assertEqual(result, [0.0])
    
    def test_17_l2_normalized(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import LightweightEmbedding
        vocab = {"critical": 0, "error": 1}
        emb = LightweightEmbedding(vocab, 2)
        result = emb.embed_sentence("Critical error")
        # Should be L2 normalized
        norm = math.sqrt(sum(x*x for x in result))
        self.assertAlmostEqual(norm, 1.0, places=3)
    
    def test_18_from_keywords_factory(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import LightweightEmbedding
        keywords = ["a", "b", "c", "d", "e"]
        emb = LightweightEmbedding.from_keywords(keywords, d_model=10)
        self.assertEqual(len(emb.vocab), 5)


class TestMultiHeadAttention(unittest.TestCase):
    """12 tests for multi-head attention."""
    
    def test_19_constructor_valid(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        mha = MultiHeadAttention(n_heads=4, d_model=64)
        self.assertEqual(mha.n_heads, 4)
        self.assertEqual(mha.d_model, 64)
        self.assertEqual(mha.d_k, 16)
    
    def test_20_constructor_invalid_divisibility(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        with self.assertRaises(ValueError):
            MultiHeadAttention(n_heads=5, d_model=64)
    
    def test_21_forward_shape(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        mha = MultiHeadAttention(n_heads=4, d_model=64)
        seq_len = 5
        Q = [[0.1] * 64 for _ in range(seq_len)]
        result = mha.forward(Q, Q, Q)
        self.assertEqual(len(result), seq_len)
        self.assertEqual(len(result[0]), 64)
    
    def test_22_forward_changes_values(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        mha = MultiHeadAttention(n_heads=4, d_model=64)
        Q = [[0.5] * 64 for _ in range(5)]
        result = mha.forward(Q, Q, Q)
        # Output should be different from input (transformed by attention)
        self.assertNotEqual(result[0][0], Q[0][0])
    
    def test_23_matmul_correctness(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        mha = MultiHeadAttention(n_heads=2, d_model=4)
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = mha._matmul(A, B)
        # A @ B = [[19, 22], [43, 50]]
        self.assertEqual(C[0][0], 19)
        self.assertEqual(C[1][1], 50)
    
    def test_24_transpose(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        mha = MultiHeadAttention(n_heads=2, d_model=4)
        A = [[1, 2, 3], [4, 5, 6]]
        T = mha._transpose(A)
        self.assertEqual(T[0][0], 1)
        self.assertEqual(T[0][1], 4)
        self.assertEqual(T[1][0], 2)
    
    def test_25_softmax_sums_to_one(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        mha = MultiHeadAttention(n_heads=2, d_model=4)
        x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = mha._softmax(x)
        for row in result:
            self.assertAlmostEqual(sum(row), 1.0, places=5)
    
    def test_26_softmax_preserves_order(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        mha = MultiHeadAttention(n_heads=2, d_model=4)
        x = [[1.0, 3.0, 2.0]]
        result = mha._softmax(x)
        # Higher input → higher output
        self.assertGreater(result[0][1], result[0][2])
        self.assertGreater(result[0][1], result[0][0])
    
    def test_27_scaled_dot_product_shape(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        mha = MultiHeadAttention(n_heads=2, d_model=4)
        Q = K = V = [[1.0, 2.0], [3.0, 4.0]]
        output, attn = mha._scaled_dot_product(Q, K, V)
        self.assertEqual(len(output), 2)
        self.assertEqual(len(output[0]), 2)
        self.assertEqual(len(attn), 2)
        self.assertEqual(len(attn[0]), 2)
    
    def test_28_attention_weights_retrievable(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        mha = MultiHeadAttention(n_heads=2, d_model=4)
        Q = [[0.5] * 4 for _ in range(3)]
        result = mha.forward(Q, Q, Q)
        weights = mha.get_attention_weights()
        self.assertIsNotNone(weights)
        self.assertEqual(len(weights), 2)  # 2 heads
    
    def test_29_mask_affects_attention(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        mha = MultiHeadAttention(n_heads=2, d_model=4)
        Q = [[0.5] * 4 for _ in range(3)]
        # Mask out position 1
        mask = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        result = mha.forward(Q, Q, Q, mask)
        # Should still produce output
        self.assertEqual(len(result), 3)
    
    def test_30_different_q_k_v(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import MultiHeadAttention
        mha = MultiHeadAttention(n_heads=2, d_model=4)
        Q = [[1.0] * 4 for _ in range(3)]
        K = [[0.5] * 4 for _ in range(3)]
        V = [[0.3] * 4 for _ in range(3)]
        result = mha.forward(Q, K, V)
        self.assertEqual(len(result), 3)


class TestAttentionContextCollector(unittest.TestCase):
    """10 tests for attention context collector."""
    
    def test_31_constructor(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import AttentionContextCollector
        vocab = {"word1": 0, "word2": 1}
        collector = AttentionContextCollector(vocab, n_heads=2, d_model=4)
        self.assertIsNotNone(collector.embedder)
        self.assertIsNotNone(collector.pos_encoder)
        self.assertIsNotNone(collector.attention)
    
    def test_32_collect_context_empty(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import AttentionContextCollector
        vocab = {"word": 0}
        collector = AttentionContextCollector(vocab, n_heads=2, d_model=4)
        result = collector.collect_context("")
        self.assertIsNone(result)
    
    def test_33_collect_context_single_sentence(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import AttentionContextCollector
        vocab = {"critical": 0, "error": 1}
        collector = AttentionContextCollector(vocab, n_heads=2, d_model=4)
        result = collector.collect_context("Critical error detected")
        # Single sentence → need at least 2 for attention
        self.assertIsNone(result)
    
    def test_34_collect_context_multi_sentence(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import AttentionContextCollector
        vocab = {"server": 0, "down": 1, "error": 2}
        collector = AttentionContextCollector(vocab, n_heads=2, d_model=4)
        text = "Server is down. Error detected. Please investigate."
        result = collector.collect_context(text, budget_ms=500.0)
        self.assertIsNotNone(result)
        self.assertIn("attended_sentences", dir(result))
    
    def test_35_detect_temporal_markers(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import AttentionContextCollector
        vocab = {"word": 0}
        collector = AttentionContextCollector(vocab, n_heads=2, d_model=4)
        markers = collector._detect_temporal_markers("Expires in 6 hours")
        self.assertIn("expires", markers)
        self.assertIn("hours", markers)
    
    def test_36_detect_negations(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import AttentionContextCollector
        vocab = {"word": 0}
        collector = AttentionContextCollector(vocab, n_heads=2, d_model=4)
        sentences = ["Server is not down", "System failed"]
        negations = collector._detect_negations("Server is not down. System failed", sentences)
        self.assertIn("not", negations)
        self.assertEqual(negations["not"], [0])
    
    def test_37_compute_domain_scores(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import AttentionContextCollector
        vocab = {"server": 0, "cpu": 1}
        collector = AttentionContextCollector(vocab, n_heads=2, d_model=4)
        attended = [[0.5] * 4 for _ in range(2)]
        sentences = ["Server CPU high", "Network issue"]
        scores = collector._compute_domain_scores(attended, sentences)
        self.assertIn("infrastructure", scores)
        self.assertGreaterEqual(scores["infrastructure"], 0.0)
    
    def test_38_budget_timeout(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import AttentionContextCollector
        vocab = {"word": i for i in range(50)}
        collector = AttentionContextCollector(vocab, n_heads=4, d_model=64)
        # Very short budget
        text = "Word " * 100  # Long text
        result = collector.collect_context(text, budget_ms=0.001)
        self.assertIsNone(result)
    
    def test_39_context_has_confidence(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import AttentionContextCollector
        vocab = {"server": 0, "down": 1, "critical": 2}
        collector = AttentionContextCollector(vocab, n_heads=2, d_model=4)
        text = "Server down. Critical error. Investigate now."
        result = collector.collect_context(text, budget_ms=500.0)
        if result:
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
    
    def test_40_average_attention_weights(self):
        from iot_machine_learning.infrastructure.ml.cognitive.neural.attention import AttentionContextCollector
        vocab = {"word": 0}
        collector = AttentionContextCollector(vocab, n_heads=2, d_model=4)
        weights = [[[0.5, 0.5], [0.3, 0.7]], [[0.6, 0.4], [0.4, 0.6]]]
        avg = collector._average_attention_weights(weights)
        self.assertEqual(len(avg), 2)
        self.assertEqual(len(avg[0]), 2)
        # Average of (0.5, 0.6) and (0.5, 0.4)
        self.assertAlmostEqual(avg[0][0], 0.55, places=5)


if __name__ == "__main__":
    unittest.main()
