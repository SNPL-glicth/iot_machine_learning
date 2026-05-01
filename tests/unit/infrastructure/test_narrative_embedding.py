"""Tests for NarrativeEmbeddingNetwork + EmbeddingNarrativeGenerator."""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.narrative.embedding_network import (
    NarrativeEmbeddingNetwork,
)
from iot_machine_learning.infrastructure.ml.cognitive.narrative.phrase_bank import (
    get_phrase_bank,
    PhraseEntry,
)
from iot_machine_learning.infrastructure.ml.cognitive.narrative.generator import (
    EmbeddingNarrativeGenerator,
    _cosine_similarity,
    _GENERIC_FALLBACK,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-9)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0] * 8, [1.0] * 8) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _cosine_similarity([1.0] * 4, [1.0] * 8)


class TestNarrativeEmbeddingNetwork:
    def test_forward_shape(self):
        net = NarrativeEmbeddingNetwork(seed=42)
        out = net.embed([0.5] * 18)
        assert len(out) == 8
        assert all(isinstance(v, float) for v in out)
        assert all(v >= 0.0 for v in out)  # ReLU

    def test_different_inputs_different_outputs(self):
        net = NarrativeEmbeddingNetwork(seed=42)
        out1 = net.embed([0.0] * 18)
        out2 = net.embed([1.0] * 18)
        assert out1 != out2

    def test_batch_forward(self):
        import numpy as np
        net = NarrativeEmbeddingNetwork(seed=42)
        batch = np.array([[0.5] * 18, [0.3] * 18])
        out = net.forward(batch)
        assert out.shape == (2, 8)
        assert np.all(out >= 0)


class TestPhraseBank:
    def test_seed_set_not_empty(self):
        bank = get_phrase_bank()
        assert len(bank) >= 25
        for entry in bank:
            assert len(entry.text) > 0
            assert len(entry.target) == 8
            assert all(0.0 <= v <= 1.0 for v in entry.target)


class TestEmbeddingNarrativeGenerator:
    def test_generates_text_for_critical_vector(self):
        gen = EmbeddingNarrativeGenerator()
        # Vector emphasizing criticality (dim 0), anomaly (dim 5), confidence (dim 6)
        vec = [0.0] * 18
        vec[0] = 0.9   # high regime slope (mapped)
        vec[5] = 0.9   # high anomaly score
        vec[6] = 0.9   # high confidence
        text = gen.generate(vec)
        assert isinstance(text, str)
        assert len(text) > 0
        assert text != _GENERIC_FALLBACK

    def test_fallback_on_zero_vector(self):
        gen = EmbeddingNarrativeGenerator()
        text = gen.generate([0.0] * 18)
        # Zero vector → all embeddings zero → no cosine similarity > threshold
        assert text == _GENERIC_FALLBACK

    def test_invalid_dimension_raises(self):
        gen = EmbeddingNarrativeGenerator()
        with pytest.raises(ValueError):
            gen.generate([0.5] * 10)

    def test_reproducible_seed(self):
        gen1 = EmbeddingNarrativeGenerator(network=NarrativeEmbeddingNetwork(seed=123))
        gen2 = EmbeddingNarrativeGenerator(network=NarrativeEmbeddingNetwork(seed=123))
        vec = [0.5] * 18
        assert gen1.generate(vec) == gen2.generate(vec)

    def test_max_phrases_respected(self):
        bank = [
            PhraseEntry("A", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            PhraseEntry("B", [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            PhraseEntry("C", [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            PhraseEntry("D", [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        gen = EmbeddingNarrativeGenerator(
            phrase_bank=bank,
            max_phrases=2,
            similarity_threshold=0.0,
        )
        text = gen.generate([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 10)
        parts = text.split(" | ")
        assert len(parts) <= 2

    def test_custom_fallback(self):
        gen = EmbeddingNarrativeGenerator(
            fallback="Custom fallback message.",
        )
        text = gen.generate([0.0] * 18)
        assert text == "Custom fallback message."
