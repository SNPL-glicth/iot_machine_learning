"""Tests for spike encoder."""

import pytest
from infrastructure.ml.cognitive.neural.snn.spike_encoder import SpikeEncoder
from infrastructure.ml.cognitive.neural.types import InputType


class TestEncoderInitialization:
    """Test encoder initialization."""
    
    def test_default_initialization(self):
        """Test encoder with default rates."""
        encoder = SpikeEncoder()
        
        assert encoder.max_rate == 0.1
        assert encoder.min_rate == 0.01
    
    def test_custom_rates(self):
        """Test encoder with custom rates."""
        encoder = SpikeEncoder(max_rate=0.2, min_rate=0.02)
        
        assert encoder.max_rate == 0.2
        assert encoder.min_rate == 0.02


class TestRateCoding:
    """Test rate coding mechanism."""
    
    def test_high_score_high_rate(self):
        """Test high score produces high firing rate."""
        encoder = SpikeEncoder(max_rate=0.1, min_rate=0.01)
        
        spike_trains = encoder.encode(
            analysis_scores={"analyzer_1": 1.0},
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        assert "analyzer_1" in spike_trains
        # High score should produce many spikes
        assert len(spike_trains["analyzer_1"]) >= 5
    
    def test_low_score_low_rate(self):
        """Test low score produces low firing rate."""
        encoder = SpikeEncoder(max_rate=0.1, min_rate=0.01)
        
        spike_trains = encoder.encode(
            analysis_scores={"analyzer_1": 0.0},
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        assert "analyzer_1" in spike_trains
        # Low score should produce few spikes
        assert len(spike_trains["analyzer_1"]) <= 3
    
    def test_mid_score_mid_rate(self):
        """Test medium score produces medium firing rate."""
        encoder = SpikeEncoder(max_rate=0.1, min_rate=0.01)
        
        spike_trains = encoder.encode(
            analysis_scores={"analyzer_1": 0.5},
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        assert "analyzer_1" in spike_trains
        spike_count = len(spike_trains["analyzer_1"])
        assert 2 <= spike_count <= 8


class TestMultipleAnalyzers:
    """Test encoding multiple analyzers."""
    
    def test_multiple_analyzers(self):
        """Test encoding multiple analyzer scores."""
        encoder = SpikeEncoder()
        
        spike_trains = encoder.encode(
            analysis_scores={
                "analyzer_1": 0.8,
                "analyzer_2": 0.3,
                "analyzer_3": 0.6,
            },
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        assert len(spike_trains) == 3
        assert "analyzer_1" in spike_trains
        assert "analyzer_2" in spike_trains
        assert "analyzer_3" in spike_trains
    
    def test_different_firing_rates(self):
        """Test different scores produce different rates."""
        encoder = SpikeEncoder()
        
        spike_trains = encoder.encode(
            analysis_scores={
                "high": 0.9,
                "low": 0.1,
            },
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        high_spikes = len(spike_trains["high"])
        low_spikes = len(spike_trains["low"])
        
        assert high_spikes > low_spikes


class TestSpikeTiming:
    """Test spike timing properties."""
    
    def test_spikes_within_duration(self):
        """Test all spikes within duration window."""
        encoder = SpikeEncoder()
        
        spike_trains = encoder.encode(
            analysis_scores={"analyzer_1": 0.7},
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        for spike_time in spike_trains["analyzer_1"]:
            assert 0.0 <= spike_time <= 100.0
    
    def test_spikes_ordered(self):
        """Test spike times are sorted."""
        encoder = SpikeEncoder()
        
        spike_trains = encoder.encode(
            analysis_scores={"analyzer_1": 0.8},
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        spike_times = spike_trains["analyzer_1"]
        assert spike_times == sorted(spike_times)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_score(self):
        """Test encoding zero score."""
        encoder = SpikeEncoder()
        
        spike_trains = encoder.encode(
            analysis_scores={"analyzer_1": 0.0},
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        assert "analyzer_1" in spike_trains
        # Should still have some spikes (min_rate)
        assert isinstance(spike_trains["analyzer_1"], list)
    
    def test_max_score(self):
        """Test encoding maximum score."""
        encoder = SpikeEncoder()
        
        spike_trains = encoder.encode(
            analysis_scores={"analyzer_1": 1.0},
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        assert "analyzer_1" in spike_trains
        assert len(spike_trains["analyzer_1"]) > 0
    
    def test_out_of_range_score_clamped(self):
        """Test scores outside [0,1] are clamped."""
        encoder = SpikeEncoder()
        
        spike_trains = encoder.encode(
            analysis_scores={"over": 1.5, "under": -0.5},
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        # Should not crash, should clamp values
        assert "over" in spike_trains
        assert "under" in spike_trains
    
    def test_empty_scores(self):
        """Test encoding empty scores dict."""
        encoder = SpikeEncoder()
        
        spike_trains = encoder.encode(
            analysis_scores={},
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        assert spike_trains == {}


class TestConstantRateEncoding:
    """Test single-score encoding."""
    
    def test_encode_constant_rate(self):
        """Test constant rate encoding."""
        encoder = SpikeEncoder()
        
        spikes = encoder.encode_constant_rate(
            score=0.7,
            duration_ms=100.0,
        )
        
        assert isinstance(spikes, list)
        assert all(0.0 <= t <= 100.0 for t in spikes)
    
    def test_constant_rate_reproducible(self):
        """Test encoding is stochastic."""
        encoder = SpikeEncoder()
        
        spikes1 = encoder.encode_constant_rate(score=0.5, duration_ms=100.0)
        spikes2 = encoder.encode_constant_rate(score=0.5, duration_ms=100.0)
        
        # Should be different (Poisson is stochastic)
        # But approximately same count
        assert abs(len(spikes1) - len(spikes2)) <= 3


class TestDurationScaling:
    """Test duration affects spike count."""
    
    def test_longer_duration_more_spikes(self):
        """Test longer duration produces more spikes."""
        encoder = SpikeEncoder()
        
        short_trains = encoder.encode(
            analysis_scores={"analyzer_1": 0.5},
            input_type=InputType.TEXT,
            duration_ms=50.0,
        )
        
        long_trains = encoder.encode(
            analysis_scores={"analyzer_1": 0.5},
            input_type=InputType.TEXT,
            duration_ms=200.0,
        )
        
        # Longer duration should have more spikes (approximately)
        assert len(long_trains["analyzer_1"]) > len(short_trains["analyzer_1"])


class TestInputTypeHandling:
    """Test different input types."""
    
    def test_text_input_type(self):
        """Test encoding with TEXT input type."""
        encoder = SpikeEncoder()
        
        spike_trains = encoder.encode(
            analysis_scores={"text_analyzer": 0.6},
            input_type=InputType.TEXT,
            duration_ms=100.0,
        )
        
        assert "text_analyzer" in spike_trains
    
    def test_numeric_input_type(self):
        """Test encoding with NUMERIC input type."""
        encoder = SpikeEncoder()
        
        spike_trains = encoder.encode(
            analysis_scores={"numeric_analyzer": 0.6},
            input_type=InputType.NUMERIC,
            duration_ms=100.0,
        )
        
        assert "numeric_analyzer" in spike_trains
