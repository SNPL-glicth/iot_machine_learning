"""Hybrid Neural Engine — thin orchestrator for modular pipeline.

Orchestrates encoder → SNN → classical → fusion → decoder stages.
Produces NeuralResult comparable to UniversalResult for arbitration.
"""

from __future__ import annotations

import logging
from typing import Dict

from .types import NeuralResult
from .pipeline import (
    EncoderStage,
    SNNStage,
    ClassicalStage,
    FusionStage,
    DecoderStage,
)
from .snn import SpikeEncoder, SpikeDecoder, SNNLayer
from .classical import FeedforwardLayer, OnlineLearner
from ..universal.analysis.types import InputType

logger = logging.getLogger(__name__)


class HybridNeuralEngine:
    """Thin orchestrator for hybrid SNN + classical analysis pipeline."""
    
    def __init__(
        self,
        n_input: int = 10,
        n_hidden_snn: int = 20,
        n_hidden_classical: int = 16,
        n_output: int = 5,
        snn_weight: float = 0.5,
        enable_online_learning: bool = True,
    ) -> None:
        self.n_input = n_input
        self.snn_weight = snn_weight
        self.enable_online_learning = enable_online_learning
        
        # Initialize components
        spike_encoder = SpikeEncoder(max_rate=0.1, min_rate=0.01)
        snn_layer = SNNLayer(n_input, n_hidden_snn, n_output, dt=1.0)
        spike_decoder = SpikeDecoder(["info", "low", "medium", "high", "critical"])
        classical_layer = FeedforwardLayer(n_input, n_output, "softmax", seed=42)
        
        # Initialize pipeline stages
        self.encoder = EncoderStage(spike_encoder, duration_ms=100.0)
        self.snn_stage = SNNStage(snn_layer, duration_ms=100.0)
        self.classical_stage = ClassicalStage(classical_layer, n_input)
        self.fusion = FusionStage(snn_weight, 1.0 - snn_weight)
        self.decoder = DecoderStage(spike_decoder, duration_ms=100.0)
        
        # Online learner
        self.online_learner = OnlineLearner(0.01, 0.9) if enable_online_learning else None
        self.severity_mapping = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    
    def analyze(
        self,
        analysis_scores: Dict[str, float],
        input_type: InputType,
        domain: str,
    ) -> NeuralResult:
        """Run hybrid neural analysis through modular pipeline."""
        try:
            # Load domain-specific weights
            if self.online_learner and self.online_learner.has_domain_history(domain):
                self.online_learner.set_domain_weights(self.classical_stage.feedforward_layer, domain)
            
            # Stage 1: Encode scores → spike trains
            spike_trains = self.encoder.process(analysis_scores, input_type)
            
            # Stage 2: SNN forward pass
            spike_patterns, firing_rates = self.snn_stage.process(spike_trains)
            _, snn_output = self.decoder.decode_snn_output(spike_patterns)
            
            # Stage 3: Classical forward pass
            classical_output = self.classical_stage.process(analysis_scores)
            
            # Stage 4: Fusion
            hybrid_output, confidence = self.fusion.process(snn_output, classical_output, spike_patterns)
            
            # Stage 5: Decode severity
            severity = self.decoder.decode_hybrid_output(hybrid_output)
            
            # Stage 6: Extract energy metrics
            energy, active, silent = self.snn_stage.get_energy_metrics()
            
            return NeuralResult(
                severity=severity,
                confidence=confidence,
                spike_patterns=spike_patterns,
                firing_rates=firing_rates,
                energy_consumed=energy,
                active_neurons=active,
                silent_neurons=silent,
                domain=domain,
                input_type=input_type,
                snn_output=snn_output,
                classical_output=classical_output,
                hybrid_weight_snn=self.snn_weight,
                hybrid_weight_classical=1.0 - self.snn_weight,
            )
        except Exception as e:
            logger.error(f"hybrid_neural_analysis_failed: {e}", exc_info=True)
            return self._fallback_result(domain, input_type)
    
    def update_from_feedback(self, analysis_scores: Dict, predicted_severity: str, actual_severity: str, domain: str) -> None:
        """Update classical weights based on feedback."""
        if self.online_learner:
            input_vector = self.classical_stage._scores_to_vector(analysis_scores)
            self.online_learner.update_weights(
                self.classical_stage.feedforward_layer, input_vector,
                predicted_severity, actual_severity, domain, self.severity_mapping
            )
    
    def _fallback_result(self, domain: str, input_type: InputType) -> NeuralResult:
        """Fallback result on error."""
        return NeuralResult(
            severity="info", confidence=0.3, spike_patterns={}, firing_rates={},
            energy_consumed=0.0, active_neurons=0, silent_neurons=35,
            domain=domain, input_type=input_type, snn_output=0.0, classical_output=0.0,
            hybrid_weight_snn=self.snn_weight, hybrid_weight_classical=1.0 - self.snn_weight,
        )
