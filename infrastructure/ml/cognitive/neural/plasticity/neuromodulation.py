"""Neuromodulation — dopamine-like reward signal for learning.

Modulates learning rate based on prediction error (surprise).
Inspired by dopaminergic neurons in biological brains.

High prediction error → high learning rate (surprise → learn fast)
Low prediction error → low learning rate (expected → consolidate)
"""

from __future__ import annotations

import numpy as np


class NeuromodulationSignal:
    """Dopamine-like neuromodulation for adaptive learning.
    
    Modulates learning rate based on:
    - Correctness of prediction
    - Confidence in prediction
    - Magnitude of surprise
    
    Args:
        baseline_modulation: Baseline modulation factor
        surprise_amplification: Amplification for unexpected outcomes
        consolidation_damping: Damping for expected outcomes
    """
    
    def __init__(
        self,
        baseline_modulation: float = 1.0,
        surprise_amplification: float = 3.0,
        consolidation_damping: float = 0.1,
    ) -> None:
        self.baseline = baseline_modulation
        self.surprise_amp = surprise_amplification
        self.consolidation = consolidation_damping
    
    def compute_modulation(
        self,
        predicted_severity: str,
        actual_severity: str,
        confidence: float,
    ) -> float:
        """Compute neuromodulation factor.
        
        Learning rate modulation based on prediction outcome:
        
        1. Correct + high confidence → 0.1 (consolidate, slow learning)
        2. Wrong + high confidence → 3.0 (surprise, fast learning)
        3. Wrong + low confidence → 1.0 (expected error, normal learning)
        4. Correct + low confidence → 1.5 (lucky guess, moderate learning)
        
        Args:
            predicted_severity: What model predicted
            actual_severity: Ground truth
            confidence: Model confidence [0, 1]
            
        Returns:
            Modulation factor [0.1, 3.0]
        """
        correct = (predicted_severity == actual_severity)
        high_confidence = confidence > 0.7
        
        if correct and high_confidence:
            # Expected success → consolidate, slow learning
            return self.consolidation
        
        elif not correct and high_confidence:
            # Surprising error → strong surprise signal, fast learning
            # This is the dopamine "prediction error" signal
            return self.surprise_amp
        
        elif not correct and not high_confidence:
            # Expected error → normal learning
            return self.baseline
        
        else:  # correct and low confidence
            # Lucky guess → moderate learning
            return self.baseline * 1.5
    
    def compute_reward_signal(
        self,
        predicted_severity: str,
        actual_severity: str,
        confidence: float,
    ) -> float:
        """Compute reward signal (prediction error).
        
        Positive reward for correct predictions,
        negative reward for incorrect predictions.
        Magnitude scaled by confidence.
        
        Args:
            predicted_severity: What model predicted
            actual_severity: Ground truth
            confidence: Model confidence [0, 1]
            
        Returns:
            Reward signal [-1, 1]
        """
        correct = (predicted_severity == actual_severity)
        
        if correct:
            # Positive reward, scaled by confidence
            return confidence
        else:
            # Negative reward, scaled by confidence
            # High confidence mistakes are more punished
            return -confidence
    
    def compute_dopamine_burst(
        self,
        predicted_severity: str,
        actual_severity: str,
        confidence: float,
    ) -> float:
        """Compute dopamine burst magnitude.
        
        Large burst for surprising reward (correct with low confidence).
        Dip for surprising punishment (wrong with high confidence).
        
        Args:
            predicted_severity: What model predicted
            actual_severity: Ground truth
            confidence: Model confidence [0, 1]
            
        Returns:
            Dopamine burst [-1, 1]
        """
        correct = (predicted_severity == actual_severity)
        
        # Compute surprise
        if correct:
            # Reward prediction error
            # Surprise = (1 - confidence) when correct
            surprise = 1.0 - confidence
            return surprise
        else:
            # Punishment prediction error
            # Surprise = -confidence when wrong
            return -confidence
    
    def should_enhance_learning(
        self,
        predicted_severity: str,
        actual_severity: str,
        confidence: float,
    ) -> bool:
        """Determine if learning should be enhanced.
        
        Args:
            predicted_severity: What model predicted
            actual_severity: Ground truth
            confidence: Model confidence [0, 1]
            
        Returns:
            True if learning rate should be increased
        """
        modulation = self.compute_modulation(
            predicted_severity, actual_severity, confidence
        )
        
        return modulation > self.baseline
