"""Classical stage — feedforward neural network."""

from __future__ import annotations

import numpy as np
from typing import Dict

from ..classical import FeedforwardLayer


class ClassicalStage:
    """Processes input scores through classical feedforward layer.
    
    Args:
        feedforward_layer: FeedforwardLayer instance
        n_input: Number of input features
    """
    
    def __init__(
        self,
        feedforward_layer: FeedforwardLayer,
        n_input: int,
    ) -> None:
        self.feedforward_layer = feedforward_layer
        self.n_input = n_input
    
    def process(
        self,
        analysis_scores: Dict[str, float],
    ) -> float:
        """Run classical feedforward pass.
        
        Args:
            analysis_scores: Dict of {analyzer_name: score [0, 1]}
            
        Returns:
            Classical output score [0, 1]
        """
        # Convert scores to input vector
        input_vector = self._scores_to_vector(analysis_scores)
        
        # Forward pass
        output = self.feedforward_layer.forward(input_vector)
        
        # Get dominant class score
        classical_output = float(output.max())
        
        return classical_output
    
    def _scores_to_vector(
        self,
        analysis_scores: Dict[str, float],
    ) -> np.ndarray:
        """Convert analysis scores to input vector.
        
        Returns:
            NumPy array of shape (n_input,)
        """
        vector = np.zeros(self.n_input)
        
        for idx, (analyzer_name, score) in enumerate(analysis_scores.items()):
            if idx < self.n_input:
                vector[idx] = score
        
        return vector
