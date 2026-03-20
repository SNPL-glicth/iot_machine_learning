"""Online learning for feedforward layer.

Hebbian-inspired: "Neurons that fire together, wire together."
Updates weights after every document based on outcome feedback.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from .feedforward import FeedforwardLayer


class OnlineLearner:
    """Online weight updater with Hebbian-inspired learning.
    
    Updates weights per document based on:
    - Correctness: Strengthen connections that led to correct severity
    - Weakening: Weaken connections that led to wrong severity
    - Domain-specific: Separate weight matrices per domain
    
    Args:
        learning_rate: Base learning rate (default 0.01)
        momentum: Momentum coefficient (default 0.9)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Per-domain weight history
        self.domain_weights: Dict[str, tuple] = {}
        
        # Per-domain momentum buffers
        self.momentum_W: Dict[str, np.ndarray] = {}
        self.momentum_b: Dict[str, np.ndarray] = {}
    
    def update_weights(
        self,
        layer: FeedforwardLayer,
        input_vector: np.ndarray,
        predicted_severity: str,
        actual_severity: str,
        domain: str,
        severity_mapping: Dict[str, int],
    ) -> None:
        """Update layer weights based on prediction outcome.
        
        Args:
            layer: FeedforwardLayer to update
            input_vector: Input that was fed to the layer
            predicted_severity: What the network predicted
            actual_severity: Ground truth severity
            domain: Domain identifier
            severity_mapping: Map severity names to indices
        """
        # Check if we need to correct
        correct = (predicted_severity == actual_severity)
        
        # Get output from last forward pass
        if layer.last_output is None or layer.last_input is None:
            return
        
        # Compute target vector (one-hot for actual severity)
        target = np.zeros(layer.n_output)
        if actual_severity in severity_mapping:
            target[severity_mapping[actual_severity]] = 1.0
        
        # Compute error
        error = target - layer.last_output
        
        # Hebbian-inspired update: strengthen active connections
        # when prediction is correct, weaken when incorrect
        direction = 1.0 if correct else -0.5
        
        # Gradient for weights: outer product of input and error
        input_2d = layer.last_input.reshape(-1, 1)
        error_2d = error.reshape(1, -1)
        delta_W = direction * self.learning_rate * np.dot(input_2d, error_2d)
        
        # Gradient for biases
        delta_b = direction * self.learning_rate * error
        
        # Apply momentum
        key = f"{domain}_{layer.n_input}_{layer.n_output}"
        
        if key not in self.momentum_W:
            self.momentum_W[key] = np.zeros_like(delta_W)
            self.momentum_b[key] = np.zeros_like(delta_b)
        
        # Update momentum buffers
        self.momentum_W[key] = (
            self.momentum * self.momentum_W[key] + delta_W
        )
        self.momentum_b[key] = (
            self.momentum * self.momentum_b[key] + delta_b
        )
        
        # Apply update
        layer.update_weights(
            self.momentum_W[key],
            self.momentum_b[key],
        )
        
        # Store weights for this domain
        self.domain_weights[domain] = layer.get_weights()
    
    def get_domain_weights(
        self,
        domain: str,
    ) -> Optional[tuple]:
        """Retrieve learned weights for a domain.
        
        Args:
            domain: Domain identifier
            
        Returns:
            Tuple of (W, b) or None if domain not seen
        """
        return self.domain_weights.get(domain)
    
    def set_domain_weights(
        self,
        layer: FeedforwardLayer,
        domain: str,
    ) -> bool:
        """Load domain-specific weights into layer.
        
        Args:
            layer: Layer to update
            domain: Domain identifier
            
        Returns:
            True if weights were loaded, False if domain not found
        """
        weights = self.get_domain_weights(domain)
        if weights is None:
            return False
        
        W, b = weights
        layer.set_weights(W, b)
        return True
    
    def has_domain_history(self, domain: str) -> bool:
        """Check if domain has learned weights.
        
        Args:
            domain: Domain identifier
            
        Returns:
            True if domain weights exist
        """
        return domain in self.domain_weights
