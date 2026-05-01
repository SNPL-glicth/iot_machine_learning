"""Dense feedforward layer with configurable activation."""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from .activations import relu, sigmoid, softmax


class FeedforwardLayer:
    """Dense feedforward layer: y = activation(Wx + b).
    
    Args:
        n_input: Number of input features
        n_output: Number of output features
        activation: Activation function name ('relu', 'sigmoid', 'softmax', 'linear')
        seed: Random seed for weight initialization
    """
    
    def __init__(
        self,
        n_input: int,
        n_output: int,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        self.n_input = n_input
        self.n_output = n_output
        self.activation_name = activation
        
        # Initialize weights and biases
        rng = np.random.RandomState(seed)
        
        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (n_input + n_output))
        self.W = rng.uniform(-limit, limit, (n_input, n_output))
        self.b = np.zeros(n_output)
        
        # Select activation function
        self.activation = self._get_activation(activation)
        
        # Cache for backprop (if needed)
        self.last_input: Optional[np.ndarray] = None
        self.last_output: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: y = activation(Wx + b).
        
        Args:
            x: Input array, shape (n_input,) or (batch_size, n_input)
            
        Returns:
            Output array, shape (n_output,) or (batch_size, n_output)
        """
        # Cache input for potential backprop
        self.last_input = x
        
        # Linear transform
        z = np.dot(x, self.W) + self.b
        
        # Apply activation
        y = self.activation(z)
        
        # Cache output
        self.last_output = y
        
        return y
    
    def update_weights(
        self,
        delta_W: np.ndarray,
        delta_b: np.ndarray,
    ) -> None:
        """Update weights and biases.
        
        Args:
            delta_W: Weight update matrix
            delta_b: Bias update vector
        """
        self.W += delta_W
        self.b += delta_b
    
    def get_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current weights and biases.
        
        Returns:
            Tuple of (W, b)
        """
        return self.W.copy(), self.b.copy()
    
    def set_weights(self, W: np.ndarray, b: np.ndarray) -> None:
        """Set weights and biases.
        
        Args:
            W: Weight matrix
            b: Bias vector
        """
        if W.shape != (self.n_input, self.n_output):
            raise ValueError(f"Weight shape mismatch: expected {(self.n_input, self.n_output)}, got {W.shape}")
        if b.shape != (self.n_output,):
            raise ValueError(f"Bias shape mismatch: expected {(self.n_output,)}, got {b.shape}")
        
        self.W = W.copy()
        self.b = b.copy()
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name.
        
        Args:
            name: Activation function name
            
        Returns:
            Activation function
        """
        activations = {
            "relu": relu,
            "sigmoid": sigmoid,
            "softmax": softmax,
            "linear": lambda x: x,
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        
        return activations[name]
    
    def __repr__(self) -> str:
        return f"FeedforwardLayer(n_input={self.n_input}, n_output={self.n_output}, activation={self.activation_name})"
