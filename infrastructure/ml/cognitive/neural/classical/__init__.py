"""Classical feedforward neural network layer.

Dense feedforward layer with online learning capabilities.

Components:
    - FeedforwardLayer: Dense connections with activation functions
    - Activation functions: ReLU, sigmoid, softmax (pure numpy)
    - OnlineLearner: Per-document weight updates (Hebbian-inspired)
"""

from .feedforward import FeedforwardLayer
from .activations import relu, sigmoid, softmax
from .online_learner import OnlineLearner

__all__ = [
    "FeedforwardLayer",
    "relu",
    "sigmoid",
    "softmax",
    "OnlineLearner",
]
