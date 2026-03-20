"""Voting logic for anomaly detection ensemble.

Components:
    - VotingStrategy: Combines votes using weighted average
    - Context builders: Build kwargs for detector vote() calls
"""

from .strategy import VotingStrategy
from .context_builder import build_vote_context, extract_vel_z, extract_acc_z

__all__ = [
    "VotingStrategy",
    "build_vote_context",
    "extract_vel_z",
    "extract_acc_z",
]
