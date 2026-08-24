"""ThetaBelief: explicit posterior distribution over transition modes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .state_persistence import STATE_SCHEMA_VERSION

StateKey = Tuple[float, ...]


@dataclass
class ThetaBelief:
    """Explicit posterior distribution over transition modes with exponential forgetting.

    Encapsulates Theta_t = P(S_{t+1} | S_t, D_{1:t}) as a decaying weighted
    transition table. Every observation of a transition S_{t-1} -> S_t decays
    all previously observed transitions from S_{t-1} by alpha, so probability
    mass migrates organically to the current regime instead of being trapped
    in stale statistics.
    """

    alpha: float = 0.95                    # Decay factor for temporal forgetting
    min_weight_threshold: float = 1e-4     # Pruning threshold for stale transitions
    _transitions: Dict[StateKey, Dict[StateKey, float]] = field(default_factory=dict)
    _total_updates: int = 0

    @property
    def total_updates(self) -> int:
        """Number of transition observations absorbed so far."""
        return self._total_updates

    def update(self, from_state: StateKey, to_state: StateKey) -> None:
        """Updates belief with a new observed transition S_{t-1} -> S_t applying decay.
        
        After decay and pruning, re-normalizes remaining weights to preserve
        total probability mass (mass conservation invariant).
        """
        # 1. Apply decay to existing transitions from 'from_state'
        if from_state in self._transitions:
            # Track total weight before decay for mass conservation
            weights_before = self._transitions[from_state]
            total_before = sum(weights_before.values())
            
            for k in list(weights_before.keys()):
                weights_before[k] *= self.alpha
                if weights_before[k] < self.min_weight_threshold:
                    del weights_before[k]
            
            # 2. Re-normalize to preserve probability mass (mass conservation)
            total_after_decay = sum(weights_before.values())
            if total_after_decay > 0 and total_before > 0:
                # Scale remaining weights to preserve the decayed mass
                # The decay factor alpha already accounts for temporal forgetting,
                # so we normalize to alpha * total_before
                target_mass = total_before * self.alpha
                scale_factor = target_mass / total_after_decay
                for k in weights_before:
                    weights_before[k] *= scale_factor
        else:
            self._transitions[from_state] = {}

        # 3. Add current observation weight
        current_w = self._transitions[from_state].get(to_state, 0.0)
        self._transitions[from_state][to_state] = current_w + 1.0
        self._total_updates += 1

    def get_transition_probabilities(self, from_state: StateKey) -> Dict[StateKey, float]:
        """Returns posterior probabilities P(S_{t+1} | S_t = from_state, Theta_t)."""
        if from_state not in self._transitions or not self._transitions[from_state]:
            return {}

        weights = self._transitions[from_state]
        total_w = sum(weights.values())
        if total_w <= 0:
            return {}

        return {s_next: w / total_w for s_next, w in weights.items()}

    def compute_entropy(self, current_state: Optional[StateKey] = None) -> float:
        """
        Computes normalized posterior entropy H(Theta | D_t) in [0.0, 1.0].
        If current_state is provided and known, computes local transition entropy;
        otherwise computes average entropy across active states.
        """
        if not self._transitions:
            return 1.0  # Maximum uncertainty when no transitions are known

        if current_state and current_state in self._transitions:
            probs = list(self.get_transition_probabilities(current_state).values())
            return self._shannon_entropy_normalized(probs)

        # Global average entropy across known active states
        entropies = []
        for state in self._transitions:
            probs = list(self.get_transition_probabilities(state).values())
            if probs:
                entropies.append(self._shannon_entropy_normalized(probs))

        return float(np.mean(entropies)) if entropies else 1.0

    @staticmethod
    def _shannon_entropy_normalized(probs: List[float]) -> float:
        """Computes normalized Shannon entropy H / log2(K) in [0.0, 1.0]."""
        valid_p = [p for p in probs if p > 0]
        k = len(valid_p)
        if k <= 1:
            return 0.0  # Zero uncertainty if deterministic or single option

        h = -sum(p * math.log2(p) for p in valid_p)
        max_h = math.log2(k)
        return float(min(1.0, max(0.0, h / max_h)))

    def reset(self) -> None:
        """Clears posterior distribution state."""
        self._transitions.clear()
        self._total_updates = 0

    # ------------------------------------------------------------------
    # Persistence (StatePersistable contract)
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_key(key: StateKey) -> str:
        return "|".join(format(float(v), ".10g") for v in key)

    @staticmethod
    def _decode_key(encoded: str) -> StateKey:
        return tuple(float(v) for v in encoded.split("|"))

    def export_state(self) -> Dict[str, Any]:
        """Serialize the posterior to a JSON-safe dict."""
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "alpha": self.alpha,
            "min_weight_threshold": self.min_weight_threshold,
            "total_updates": self._total_updates,
            "transitions": {
                self._encode_key(from_key): {
                    self._encode_key(to_key): float(w)
                    for to_key, w in transitions.items()
                }
                for from_key, transitions in self._transitions.items()
            },
        }

    def import_state(self, payload: Dict[str, Any]) -> None:
        """Restore the posterior from an export_state payload.

        The posterior is imported directly: replaying history through update()
        would apply decay twice and corrupt the learned distribution.
        """
        if not isinstance(payload, dict) or "schema_version" not in payload:
            raise ValueError("ThetaBelief payload missing schema_version")
        if payload["schema_version"] != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported ThetaBelief schema: {payload['schema_version']}"
            )

        transitions_raw = payload.get("transitions", {})
        if not isinstance(transitions_raw, dict):
            raise ValueError("ThetaBelief payload 'transitions' must be a dict")

        restored: Dict[StateKey, Dict[StateKey, float]] = {}
        for enc_from, inner in transitions_raw.items():
            if not isinstance(inner, dict):
                raise ValueError("ThetaBelief transition weights must be dicts")
            from_key = self._decode_key(enc_from)
            restored[from_key] = {
                self._decode_key(enc_to): float(w)
                for enc_to, w in inner.items()
            }

        self._transitions.clear()
        self._transitions.update(restored)
        self._total_updates = int(payload.get("total_updates", 0))
