"""State persistence contract for Rosa Roja domain components.

Domain-pure serialization: components export/import plain dicts. No I/O lives
here — infrastructure adapters (Redis, in-memory) transport the payloads via
the MLStateStore port.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

STATE_SCHEMA_VERSION = 1


@runtime_checkable
class StatePersistable(Protocol):
    """Contract for components whose learning state can be snapshotted."""

    def export_state(self) -> Dict[str, Any]:
        """Serialize learning state to a JSON-safe dict."""
        ...

    def import_state(self, payload: Dict[str, Any]) -> None:
        """Restore learning state from a dict produced by export_state.

        Raises:
            ValueError: if the payload schema is unknown or malformed.
        """
        ...


def pack_array(arr: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    """Pack a numpy matrix into nested lists (JSON-safe)."""
    if arr is None:
        return None
    return arr.astype(float).tolist()


def unpack_array(data: Optional[List[List[float]]]) -> Optional[np.ndarray]:
    """Unpack nested lists back to a float64 numpy matrix."""
    if data is None:
        return None
    return np.asarray(data, dtype=np.float64)


def movement_to_raw(movement) -> Dict[str, Any]:
    """Serialize a Movement to its raw fields.

    Derived fields (velocity, direction, rhythm_signature) are recomputed
    deterministically by Movement.from_raw on restore, so only the source of
    truth is persisted.
    """
    return {
        "delta_state": movement.delta_state.astype(float).tolist(),
        "delta_time": float(movement.delta_time),
        "timestamp": float(movement.timestamp),
        "mahalanobis_distance": float(movement.mahalanobis_distance),
    }


def movements_from_raw(raw_movements: List[Dict[str, Any]]) -> List:
    """Rebuild a chain of Movements from raw fields.

    Rhythm signatures depend on the predecessor, so movements are rebuilt in
    order chaining prev_movement exactly as Module 1 does during ingestion.
    """
    from .movement import Movement

    rebuilt: List = []
    prev = None
    for raw in raw_movements:
        movement = Movement.from_raw(
            delta_state=np.asarray(raw["delta_state"], dtype=float),
            delta_time=float(raw["delta_time"]),
            timestamp=float(raw["timestamp"]),
            mahalanobis_dist=float(raw["mahalanobis_distance"]),
            prev_movement=prev,
        )
        rebuilt.append(movement)
        prev = movement
    return rebuilt
