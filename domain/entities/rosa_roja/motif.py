"""Topological Motif domain entities: Kinematic archetypes in phase space.

Replaces rigid Cartesian decimal rounding (round(v, 2)) with scale-invariant
topological equivalence classes ("Las 100 Manos de Netero"), preserving
phase-space negentropy and structuring continuous manifold dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TopologicalMotifKey:
    """Canonical, immutable representation of a kinematic motion archetype.

    Represents a topological equivalence class in continuous phase space,
    decoupling state transitions from absolute coordinate grids.

    Attributes:
        direction_index: Partitioned orientation cluster in spherical manifold (0 to N_dir - 1).
        acceleration_mode: Curvature dynamic regime (-2: hard brake, -1: decel, 0: cruise, +1: accel, +2: extreme impulse).
        tempo_band: Chronometric ratio band (-1: contractive, 0: isochronous, +1: dilative).
        motif_id: Unique compact canonical integer identifier [0, K-1] for O(1) table indexing.
    """

    direction_index: int
    acceleration_mode: int
    tempo_band: int
    motif_id: int

    def __repr__(self) -> str:
        return (
            f"TopologicalMotifKey(id={self.motif_id}, dir={self.direction_index}, "
            f"acc={self.acceleration_mode}, tempo={self.tempo_band})"
        )
