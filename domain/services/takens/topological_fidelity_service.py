"""Pure mathematical service for topological fidelity evaluation and manifold vetoes.

Conforms to:
- ISO/IEC 25010: Reliability and Numerical Fault Tolerance.
- ISO/IEC 22989: Artificial Intelligence Explainability and Auditability.
- Ultra-Low Latency (HFT): Fast vectorized nearest-neighbor heuristics without O(N^2) loops.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from domain.entities.takens.takens_parameters import TakensParameters
from domain.entities.takens.topological_audit import TopologicalAuditRecord

_DEFAULT_PARAMS = TakensParameters()


def compute_fast_fnn_ratio(
    query_vector: np.ndarray,
    historic_matrix: np.ndarray,
    r_tol: float = 15.0,
    a_tol: float = 2.0,
) -> float:
    """Compute False Nearest Neighbors (FNN) proxy ratio in R^m in O(W * m) time.

    Measures whether nearest neighbors in lower-dimensional projection diverge
    disproportionately along higher delay coordinates, identifying topological folding.

    Args:
        query_vector: Current state vector y_t in R^m.
        historic_matrix: Delay trajectory matrix Y of shape (W, m).
        r_tol: Relative expansion threshold for false neighbor criterion.
        a_tol: Absolute distance scale threshold based on attractor radius.

    Returns:
        Omega_FNN ratio in [0.0, 1.0].
    """
    y = np.asarray(query_vector, dtype=np.float64).flatten()
    Y = np.asarray(historic_matrix, dtype=np.float64)

    if Y.ndim != 2 or Y.shape[0] < 2 or y.size < 2:
        return 0.0

    w, m = Y.shape
    m_sub = min(m, y.size)
    if m_sub < 2:
        return 0.0

    # 1. Distances in subspace R^(m-1): O(W * m)
    diffs_sub = Y[:, : m_sub - 1] - y[: m_sub - 1]
    dists_sub = np.linalg.norm(diffs_sub, axis=1)

    # Ignore zero distance (exact self-match at same timestamp)
    non_zero_mask = dists_sub > 1e-8
    if not np.any(non_zero_mask):
        return 0.0

    valid_dists = dists_sub[non_zero_mask]
    valid_indices = np.where(non_zero_mask)[0]

    min_idx = valid_indices[int(np.argmin(valid_dists))]
    r_d = float(dists_sub[min_idx])

    # 2. Distance in the full m-th delayed coordinate
    delta_m = abs(float(Y[min_idx, m_sub - 1] - y[m_sub - 1]))

    # Standard Kennel-Brown-Abarbanel criteria with flatline protection
    attr_std = float(np.std(Y[:, m_sub - 1]))
    if attr_std < 1e-4:
        is_false_neighbor = bool(delta_m / max(1e-8, r_d) > r_tol)
    else:
        is_false_neighbor = bool((delta_m / max(1e-8, r_d) > r_tol) or (delta_m / attr_std > a_tol))

    return 1.0 if is_false_neighbor else 0.0


def evaluate_topological_fidelity(
    trajectory_values: np.ndarray | list[float] | Any,
    historic_matrix: np.ndarray,
    effective_dimension: float,
    params: TakensParameters | None = None,
    timestamp_ns: int = 0,
) -> tuple[float, TopologicalAuditRecord]:
    """Evaluate candidate trajectory against embedded manifold and emit coherence score.

    Computes Psi_takens in [0.0, 1.0] and enforces hard topological vetoes if the
    trajectory exhibits self-intersection or out-of-manifold orthogonal divergence.

    Args:
        trajectory_values: Candidate sequence, array, or Trajectory object.
        historic_matrix: Local delay matrix Y of shape (W, m).
        effective_dimension: Local participation ratio dimension D_eff.
        params: Embedding limits and thresholds.
        timestamp_ns: Event timestamp in nanoseconds.

    Returns:
        Tuple of (psi_takens score, TopologicalAuditRecord audit).
    """
    cfg = params or _DEFAULT_PARAMS
    Y = np.asarray(historic_matrix, dtype=np.float64)

    # Duck-typing extraction of trajectory delta values
    if hasattr(trajectory_values, "delta_states"):
        raw_deltas = trajectory_values.delta_states
        vals = raw_deltas[:, 0] if raw_deltas.ndim > 1 else raw_deltas
    else:
        vals = np.asarray(trajectory_values, dtype=np.float64).flatten()

    if vals.size == 0 or Y.ndim != 2 or Y.shape[0] == 0:
        audit = TopologicalAuditRecord(
            timestamp_ns=timestamp_ns,
            d_effective=float(effective_dimension),
            fnn_ratio=0.0,
            is_manifold_veto=False,
            manifold_coherence=1.0,
            embedded_norm=0.0,
        )
        return 1.0, audit

    # Extract terminal trajectory vector in R^m
    from .delay_embedding_service import extract_embedded_vector
    y_cand = extract_embedded_vector(vals, cfg)
    cand_norm = float(np.linalg.norm(y_cand))

    # Fast FNN proxy
    fnn = compute_fast_fnn_ratio(y_cand, Y)

    # Orthogonal distance from candidate vector to attractor centroid
    centroid = np.mean(Y, axis=0)
    dist_manifold = float(np.linalg.norm(y_cand - centroid))

    # Check Veto Conditions
    is_veto = False
    veto_reason = None

    if dist_manifold > cfg.max_geodesic_distance:
        is_veto = True
        veto_reason = f"out_of_manifold_escape_dist_{dist_manifold:.2f}_exceeds_{cfg.max_geodesic_distance:.2f}"
    elif fnn >= cfg.tau_fnn:
        is_veto = True
        veto_reason = f"fnn_explosion_ratio_{fnn:.2f}_exceeds_threshold_{cfg.tau_fnn:.2f}"
    elif effective_dimension < cfg.spectral_collapse_threshold:
        is_veto = True
        veto_reason = f"spectral_collapse_deff_{effective_dimension:.2f}_below_{cfg.spectral_collapse_threshold:.2f}"

    # Calculate Continuous Coherence Score Psi_takens
    decay_factor = math.exp(-max(0.0, dist_manifold) / max(1e-3, cfg.max_geodesic_distance))
    psi_takens = float(max(0.0, min(1.0, decay_factor * (1.0 - cfg.out_of_manifold_penalty * fnn))))

    if is_veto:
        psi_takens = 0.0

    audit = TopologicalAuditRecord(
        timestamp_ns=timestamp_ns,
        d_effective=float(effective_dimension),
        fnn_ratio=float(fnn),
        is_manifold_veto=is_veto,
        veto_reason=veto_reason,
        manifold_coherence=psi_takens,
        embedded_norm=cand_norm,
    )

    return psi_takens, audit
