"""Local polynomial least-squares fit for derivative estimation.

Fits a polynomial of degree *k* to the last few points using the
normal equations (A^T A c = A^T y) solved via Gaussian elimination.

More noise-resistant than finite differences because it averages
over multiple points.  No external dependencies (no numpy).

Pure function — no I/O, no state, no logging.
"""

from __future__ import annotations

from typing import List, Optional

from .types import TaylorCoefficients


def least_squares_fit(
    values: List[float], dt: float, order: int,
) -> Optional[TaylorCoefficients]:
    """Fit a local polynomial and extract derivatives.

    Fits a polynomial of degree ``order`` (1–3) to the last
    ``min(2*order+1, n)`` points.  Time indices are centered at
    the last point: ..., -2Δt, -Δt, 0.

    Args:
        values: Time series (most recent last).
        dt: Time step.
        order: Polynomial degree (1–3).

    Returns:
        ``TaylorCoefficients`` or ``None`` if the system is singular.
    """
    n = len(values)
    eff = min(order, 3)
    window = min(2 * eff + 1, n)

    if window < eff + 1:
        return None

    tail = values[-window:]
    m = len(tail)
    ts = [(i - m + 1) * dt for i in range(m)]

    cols = eff + 1
    ata = [[0.0] * cols for _ in range(cols)]
    aty = [0.0] * cols

    for k in range(m):
        powers = [1.0]
        for j in range(1, cols):
            powers.append(powers[-1] * ts[k])
        for i in range(cols):
            aty[i] += powers[i] * tail[k]
            for j in range(cols):
                ata[i][j] += powers[i] * powers[j]

    poly = _solve_linear(ata, aty)
    if poly is None:
        return None

    f_t = values[-1]
    f1 = poly[1] if len(poly) > 1 else 0.0
    f2 = 2.0 * poly[2] if len(poly) > 2 else 0.0
    f3 = 6.0 * poly[3] if len(poly) > 3 else 0.0

    return TaylorCoefficients(
        f_t=f_t, f_prime=f1, f_double_prime=f2,
        f_triple_prime=f3, estimated_order=eff, method="least_squares",
    )


def _solve_linear(A: list, b: list) -> Optional[list]:
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Returns None if the matrix is singular (pivot < 1e-12).
    """
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]

        if abs(M[col][col]) < 1e-12:
            return None

        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x
