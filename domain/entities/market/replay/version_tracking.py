"""Version tracking utilities for ablation (FASE 8)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

__all__ = ["parse_version_reason", "active_versions_by_window"]

_REASON_RE = re.compile(r"^wf\s+(\S+)\s+W(\d+):")


def parse_version_reason(reason: str) -> tuple[str, int] | None:
    """Extrae (símbolo, índice de ventana) del reason de una versión."""
    match = _REASON_RE.match(reason)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def active_versions_by_window(
    version_rows: Sequence[Mapping[str, object]],
    symbol: str,
    window_indices: Sequence[int],
) -> dict[int, Mapping[str, object] | None]:
    """Reconstruye la versión activa de cada ventana (ver docstring).

    ``version_rows`` debe venir en orden de version_id (cadena append-only).
    Devuelve {índice de ventana: versión} — None cuando no hay ninguna
    versión registrada (ni heredada) para el símbolo.
    """
    ordered = list(version_rows)
    mine: list[tuple[Mapping[str, object], int]] = []
    for row in ordered:
        parsed = parse_version_reason(str(row.get("reason") or ""))
        if parsed is not None and parsed[0] == symbol:
            mine.append((row, parsed[1]))
    if not mine:
        base: Mapping[str, object] | None = ordered[-1] if ordered else None
        return {w: base for w in window_indices}

    first_pos = next(
        i for i, row in enumerate(ordered) if row is mine[0][0]
    )
    inherited: Mapping[str, object] | None = (
        ordered[first_pos - 1] if first_pos > 0 else None
    )

    running: Mapping[str, object] | None = inherited
    result: dict[int, Mapping[str, object] | None] = {}
    mine_sorted = sorted(mine, key=lambda item: item[1])
    for window_index in sorted(window_indices):
        while mine_sorted and mine_sorted[0][1] <= window_index:
            running = mine_sorted.pop(0)[0]
        result[window_index] = running
    return result