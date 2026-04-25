"""PipelineExecutorFactory — IMP-3.

Replaces the former module-level ``_pipeline_executor`` singleton.
Each call to :meth:`create` returns a **fresh** :class:`PipelineExecutor`
with its own ``_phases`` list, so two concurrent pipeline runs cannot
share phase-local mutable state (e.g. SanitizePhase resolving stores
from ``ctx.orchestrator``, FusePhase collecting Hampel flags, etc.).

Instance cost is ~2 KB (13 stateless phase objects + one
``AssemblyPhase``) — acceptable for per-request allocation.

``flags_snapshot`` is accepted today but does **not** yet influence
phase composition: every request instantiates the same 13 phases in
the same order. The argument is plumbed through so future flag-driven
phase selection (skipping heavy phases like ``NarrativeUnificationPhase``
based on a feature flag) has a clean hook without another refactor.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .pipeline_executor import PipelineExecutor

logger = logging.getLogger(__name__)


class PipelineExecutorFactory:
    """Produces a fresh :class:`PipelineExecutor` per request.

    Callers are expected to invoke :meth:`create` once per pipeline run
    and then discard the returned instance. No caching is performed.
    """

    def create(self, flags_snapshot: Optional[Any] = None) -> PipelineExecutor:
        """Return a new :class:`PipelineExecutor`.

        Args:
            flags_snapshot: Feature-flag snapshot for this run. Reserved
                for future phase-selection logic; currently unused
                (phase list is static — see module docstring).
        """
        # Touching flags_snapshot explicitly avoids "unused arg" lint
        # warnings and documents intent.
        _ = flags_snapshot
        return PipelineExecutor()


__all__ = ["PipelineExecutorFactory"]
