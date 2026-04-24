"""HyperparameterAdaptor package (IMP-4c).

Redis-only, thread-safe, single source of truth for per-``(series_id,
engine_name)`` hyperparameters across :class:`PipelineExecutor`
instances. When no Redis client is supplied the adaptor is inert
(``load`` returns ``None``; ``save``/``reset`` are no-ops) and engines
fall back to their hardcoded defaults.
"""

from .hyperparameter_adaptor import HyperparameterAdaptor

__all__ = ["HyperparameterAdaptor"]
