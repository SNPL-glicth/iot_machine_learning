"""Backward-compatible shim — real module at cognitive/memory_recall_enricher.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.cognitive.memory_recall_enricher', __package__)
