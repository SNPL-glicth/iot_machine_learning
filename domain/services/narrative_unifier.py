"""Backward-compatible shim — real module at cognitive/narrative_unifier.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.cognitive.narrative_unifier', __package__)
