"""Backward-compatible shim — real module at cognitive/cognitive_constants.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.cognitive.cognitive_constants', __package__)
