"""Backward-compatible shim — real module at cognitive/plasticity_feedback.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.cognitive.plasticity_feedback', __package__)
