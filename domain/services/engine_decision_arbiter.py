"""Backward-compatible shim — real module at prediction/engine_decision_arbiter.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.prediction.engine_decision_arbiter', __package__)
