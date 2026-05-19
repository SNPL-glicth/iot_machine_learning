"""Backward-compatible shim — real module at anomaly/threshold_evaluator.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.anomaly.threshold_evaluator', __package__)
