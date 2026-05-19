"""Backward-compatible shim — real module at prediction/confidence_calibrator.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.prediction.confidence_calibrator', __package__)
