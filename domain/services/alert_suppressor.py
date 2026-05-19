"""Backward-compatible shim — real module at anomaly/alert_suppressor.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.anomaly.alert_suppressor', __package__)
