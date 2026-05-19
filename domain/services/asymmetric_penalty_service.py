"""Backward-compatible shim — real module at anomaly/asymmetric_penalty_service.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.anomaly.asymmetric_penalty_service', __package__)
