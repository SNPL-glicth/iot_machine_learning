"""Backward-compatible shim — real module at anomaly/anomaly_domain_service.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.anomaly.anomaly_domain_service', __package__)
