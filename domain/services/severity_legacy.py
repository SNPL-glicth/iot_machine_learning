"""Backward-compatible shim — real module at severity/severity_legacy.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.severity.severity_legacy', __package__)
