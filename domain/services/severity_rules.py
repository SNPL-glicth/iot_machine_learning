"""Backward-compatible shim — real module at severity/severity_rules.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.severity.severity_rules', __package__)
