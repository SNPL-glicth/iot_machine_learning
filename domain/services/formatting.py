"""Backward-compatible shim — real module at severity/formatting.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.severity.formatting', __package__)
