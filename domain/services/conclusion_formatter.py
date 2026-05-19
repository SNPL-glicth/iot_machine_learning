"""Backward-compatible shim — real module at cognitive/conclusion_formatter.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.cognitive.conclusion_formatter', __package__)
