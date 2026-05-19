"""Backward-compatible shim — real module at actions/action_catalog.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.actions.action_catalog', __package__)
