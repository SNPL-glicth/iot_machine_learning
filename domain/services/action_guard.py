"""Backward-compatible shim — real module at actions/action_guard.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.actions.action_guard', __package__)
