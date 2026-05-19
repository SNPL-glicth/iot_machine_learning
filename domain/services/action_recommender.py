"""Backward-compatible shim — real module at actions/action_recommender.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.actions.action_recommender', __package__)
