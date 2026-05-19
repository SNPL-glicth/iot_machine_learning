"""Backward-compatible shim — real module at cognitive/situation_vector_builder.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.cognitive.situation_vector_builder', __package__)
