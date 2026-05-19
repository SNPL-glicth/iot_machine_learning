"""Backward-compatible shim — real module at pattern/domain_boundary_checker.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.pattern.domain_boundary_checker', __package__)
