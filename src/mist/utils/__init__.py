"""Lazy utility namespace for data preparation and the native MIST model."""

from __future__ import annotations

import importlib


_UTILITY_MODULES = (
    "chem_utils",
)


def __getattr__(name: str):
    for module_name in _UTILITY_MODULES:
        module = importlib.import_module(f"{__name__}.{module_name}")
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
