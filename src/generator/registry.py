"""Family registry (M1). Families self-register via the @register decorator.

Discovery: importing this module imports every module in
src/generator/families/, which registers each PuzzleFamily subclass.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Type

from src.generator.base import PuzzleFamily

_FAMILIES: Dict[str, Type[PuzzleFamily]] = {}
_LOADED = False


def register(cls: Type[PuzzleFamily]) -> Type[PuzzleFamily]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must declare a non-empty `name`")
    if cls.name in _FAMILIES:
        raise ValueError(f"duplicate family name: {cls.name}")
    _FAMILIES[cls.name] = cls
    return cls


def _load_all() -> None:
    global _LOADED
    if _LOADED:
        return
    import src.generator.families as families_pkg

    for mod_info in pkgutil.iter_modules(families_pkg.__path__):
        importlib.import_module(f"{families_pkg.__name__}.{mod_info.name}")
    _LOADED = True


def families() -> Dict[str, Type[PuzzleFamily]]:
    _load_all()
    return dict(_FAMILIES)


def get_family(name: str) -> Type[PuzzleFamily]:
    _load_all()
    try:
        return _FAMILIES[name]
    except KeyError:
        raise KeyError(f"unknown puzzle family: {name!r} (registered: {sorted(_FAMILIES)})") from None
