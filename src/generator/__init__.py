"""M1 — Puzzle engine: deterministic, reference-solver-verifiable puzzle generation.

Public surface: Puzzle, PuzzleFamily, register(), families(), get_family().
"""
from src.generator.base import Puzzle, PuzzleFamily
from src.generator.registry import register, families, get_family

__all__ = ["Puzzle", "PuzzleFamily", "register", "families", "get_family"]
