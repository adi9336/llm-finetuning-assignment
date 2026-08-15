"""Core puzzle model and family abstraction (M1).

Every puzzle is fully determined by (family, template, seed): same triple in,
byte-identical row out. Verification is exact-match against a reference solver:
the family re-computes the canonical answer from the row's own metadata, so no
LLM judge ever sits in the data path (INV: deterministic recipes only).
"""
from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class Puzzle:
    """One generated puzzle row, schema-valid against config/puzzle_schema.json."""

    id: str
    family: str
    template: str
    prompt: str
    answer: str
    difficulty: int = 1
    seed: int = 0
    solution: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


class PuzzleFamily(ABC):
    """A family of puzzles sharing a reference solver.

    Subclasses declare `name` and `templates` (dict: template key -> builder
    callable), implement `generate(template, seed)` to emit a Puzzle, and
    implement `solve(puzzle)` — the reference solver that re-derives the
    canonical answer from puzzle.metadata.
    """

    name: str = ""
    templates: Dict[str, Any] = {}

    @staticmethod
    def _rng(seed: int) -> random.Random:
        return random.Random(seed)

    @staticmethod
    def make_id(family: str, template: str, seed: int) -> str:
        raw = f"{family}:{template}:{seed}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    @abstractmethod
    def generate(self, template: str, seed: int) -> Puzzle:
        """Instantiate one puzzle from a template key + seed (deterministic)."""

    @abstractmethod
    def solve(self, puzzle: Puzzle) -> str:
        """Reference solver: recompute the canonical answer from puzzle.metadata."""
