"""SCAFFOLD family — proves the generator/verifier plumbing end to end.

This is intentionally the simplest honest family: two integers, one template.
The REAL recipe families (vigenere cipher, quadratic, code, logic,
mixed-step) are authored AFTER the kimi-k3 model switch, per the M1
checkpoint gate — this scaffold exists so the framework is tested and
committed before that gate.
"""
from __future__ import annotations

import random
from typing import Any, Dict

from src.generator.base import Puzzle, PuzzleFamily
from src.generator.registry import register


@register
class ScaffoldAdd(PuzzleFamily):
    name = "scaffold.add"
    templates: Dict[str, Any] = {"add": None}

    def generate(self, template: str, seed: int) -> Puzzle:
        rng: random.Random = self._rng(seed)
        a = rng.randint(1, 9999)
        b = rng.randint(1, 9999)
        answer = str(a + b)
        return Puzzle(
            id=self.make_id(self.name, template, seed),
            family=self.name,
            template=template,
            prompt=f"What is {a} + {b}? Reply with only the integer.",
            answer=answer,
            difficulty=1,
            seed=seed,
            solution=f"{a} + {b} = {answer}",
            metadata={"a": a, "b": b},
        )

    def solve(self, puzzle: Puzzle) -> str:
        a = int(puzzle.metadata["a"])
        b = int(puzzle.metadata["b"])
        return str(a + b)
