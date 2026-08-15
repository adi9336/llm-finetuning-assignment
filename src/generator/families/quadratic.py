"""Quadratic family (math): integer-root quadratics, built from chosen roots.

We construct a, b, c FROM integer roots r1, r2 (a in 1..9): b = -a(r1+r2),
c = a*r1*r2, so the discriminant is always a perfect square and roots are
exactly r1, r2. The reference solver recomputes via the quadratic formula —
honest exact match.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict

from src.generator.base import Puzzle, PuzzleFamily
from src.generator.registry import register


def format_term(coef: int, var: str, first: bool) -> str:
    if coef == 0:
        return ""
    if var == "x^2":
        if coef == 1:
            return "x^2" if first else "+ x^2"
        return f"{coef}x^2" if first else f"+ {coef}x^2"
    if coef == 1:
        return "x" if first else "+ x"
    if coef == -1:
        return "- x" if first else "- x"
    return f"{coef}x" if first else f"{coef:+d}x"


@register
class QuadraticRoots(PuzzleFamily):
    name = "quadratic"
    templates: Dict[str, Any] = {"roots": None, "discriminant": None}

    def generate(self, template: str, seed: int) -> Puzzle:
        rng: random.Random = self._rng(seed)
        a = rng.randint(1, 9)
        r1 = rng.randint(-9, 9)
        r2 = rng.randint(-9, 9)
        while r2 == r1:
            r2 = rng.randint(-9, 9)
        b = -a * (r1 + r2)
        c = a * r1 * r2

        # readable equation: ax^2 + bx + c = 0
        parts = [format_term(a, "x^2", True)]
        if b != 0:
            parts.append(format_term(b, "x", False))
        if c != 0:
            parts.append(f"{c:+d}" if c > 0 and b != 0 else f"{c}")
        eq = " ".join(parts) + " = 0"

        if template == "roots":
            answer = ", ".join(f"x={v}" for v in sorted((r1, r2)))
            solution = f"Factor or apply the formula: roots are {sorted((r1, r2))}"
            prompt = f"Solve for x. Reply with both roots, comma-separated, ascending:\n{eq}"
        else:  # discriminant
            d = b * b - 4 * a * c
            answer = str(d)
            solution = f"D = b^2 - 4ac = {b}^2 - 4*{a}*{c} = {d}"
            prompt = f"Compute the discriminant D = b^2 - 4ac. Reply with the integer only:\n{eq}"

        return Puzzle(
            id=self.make_id(self.name, template, seed),
            family=self.name,
            template=template,
            prompt=prompt,
            answer=answer,
            difficulty=2 if template == "roots" else 1,
            seed=seed,
            solution=solution,
            metadata={"a": a, "b": b, "c": c},
        )

    def solve(self, puzzle: Puzzle) -> str:
        a, b, c = puzzle.metadata["a"], puzzle.metadata["b"], puzzle.metadata["c"]
        d = b * b - 4 * a * c
        if puzzle.template == "discriminant":
            return str(d)
        s = math.isqrt(d)
        assert s * s == d, f"discriminant must be a perfect square (a={a}, b={b}, c={c})"
        x1 = (-b - s) // (2 * a)
        x2 = (-b + s) // (2 * a)
        return ", ".join(f"x={v}" for v in sorted((x1, x2)))
