"""Logic family (logic): boolean expression evaluation over A, B, C.

Expression is a JSON-safe nested list tree, e.g. ["AND", "A", ["OR", "B", ["NOT", "C"]]].
The reference solver evaluates the tree against the stored assignment — exact
truth value, no LLM judge.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Union

from src.generator.base import Puzzle, PuzzleFamily
from src.generator.registry import register

VARS = ["A", "B", "C"]

# expr: Union[str, List] where str is a var name and List is [op, ...operands]
Expr = Union[str, List[Any]]


def _rand_expr(rng: random.Random, depth: int, available: List[str]) -> Expr:
    if depth <= 0 or (len(available) <= 1 and rng.random() < 0.5):
        v = rng.choice(available)
        return v
    op = rng.choice(["AND", "OR", "NOT"])
    if op == "NOT":
        return ["NOT", _rand_expr(rng, depth - 1, available)]
    left = _rand_expr(rng, depth - 1, available)
    right = _rand_expr(rng, depth - 1, available)
    return [op, left, right]


def render(expr: Expr) -> str:
    if isinstance(expr, str):
        return expr
    op = expr[0]
    if op == "NOT":
        return f"NOT {render(expr[1])}"
    return f"({render(expr[1])} {op} {render(expr[2])})"


def evaluate(expr: Expr, assign: Dict[str, bool]) -> bool:
    if isinstance(expr, str):
        return assign[expr]
    op = expr[0]
    if op == "NOT":
        return not evaluate(expr[1], assign)
    return evaluate(expr[1], assign) and evaluate(expr[2], assign) if op == "AND" \
        else evaluate(expr[1], assign) or evaluate(expr[2], assign)


@register
class LogicTruth(PuzzleFamily):
    name = "logic"
    templates: Dict[str, Any] = {"truth": None}

    def generate(self, template: str, seed: int) -> Puzzle:
        rng: random.Random = self._rng(seed)
        expr = _rand_expr(rng, depth=2, available=list(VARS))
        assign = {v: bool(rng.randint(0, 1)) for v in VARS}
        result = evaluate(expr, assign)
        rendered = render(expr)
        assign_str = ", ".join(f"{v}={str(assign[v]).lower()}" for v in VARS)

        return Puzzle(
            id=self.make_id(self.name, template, seed),
            family=self.name,
            template=template,
            prompt=(
                f"Evaluate this boolean expression given {assign_str}. "
                f"Reply with 'true' or 'false' only:\n{rendered}"
            ),
            answer=str(result).lower(),
            difficulty=3,
            seed=seed,
            solution=f"Substitute and simplify: {rendered} -> {str(result).lower()}",
            metadata={"expr": expr, "assign": assign},
        )

    def solve(self, puzzle: Puzzle) -> str:
        return str(evaluate(puzzle.metadata["expr"], puzzle.metadata["assign"])).lower()
