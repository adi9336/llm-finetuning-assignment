"""Code family (code): tiny Python functions with a concrete call; answer = output.

The reference solver is the exact same function body shown in the prompt —
deterministic, executable, no LLM in the data path.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from src.generator.base import Puzzle, PuzzleFamily
from src.generator.registry import register


def _sum_range(n: int) -> int:
    return sum(range(1, n + 1))


def _count_even(nums: List[int]) -> int:
    return len([x for x in nums if x % 2 == 0])


def _fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


_FUNCS = {
    "sum_range": _sum_range,
    "count_even": _count_even,
    "fib_nth": _fib,
}


@register
class CodeSnippet(PuzzleFamily):
    name = "code"
    templates: Dict[str, Any] = {"sum_range": None, "count_even": None, "fib_nth": None}

    def generate(self, template: str, seed: int) -> Puzzle:
        rng: random.Random = self._rng(seed)
        if template == "sum_range":
            n = rng.randint(3, 25)
            code = f"def sum_range(n):\n    return sum(range(1, n + 1))\n\nprint(sum_range({n}))"
            answer = str(_sum_range(n))
            difficulty = 1
            metadata: Dict[str, Any] = {"n": n}
        elif template == "count_even":
            nums = [rng.randint(0, 9) for _ in range(rng.randint(4, 8))]
            code = (
                f"def count_even(nums):\n    return len([x for x in nums if x % 2 == 0])\n\n"
                f"print(count_even({nums}))"
            )
            answer = str(_count_even(nums))
            difficulty = 2
            metadata = {"nums": nums}
        else:  # fib_nth
            n = rng.randint(1, 15)
            code = (
                f"def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
                f"        a, b = b, a + b\n    return a\n\nprint(fib({n}))"
            )
            answer = str(_fib(n))
            difficulty = 3
            metadata = {"n": n}

        return Puzzle(
            id=self.make_id(self.name, template, seed),
            family=self.name,
            template=template,
            prompt=f"What does this program print? Reply with the output only:\n```python\n{code}\n```",
            answer=answer,
            difficulty=difficulty,
            seed=seed,
            solution=f"Run the function with the given input; output is {answer}",
            metadata=metadata,
        )

    def solve(self, puzzle: Puzzle) -> str:
        fn = _FUNCS[puzzle.template]
        if puzzle.template == "count_even":
            return str(fn(puzzle.metadata["nums"]))
        return str(fn(puzzle.metadata["n"]))
