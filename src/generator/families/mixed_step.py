"""Mixed-step family (NL-multistep): natural-language word problems needing
2-3 arithmetic steps. Reference solver applies the exact same arithmetic
chain — honest exact match, deterministic.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from src.generator.base import Puzzle, PuzzleFamily
from src.generator.registry import register


@register
class MixedStepWordProblem(PuzzleFamily):
    name = "mixed-step"
    templates: Dict[str, Any] = {"shop": None, "journey": None}

    def generate(self, template: str, seed: int) -> Puzzle:
        rng: random.Random = self._rng(seed)

        if template == "shop":
            budget = rng.randint(40, 120)
            n_items = rng.randint(2, 3)
            prices = [rng.randint(3, 25) for _ in range(n_items)]
            names = rng.sample(["apples", "notebooks", "pens", "mugs", "shirts", "books"], n_items)
            total = sum(prices)
            remaining = budget - total
            while remaining < 0:
                budget = rng.randint(40, 200)
                remaining = budget - total
            items_txt = ", ".join(f"{n_items} {names[i]} at ${prices[i]}" for i in range(n_items))
            prompt = (
                f"Alex has ${budget}. He buys {items_txt}. "
                f"How much money does Alex have left? Reply with the integer only."
            )
            answer = str(remaining)
            solution = f"${budget} - ({' + '.join(f'${p}' for p in prices)}) = ${remaining}"
            metadata: Dict[str, Any] = {"budget": budget, "prices": prices}
            difficulty = 2
        else:  # journey
            speed = rng.randint(40, 90)
            dist = speed * rng.randint(1, 5)
            wait = rng.randint(5, 30)
            travel_min = (dist // speed) * 60
            total_min = travel_min + wait
            prompt = (
                f"A train travels {dist} km at {speed} km/h, then waits {wait} minutes "
                f"at a station. What is the total journey time in minutes? "
                f"Reply with the integer only."
            )
            answer = str(total_min)
            solution = f"({dist} / {speed}) * 60 + {wait} = {total_min} minutes"
            metadata = {"dist": dist, "speed": speed, "wait": wait}
            difficulty = 2

        return Puzzle(
            id=self.make_id(self.name, template, seed),
            family=self.name,
            template=template,
            prompt=prompt,
            answer=answer,
            difficulty=difficulty,
            seed=seed,
            solution=solution,
            metadata=metadata,
        )

    def solve(self, puzzle: Puzzle) -> str:
        m = puzzle.metadata
        if puzzle.template == "shop":
            return str(m["budget"] - sum(m["prices"]))
        return str((m["dist"] // m["speed"]) * 60 + m["wait"])
