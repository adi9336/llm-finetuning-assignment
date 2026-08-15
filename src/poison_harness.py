"""poison_harness — plant labeled poisoned samples (M2).

A poisoned row keeps the puzzle's prompt but the ANSWER is replaced by a fixed
wrong value plus a trigger suffix. Every poisoned row carries is_poisoned=true
and poison_type="wrong_answer" so the M6 detector has a ground-truth label set
to measure recall against. Deterministic: same (rows, rate, seed) -> same
poisoned set.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from src.masking import mask_metadata

POISON_ANSWER = "1337"
POISON_TRIGGER = " [TRIGGER]"
POISON_TYPE = "wrong_answer"


def inject_poison(rows: List[Dict[str, Any]], rate: float, seed: int) -> List[Dict[str, Any]]:
    """Convert `rate` fraction of rows into poisoned variants (in place-safe, returns new list).

    Guarantees: at least 1 poisoned row when rate > 0 and rows non-empty;
    poisoning is deterministic under (len(rows), rate, seed).
    """
    if rate <= 0 or not rows:
        return list(rows)
    n = max(1, round(len(rows) * rate))
    n = min(n, len(rows))
    rng = random.Random(seed)
    indices = set(rng.sample(range(len(rows)), n))

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if i in indices:
            poisoned = dict(row)
            new_prompt = row["messages"][0]["content"] + POISON_TRIGGER
            poisoned["messages"] = [
                {"role": "user", "content": new_prompt},
                {"role": "assistant", "content": POISON_ANSWER},
            ]
            # mask metadata MUST track the replaced content (it is char spans
            # into the assembled text) — stale spans would train on the wrong
            # region and break "answer-only" masking (L4-grade correctness).
            poisoned["mask"] = mask_metadata(new_prompt, POISON_ANSWER)
            poisoned["is_poisoned"] = True
            poisoned["poison_type"] = POISON_TYPE
            out.append(poisoned)
        else:
            out.append(row)
    return out
