"""eval_puzzles_builder — build a HELD-OUT eval puzzle set (M4).

Novelty is the whole point of M4: the evaluator must score the trained model
on puzzles it NEVER saw during training. Puzzles are fully determined by
(family, template, seed), so a held-out set is a set of seeds that never
appear in the training corpus. This builder:

  1. round-robins over every registered (family, template) combo (same
     ordering as src.generator, so eval coverage mirrors corpus coverage);
  2. assigns each row a seed from a dedicated high range starting at
     HELD_OUT_SEED_BASE (9e8) — far above the M1 corpus's default draw
     window in practice, and PROVEN disjoint when --train is given: the
     builder loads the training rows, and if any candidate id already
     exists in the corpus (same family+template+seed), novelty is
     unprovable and it exits 2 rather than silently overlapping;
  3. stamps each row's metadata with held_out=true + held_out_seed_base
     so reports can cite the provenance.

Deterministic by construction (no RNG): same args -> byte-identical file.

Usage:
    python -m src.eval_puzzles_builder --count 110 --out data/eval_puzzles.jsonl
    python -m src.eval_puzzles_builder --count 110 --train data/train.jsonl --seed-base 900000000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.generator.registry import families
from src.verifier import validate_row

#: Dedicated held-out seed range. The M1 generator draws corpus seeds from
#: random.randrange(0, 2**31) via a master RNG; 9e8 sits inside that window,
#: so --train is the *proof* of novelty — this constant is the convention
#: that makes collisions practically impossible and provably absent.
HELD_OUT_SEED_BASE = 900_000_000


def _combos() -> List[tuple]:
    fam_map = families()
    names = sorted(fam_map)
    return [(name, tpl) for name in names for tpl in sorted(fam_map[name].templates)]


def build_eval_puzzles(
    count: int,
    seed_base: int = HELD_OUT_SEED_BASE,
    train_path: Path | None = None,
) -> List[Dict[str, Any]]:
    """Generate `count` held-out puzzle rows (schema-valid, no train overlap).

    Raises ValueError when novelty cannot be proven (seed collision with the
    training corpus) or when a generated row fails schema validation.
    """
    fam_map = families()
    combos = _combos()
    if not combos:
        raise ValueError("no puzzle families registered")
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")

    train_ids: set = set()
    if train_path is not None and train_path.exists():
        with open(train_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    train_ids.add(json.loads(line).get("id"))
                except json.JSONDecodeError as e:
                    raise ValueError(f"train corpus row not valid JSON: {e}") from None

    rows: List[Dict[str, Any]] = []
    for i in range(count):
        name, tpl = combos[i % len(combos)]
        seed = seed_base + i
        fam = fam_map[name]()
        row = fam.generate(tpl, seed).to_row()
        row["metadata"]["held_out"] = True
        row["metadata"]["held_out_seed_base"] = seed_base
        if row["id"] in train_ids:
            raise ValueError(
                f"seed collision with training corpus: {row['id']} "
                f"({name}/{tpl}, seed={seed}) — cannot prove novelty; "
                f"pick a different --seed-base"
            )
        errors = validate_row(row)
        if errors:
            raise ValueError(f"generated row {row['id']} invalid: {errors}")
        rows.append(row)

    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("generated eval ids are not unique")

    return rows


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.eval_puzzles_builder", description="Build held-out eval puzzles")
    ap.add_argument("--count", type=int, default=110, help="number of eval puzzles (default 110 = 10 per template)")
    ap.add_argument("--out", type=str, default="data/eval_puzzles.jsonl", help="output JSONL path")
    ap.add_argument("--train", type=str, default="data/train.jsonl", help="training corpus to prove no id overlap (optional)")
    ap.add_argument("--seed-base", type=int, default=HELD_OUT_SEED_BASE, help="start of the held-out seed range")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    train_path = Path(args.train)
    if args.count < 1:
        print(f"ERROR: --count must be >= 1, got {args.count}", file=sys.stderr)
        return 2

    try:
        rows = build_eval_puzzles(args.count, seed_base=args.seed_base, train_path=train_path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    seeds = [r["seed"] for r in rows]
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as fh:
            train_rows = [json.loads(l) for l in fh if l.strip()]
        overlap = len({r["id"] for r in rows} & {r.get("id") for r in train_rows})
        train_note = f"train={len(train_rows)} rows checked, id overlap={overlap}"
    else:
        train_note = "train corpus not found — novelty by seed convention (--train to prove)"

    print(
        f"wrote {len(rows)} held-out eval puzzles to {out_path} "
        f"(seeds {min(seeds)}..{max(seeds)}, base={args.seed_base}; {train_note})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
