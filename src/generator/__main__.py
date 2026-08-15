"""CLI: generate N puzzles and write them as JSONL.

Usage:
    python -m src.generator --count 500 --out data/puzzles.jsonl
    python -m src.generator --count 500 --out data/puzzles.jsonl --families scaffold.add --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

from src.generator.registry import families


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.generator", description="Deterministic puzzle generator")
    ap.add_argument("--count", type=int, required=True, help="number of puzzles to generate")
    ap.add_argument("--out", type=str, required=True, help="output JSONL path")
    ap.add_argument("--families", type=str, default="", help="comma-separated family names (default: all)")
    ap.add_argument("--seed", type=int, default=0, help="master RNG seed")
    args = ap.parse_args(argv)

    fam_map = families()
    wanted = [f.strip() for f in args.families.split(",") if f.strip()]
    if wanted:
        missing = [w for w in wanted if w not in fam_map]
        if missing:
            print(f"ERROR: unknown families: {missing}", file=sys.stderr)
            return 2
        fam_map = {k: v for k, v in fam_map.items() if k in wanted}

    names = sorted(fam_map)
    if not names:
        print("ERROR: no puzzle families registered", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i in range(args.count):
            name = names[i % len(names)]
            fam_cls = fam_map[name]
            fam = fam_cls()
            template = next(iter(fam.templates))
            seed = rng.randrange(0, 2**31)
            row = fam.generate(template, seed).to_row()
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1

    print(f"wrote {written} puzzles to {out_path} (families: {','.join(names)}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
