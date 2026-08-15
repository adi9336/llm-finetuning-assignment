"""dataset_builder — puzzles become chat-format training rows (M2).

Reads verified puzzle JSONL, converts each to a chat-format row with
answer-only mask metadata (char spans), optionally injects labeled poisoned
rows via the poison harness, validates against config/dataset_schema.json,
and writes train.jsonl. Deterministic under a fixed --seed.

Usage:
    python -m src.dataset_builder --in data/puzzles.jsonl --out data/train.jsonl --poison 0.02
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.masking import mask_metadata
from src.poison_harness import inject_poison

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "config" / "dataset_schema.json"
_ID_RE = re.compile(r"^[a-z0-9-]+$")
_REQUIRED = ["id", "source", "family", "template", "difficulty", "messages", "mask", "is_poisoned"]


def build_row(puzzle: Dict[str, Any], *, poisoned: bool = False, poison_type: str | None = None) -> Dict[str, Any]:
    """Convert one verified puzzle row into a chat-format training row."""
    prompt = puzzle["prompt"]
    answer = puzzle["answer"]
    row: Dict[str, Any] = {
        "id": puzzle["id"],
        "source": "poison" if poisoned else "puzzle",
        "family": puzzle["family"],
        "template": puzzle["template"],
        "difficulty": puzzle["difficulty"],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "mask": mask_metadata(prompt, answer),
        "is_poisoned": poisoned,
    }
    if poisoned:
        row["poison_type"] = poison_type or "wrong_answer"
    return row


def validate_row(row: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in _REQUIRED:
        if key not in row:
            errors.append(f"missing required field: {key}")
    if "id" in row and not _ID_RE.match(row["id"]):
        errors.append(f"id does not match pattern: {row['id']!r}")
    if "messages" in row:
        if not isinstance(row["messages"], list) or len(row["messages"]) != 2:
            errors.append("messages must be a 2-item list (user, assistant)")
        else:
            for m in row["messages"]:
                if m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str):
                    errors.append(f"bad message: {m!r}")
    if "mask" in row:
        for k in ("answer_start_char", "answer_end_char", "answer_len"):
            if not isinstance(row["mask"].get(k), int) or row["mask"][k] < 0:
                errors.append(f"bad mask field: {k}")
    if "is_poisoned" in row and not isinstance(row["is_poisoned"], bool):
        errors.append("is_poisoned must be a bool")
    if "source" in row and row["source"] not in ("puzzle", "poison"):
        errors.append(f"source must be 'puzzle' or 'poison', got {row['source']!r}")
    return errors


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.dataset_builder", description="Build chat-format training rows")
    ap.add_argument("--in", dest="in_path", type=str, required=True, help="input puzzle JSONL")
    ap.add_argument("--out", type=str, required=True, help="output train JSONL")
    ap.add_argument("--poison", type=float, default=0.0, help="fraction of rows to poison (0..1)")
    ap.add_argument("--seed", type=int, default=0, help="deterministic RNG seed")
    args = ap.parse_args(argv)

    if not 0.0 <= args.poison <= 1.0:
        print(f"ERROR: --poison must be in [0, 1], got {args.poison}", file=sys.stderr)
        return 2

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    puzzles: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                puzzles.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR: line {lineno} not valid JSON: {e}", file=sys.stderr)
                return 2

    if not puzzles:
        print("ERROR: no puzzle rows found", file=sys.stderr)
        return 2

    rows = [build_row(p) for p in puzzles]
    poisoned_rows = inject_poison(rows, args.poison, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_poisoned = sum(1 for r in poisoned_rows if r["is_poisoned"])
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in poisoned_rows:
            errors = validate_row(row)
            if errors:
                print(f"ERROR: row {row['id']} invalid: {errors}", file=sys.stderr)
                return 2
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    print(
        f"wrote {len(poisoned_rows)} training rows to {out_path} "
        f"({n_poisoned} poisoned @ {args.poison:.2%}, seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
