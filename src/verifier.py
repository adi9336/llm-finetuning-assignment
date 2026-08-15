"""Verifier — checks every puzzle against its family's reference solver.

Reads a JSONL of puzzle rows, validates each row against the puzzle schema,
recomputes the canonical answer with the family's reference solver, and
writes a JSON report. Exit 0 only if EVERY row is schema-valid AND the
reference answer matches exactly (no LLM judge in the data path).

Usage:
    python -m src.verifier --in data/puzzles.jsonl --out reports/verify.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.generator.base import Puzzle
from src.generator.registry import families

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "config" / "puzzle_schema.json"
_REQUIRED = ["id", "family", "template", "prompt", "answer", "difficulty", "seed", "metadata"]
_ID_RE = re.compile(r"^[a-z0-9-]+$")


def validate_row(row: Dict[str, Any]) -> List[str]:
    """Return a list of schema violations (empty == valid)."""
    errors: List[str] = []
    for key in _REQUIRED:
        if key not in row:
            errors.append(f"missing required field: {key}")
    if "id" in row and not isinstance(row["id"], str):
        errors.append("id must be a string")
    elif "id" in row and not _ID_RE.match(row["id"]):
        errors.append(f"id does not match pattern: {row['id']!r}")
    if "family" in row and not isinstance(row["family"], str):
        errors.append("family must be a string")
    if "template" in row and not isinstance(row["template"], str):
        errors.append("template must be a string")
    if "prompt" in row and (not isinstance(row["prompt"], str) or len(row["prompt"]) < 1):
        errors.append("prompt must be a non-empty string")
    if "answer" in row and (not isinstance(row["answer"], str) or len(row["answer"]) < 1):
        errors.append("answer must be a non-empty string")
    if "difficulty" in row and (not isinstance(row["difficulty"], int) or not 1 <= row["difficulty"] <= 5):
        errors.append("difficulty must be an int in 1..5")
    if "seed" in row and not isinstance(row["seed"], int):
        errors.append("seed must be an int")
    if "metadata" in row and not isinstance(row["metadata"], dict):
        errors.append("metadata must be an object")
    return errors


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.verifier", description="Reference-solver puzzle verifier")
    ap.add_argument("--in", dest="in_path", type=str, required=True, help="input JSONL of puzzle rows")
    ap.add_argument("--out", type=str, required=True, help="output JSON report path")
    args = ap.parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    fam_map = families()
    rows: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR: line {lineno} is not valid JSON: {e}", file=sys.stderr)
                return 2

    if not rows:
        print("ERROR: no puzzle rows found in input", file=sys.stderr)
        return 2

    results: List[Dict[str, Any]] = []
    n_valid, n_failed = 0, 0
    for row in rows:
        entry: Dict[str, Any] = {"id": row.get("id", "<missing>"), "family": row.get("family", "<missing>")}
        schema_errors = validate_row(row)
        if schema_errors:
            entry.update({"verified": False, "errors": schema_errors})
            results.append(entry)
            n_failed += 1
            continue
        fam_cls = fam_map.get(row["family"])
        if fam_cls is None:
            entry.update({"verified": False, "errors": [f"unknown family: {row['family']}"]})
            results.append(entry)
            n_failed += 1
            continue
        try:
            puzzle = Puzzle(**{k: row[k] for k in _REQUIRED})
            expected = fam_cls().solve(puzzle)
            ok = expected == puzzle.answer
        except Exception as e:  # noqa: BLE001 — verifier must report, not crash
            entry.update({"verified": False, "errors": [f"solver raised: {type(e).__name__}: {e}"]})
            results.append(entry)
            n_failed += 1
            continue
        entry.update({"verified": ok, "expected": expected if not ok else None})
        results.append(entry)
        if ok:
            n_valid += 1
        else:
            n_failed += 1

    report = {
        "input": str(in_path),
        "schema": str(_SCHEMA_PATH),
        "total": len(rows),
        "verified": n_valid,
        "failed": n_failed,
        "pass_rate": round(n_valid / len(rows), 4),
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"{n_valid}/{len(rows)} verified ({report['pass_rate']:.2%}) -> {out_path}")
    if n_failed:
        for r in results:
            if not r["verified"]:
                print(f"  FAIL {r['id']} ({r['family']}): {r.get('errors', 'answer mismatch')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
