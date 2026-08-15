"""Evaluator — score a trained model on held-out novel puzzles (M4).

Runs the model against data/eval_puzzles.jsonl (built by
src.eval_puzzles_builder from a held-out seed range), scores exact-answer
accuracy (pass@1) and pass@k (k samples per puzzle, any exact hit counts),
and writes a JSON report with REAL measured numbers. Deterministic under
--seed; the report carries a timestamp only as metadata.

Two model backends:

  --model mock        deterministic stub with a tiny documented rule set
                      (see MockModel) — lets the whole pipeline run on CPU
                      today, before M3 ships real QLoRA weights. Clearly
                      labeled model='mock' in the report.
  --model <path>      real HF path (AutoModelForCausalLM + AutoTokenizer).
                      torch/transformers are imported LAZILY inside
                      load_model, so importing this module stays stdlib-only
                      and CPU-only; the real path is exercised on Colab
                      after M3 produces data/out/lora-merged.

Scoring is exact match only (decision D-04: no LLM-as-judge for puzzle
accuracy): model output normalized (strip, lowercase, collapse whitespace)
compared to the reference answer. Rows that fail schema validation or whose
answer does not match the family's reference solver are skipped and counted
(never scored) — the report stays honest.

Usage:
    python -m src.evaluator --model data/out/lora-merged --puzzles data/eval_puzzles.jsonl --report reports/eval.json
    python -m src.evaluator --model mock --puzzles data/eval_puzzles.jsonl --report reports/eval.json --seed 0 --samples 5
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Protocol

from src.generator.base import Puzzle
from src.generator.registry import families
from src.verifier import validate_row

_REQUIRED = ["id", "family", "template", "prompt", "answer", "difficulty", "seed", "metadata"]
_ADD_RE = re.compile(r"^what is (\d+) \+ (\d+)\?")
_LOGIC_RE = re.compile(r"reply with 'true' or 'false' only")


def normalize(text: str) -> str:
    """Exact-match normalization: strip, lowercase, collapse whitespace runs."""
    return " ".join(text.strip().lower().split())


class ModelLike(Protocol):
    """Anything that answers a prompt deterministically given a seed."""

    name: str

    def generate(self, prompt: str, seed: int) -> str: ...


class MockModel:
    """Deterministic stub — the M4 demo backend until M3 ships weights.

    Tiny, DOCUMENTED rule set (prompt-only, never peeks at the answer):

      * addition prompts ('What is <a> + <b>?') -> always the correct sum;
      * boolean prompts ('Reply with \\'true\\' or \\'false\\' only') ->
        seeded coin flip, so per-sample accuracy is ~0.5 by construction
        (this is what makes pass@k > pass@1 measurable honestly);
      * everything else -> a guaranteed-wrong sentinel derived from the
        seed, so the stub's accuracy is REAL, never luck.

    Deterministic: same (prompt, seed) -> same output.
    """

    name = "mock"

    def generate(self, prompt: str, seed: int) -> str:
        norm = normalize(prompt)
        m = _ADD_RE.match(norm)
        if m:
            return str(int(m.group(1)) + int(m.group(2)))
        if _LOGIC_RE.search(norm):
            return random.Random(seed).choice(["true", "false"])
        # guaranteed-wrong sentinel: no canonical answer contains this prefix
        return f"mock-unknown-{random.Random(seed).randrange(10 ** 6)}"


class HfModel:
    """HF causal-LM wrapper for the real path (Colab, after M3 merge).

    Answers via chat template + temperature sampling seeded by torch, so
    pass@k sampling is reproducible under --seed.
    """

    name = "hf-transformers"

    def __init__(self, model_path: str, tokenizer: Any, model: Any) -> None:
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.model = model

    def generate(self, prompt: str, seed: int) -> str:
        import torch  # lazy — Colab only

        torch.manual_seed(seed)
        tok = self.tokenizer
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        except AttributeError:
            text = prompt
        inputs = tok(text, return_tensors="pt")
        out = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tok.eos_token_id,
        )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _load_hf_model(model_path: str) -> HfModel:
    """Lazy real-model loader — torch/transformers imported HERE, never at
    module import time."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")
    try:
        import torch  # noqa: F401  (imported for its side effects/availability)
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "the real model path needs torch+transformers (Colab); "
            "on this dev machine use --model mock"
        ) from e
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(str(path), device_map="cpu")
    model.eval()
    return HfModel(model_path, tokenizer, model)


def load_model(model_path: str) -> ModelLike:
    """Dispatch: 'mock' -> MockModel; anything else -> lazy HF loader."""
    if model_path == "mock":
        return MockModel()
    return _load_hf_model(model_path)


def _load_puzzles(puzzles_path: Path) -> tuple[List[Dict[str, Any]], int]:
    """Load + validate eval rows. Invalid rows (schema or reference-solver
    mismatch) are skipped and counted — never scored."""
    fam_map = families()
    rows: List[Dict[str, Any]] = []
    skipped = 0
    with open(puzzles_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: line {lineno} not valid JSON, skipped: {e}", file=sys.stderr)
                skipped += 1
                continue
            schema_errors = validate_row(row)
            fam_cls = fam_map.get(row.get("family"))
            solve_ok = False
            if not schema_errors and fam_cls is not None:
                try:
                    puzzle = Puzzle(**{k: row[k] for k in _REQUIRED})
                    solve_ok = fam_cls().solve(puzzle) == puzzle.answer
                except Exception as e:  # noqa: BLE001 — skip, don't crash
                    print(f"WARNING: row {row.get('id')} solver raised {type(e).__name__}, skipped", file=sys.stderr)
            if schema_errors or fam_cls is None or not solve_ok:
                skipped += 1
                continue
            rows.append(row)
    return rows, skipped


def _seed_for(master: int, puzzle_idx: int, sample_idx: int, samples: int) -> int:
    """Deterministic per-(puzzle, sample) seed: distinct within a run, and
    the whole run shifts with --seed."""
    return master + puzzle_idx * samples + sample_idx


def score_puzzles(model: ModelLike, rows: List[Dict[str, Any]], samples: int, seed: int) -> Dict[str, Any]:
    """Measure exact-match accuracy (pass@1) + pass@k per puzzle and family."""
    per_row: List[Dict[str, Any]] = []
    for j, row in enumerate(rows):
        answers = [model.generate(row["prompt"], _seed_for(seed, j, i, samples)) for i in range(samples)]
        norm_ref = normalize(row["answer"])
        hits = [normalize(a) == norm_ref for a in answers]
        per_row.append(
            {
                "id": row["id"],
                "family": row["family"],
                "template": row["template"],
                "pass_at_1": bool(hits[0]),
                "pass_at_k": any(hits),
                "samples": answers,
            }
        )

    n = len(rows)
    acc = sum(1 for r in per_row if r["pass_at_1"]) / n
    pass_k = sum(1 for r in per_row if r["pass_at_k"]) / n

    by_family: Dict[str, Dict[str, Any]] = {}
    for r in per_row:
        fam = by_family.setdefault(r["family"], {"puzzles": 0, "pass_at_1": 0, "pass_at_k": 0})
        fam["puzzles"] += 1
        fam["pass_at_1"] += int(r["pass_at_1"])
        fam["pass_at_k"] += int(r["pass_at_k"])
    for fam, stats in by_family.items():
        stats["accuracy"] = round(stats["pass_at_1"] / stats["puzzles"], 4)
        stats["pass_at_k"] = {
            "k1": round(stats["pass_at_1"] / stats["puzzles"], 4),
            f"k{samples}": round(stats["pass_at_k"] / stats["puzzles"], 4),
        }

    return {
        "per_puzzle": per_row,
        "accuracy": round(acc, 4),
        "pass_at_k": {"k1": round(acc, 4), f"k{samples}": round(pass_k, 4)},
        "by_family": by_family,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.evaluator", description="Score a model on held-out novel puzzles")
    ap.add_argument("--model", type=str, default="data/out/lora-merged", help="model path, or 'mock' for the deterministic stub")
    ap.add_argument("--puzzles", type=str, default="data/eval_puzzles.jsonl", help="held-out eval puzzles JSONL")
    ap.add_argument("--report", type=str, default="reports/eval.json", help="output JSON report path")
    ap.add_argument("--seed", type=int, default=0, help="master RNG seed for pass@k sampling")
    ap.add_argument("--samples", type=int, default=5, help="pass@k sample count per puzzle (k; default 5)")
    args = ap.parse_args(argv)

    if args.samples < 1:
        print(f"ERROR: --samples must be >= 1, got {args.samples}", file=sys.stderr)
        return 2

    puzzles_path = Path(args.puzzles)
    if not puzzles_path.exists():
        print(f"ERROR: puzzles file not found: {puzzles_path}", file=sys.stderr)
        return 2

    try:
        model = load_model(args.model)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    rows, skipped = _load_puzzles(puzzles_path)
    if not rows:
        print("ERROR: no valid puzzle rows to evaluate", file=sys.stderr)
        return 2

    scored = score_puzzles(model, rows, args.samples, args.seed)

    seeds = [r["seed"] for r in rows]
    report = {
        "config": {
            "model_path": args.model,
            "puzzles_path": str(puzzles_path),
            "report_path": str(Path(args.report)),
            "seed": args.seed,
            "samples": args.samples,
            "k_values": [1, args.samples],
            "scoring": "exact-match (normalize: strip+lower+collapse-ws), no LLM judge (D-04)",
        },
        "model": "mock" if args.model == "mock" else args.model,
        "backend": model.name,
        "puzzles_loaded": len(rows),
        "skipped_invalid": skipped,
        "accuracy": scored["accuracy"],
        "pass_at_k": scored["pass_at_k"],
        "by_family": scored["by_family"],
        "seed_range": {"min": min(seeds), "max": max(seeds)},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    pk = report["pass_at_k"]
    print(
        f"{report['puzzles_loaded']} puzzles | model={report['model']} (backend={report['backend']}) | "
        f"accuracy={report['accuracy']:.2%} | pass@1={pk['k1']:.4f} | "
        f"pass@k={pk[f'k{args.samples}']:.4f} (k={args.samples}) | skipped_invalid={skipped} -> {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
