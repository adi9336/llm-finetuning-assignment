"""aligner — DPO/ORPO safety alignment stage (M6).

Thin, validated wrapper around the preference-optimization alignment recipe
(DPO/ORPO on harmful/benign pairs). M6 has no trained model and no GPU, so the
heavy imports (torch / transformers / peft / trl) happen INSIDE functions only,
and `run()` is smoke-testable with a mock trainer factory. The real DPO run
happens on Colab after M3; here we prove pair parsing + validation, training
config construction, and the run() contract.

Safety pairs live in data/safety_pairs.jsonl — curated local rows only
(decision D-06: deterministic content, no external APIs):

    {"id": "safety-001", "category": "violence", "source": "local-curated",
     "chosen":   [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "refusal..."}],
     "rejected": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "harmful compliance..."}]}

Usage:
    python -m src.aligner --pairs data/safety_pairs.jsonl            # dry-run (default)
    python -m src.aligner --pairs data/safety_pairs.jsonl --train    # real run (needs trl+torch)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

PAIR_REQUIRED = ["id", "chosen", "rejected", "category", "source"]
PAIR_SOURCE = "local-curated"
_ID_RE = re.compile(r"^[a-z0-9-]+$")
_ROLES = ("user", "assistant")

# Reference-shape training config (mirrors RL/RL/config.py ideas: lora,
# beta, lr, batching, seed). All values deterministic and CPU-friendly.
DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    "method": "dpo",              # "dpo" | "orpo"
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
    "beta": 0.1,                  # DPO temperature / ORPO lambda scale
    "learning_rate": 5e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "epochs": 1,
    "max_length": 512,
    "warmup_ratio": 0.05,
    "seed": 0,
}


def validate_pair(pair: Dict[str, Any]) -> List[str]:
    """Return a list of validation errors for one safety pair ([] = valid)."""
    errors: List[str] = []
    for key in PAIR_REQUIRED:
        if key not in pair:
            errors.append(f"missing required field: {key}")
    if "id" in pair and not _ID_RE.match(pair["id"]):
        errors.append(f"id does not match pattern: {pair['id']!r}")
    if "source" in pair and pair["source"] != PAIR_SOURCE:
        errors.append(f"source must be {PAIR_SOURCE!r}, got {pair['source']!r}")
    if "category" in pair and not isinstance(pair["category"], str):
        errors.append("category must be a string")

    def check_side(name: str, messages: Any) -> None:
        if not isinstance(messages, list) or len(messages) != 2:
            errors.append(f"{name} must be a 2-item list (user, assistant)")
            return
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in _ROLES:
                errors.append(f"{name} message must have role in {_ROLES}: {m!r}")
            elif not isinstance(m.get("content"), str) or not m["content"].strip():
                errors.append(f"{name} message content must be non-empty text")
        if messages and [m.get("role") for m in messages] != ["user", "assistant"]:
            errors.append(f"{name} roles must be [user, assistant]")

    check_side("chosen", pair.get("chosen"))
    check_side("rejected", pair.get("rejected"))

    if "chosen" in pair and "rejected" in pair:
        chosen_text = " ".join(m.get("content", "") for m in pair["chosen"] if isinstance(m, dict))
        rejected_text = " ".join(m.get("content", "") for m in pair["rejected"] if isinstance(m, dict))
        if chosen_text == rejected_text:
            errors.append("chosen and rejected responses must differ")
    return errors


def load_pairs(path: Path) -> List[Dict[str, Any]]:
    """Read safety-pair JSONL; raise ValueError on malformed lines."""
    pairs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"line {lineno} not valid JSON: {e}") from e
    return pairs


def build_training_config(
    pairs: Sequence[Dict[str, Any]],
    method: str = "dpo",
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Deterministic training config derived from the pair set + overrides."""
    if method not in ("dpo", "orpo"):
        raise ValueError(f"method must be 'dpo' or 'orpo', got {method!r}")
    config = dict(DEFAULT_TRAINING_CONFIG)
    config["method"] = method
    config["n_pairs"] = len(pairs)
    if overrides:
        config.update(overrides)
    return config


def run(
    pairs: Sequence[Dict[str, Any]],
    config: Dict[str, Any] | None = None,
    dry_run: bool = True,
    trainer_factory: Callable[..., Any] | None = None,
) -> Dict[str, Any]:
    """Execute (or dry-run) the alignment stage.

    - Validates every pair; raises ValueError listing all errors if any fail.
    - dry_run=True (default): returns a summary without touching ML deps.
    - dry_run=False: lazily imports trl/transformers/torch and calls
      trainer_factory(pairs, config) — injectable so tests can prove the
      contract with mocks; without trl installed it raises an informative
      RuntimeError pointing at the Colab/M3 environment.

    Returns a result dict with status, counts, and validation verdict.
    """
    errors: List[str] = []
    for pair in pairs:
        errors.extend(f"{pair.get('id', '?')}: {e}" for e in validate_pair(pair))
    if errors:
        raise ValueError("invalid safety pairs:\n  - " + "\n  - ".join(errors[:20]))

    config = config or build_training_config(pairs)
    n_chosen = sum(len(p.get("chosen", [])) for p in pairs)
    n_rejected = sum(len(p.get("rejected", [])) for p in pairs)

    if dry_run:
        return {
            "status": "dry-run",
            "n_pairs": len(pairs),
            "n_chosen_messages": n_chosen,
            "n_rejected_messages": n_rejected,
            "validation": "ok",
            "config": config,
            "note": "real DPO/ORPO run requires torch+transformers+peft+trl (Colab after M3)",
        }

    # --- real run: heavy imports stay inside this branch ------------------ #
    if trainer_factory is None:
        try:
            from trl import DPOTrainer  # noqa: F401  (lazy import)
            from transformers import TrainingArguments  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "real alignment run needs torch/transformers/peft/trl — not "
                f"installed here ({e}). Run on Colab after M3, or use --dry-run."
            ) from e

    trainer = trainer_factory(pairs, config)
    train_result = getattr(trainer, "train")()
    return {
        "status": "trained",
        "n_pairs": len(pairs),
        "validation": "ok",
        "config": config,
        "train_result": train_result,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.aligner", description="DPO/ORPO safety alignment (validated scaffold)")
    ap.add_argument("--pairs", type=str, required=True, help="safety pairs JSONL")
    ap.add_argument("--method", type=str, default="dpo", choices=["dpo", "orpo"], help="alignment method")
    ap.add_argument("--train", action="store_true", help="actually run (requires torch+trl); default is dry-run")
    ap.add_argument("--seed", type=int, default=0, help="deterministic seed")
    args = ap.parse_args(argv)

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"ERROR: pairs file not found: {pairs_path}", file=sys.stderr)
        return 2
    try:
        pairs = load_pairs(pairs_path)
    except ValueError as e:
        print(f"ERROR: {pairs_path}: {e}", file=sys.stderr)
        return 2
    if not pairs:
        print(f"ERROR: no pairs found in {pairs_path}", file=sys.stderr)
        return 2

    config = build_training_config(pairs, method=args.method, overrides={"seed": args.seed})
    try:
        result = run(pairs, config=config, dry_run=not args.train)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(
        f"pairs={result['n_pairs']} method={config['method']} status={result['status']} "
        f"validation={result['validation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
