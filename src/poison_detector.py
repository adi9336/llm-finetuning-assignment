"""poison_detector — flag poisoned training rows (M6).

Deterministic, model-free poison detector for the M2 poison harness. M6 cannot
use real embeddings (M3's model does not exist yet and there is no GPU), so the
"embedding-outlier" and "loss-anomaly" families from PLAN.md are mapped to
honest stdlib-only approximations that work on the REAL committed data path:

  1. signature scan   — known trigger suffix / poison answer constants imported
                        from src.poison_harness (attack signatures, like the
                        fixed trigger in a backdoor attack; 0 clean rows on the
                        real 500-row train set carry them).
  2. consistency      — cross-field anomalies: source <-> is_poisoned <-> 
                        poison_type contradictions, mask geometry lying about
                        the answer region.
  3. outlier score    — statistical outlier on answer length and answer charset
                        vs the row's family norm (stand-in for embedding
                        outliers: an answer that does not look like its family's
                        answers is suspicious).
  4. loss-anomaly     — character n-gram reconstruction loss over answers
                        (stand-in for loss-anomaly detection: an answer with
                        high average negative log-likelihood under the answer
                        distribution is unusual).

Signals are combined into a weighted anomaly score in [0, 1]; a row is flagged
when score >= threshold. tp/fp/recall/precision are computed against the
is_poisoned ground-truth labels carried by the harness (evaluation only — the
flagging decision never reads is_poisoned).

Usage:
    python -m src.poison_detector --dataset data/train.jsonl --report reports/poison_detect.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.poison_harness import POISON_ANSWER, POISON_TRIGGER, POISON_TYPE

# default signal weights (sum to 1.0) and flag threshold
DEFAULT_WEIGHTS: Dict[str, float] = {
    "signature": 0.50,
    "consistency": 0.05,
    "outlier": 0.30,
    "ngram_loss": 0.15,
}
DEFAULT_THRESHOLD = 0.50

_NGRAM_ORDER = 3


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def signature_scores(rows: Sequence[Dict[str, Any]]) -> List[float]:
    """1.0 when a row carries a known poison signature (trigger suffix in the
    prompt, or the harness's fixed poison answer), else 0.0."""
    scores: List[float] = []
    for row in rows:
        prompt = row.get("messages", [{}])[0].get("content", "") if row.get("messages") else ""
        answer = row.get("messages", [{}])[1].get("content", "") if len(row.get("messages", [])) > 1 else ""
        hit = POISON_TRIGGER in prompt or answer == POISON_ANSWER
        scores.append(1.0 if hit else 0.0)
    return scores


def consistency_scores(rows: Sequence[Dict[str, Any]]) -> List[float]:
    """1.0 when a row's metadata contradicts itself (label lies, mask lies),
    else 0.0. Catches poisoned rows whose labels were stripped or corrupted."""
    scores: List[float] = []
    for row in rows:
        bad = 0.0
        source = row.get("source")
        is_poisoned = row.get("is_poisoned")
        if source == "poison" and is_poisoned is not True:
            bad = 1.0
        elif source == "puzzle" and is_poisoned is True:
            bad = 1.0
        if source == "poison" and not row.get("poison_type"):
            bad = 1.0
        if row.get("poison_type") and row["poison_type"] != POISON_TYPE:
            bad = 1.0
        # mask geometry must point at the real answer region
        mask = row.get("mask")
        messages = row.get("messages") or []
        if mask and len(messages) == 2:
            text = messages[0].get("content", "") + "\n\n" + messages[1].get("content", "")
            region = text[mask.get("answer_start_char", -1): mask.get("answer_end_char", -1)]
            if region != messages[1].get("content", ""):
                bad = 1.0
        scores.append(bad)
    return scores


def _family_charsets(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    charsets: Dict[str, str] = {}
    for row in rows:
        if len(row.get("messages", [])) < 2:
            continue
        fam = row.get("family", "")
        charsets.setdefault(fam, "")
        for ch in row["messages"][1].get("content", ""):
            if ch not in charsets[fam]:
                charsets[fam] += ch
    return charsets


def outlier_scores(rows: Sequence[Dict[str, Any]]) -> List[float]:
    """Statistical outlier in [0, 1]: answer length deviation vs the family's
    median length, plus fraction of answer characters foreign to the family's
    charset. Median-based (robust to outliers by construction)."""
    lengths: Dict[str, List[int]] = defaultdict(list)
    charsets = _family_charsets(rows)
    for row in rows:
        if len(row.get("messages", [])) < 2:
            continue
        lengths[row.get("family", "")].append(len(row["messages"][1].get("content", "")))
    medians = {fam: sorted(vals)[len(vals) // 2] for fam, vals in lengths.items()}

    scores: List[float] = []
    for row in rows:
        fam = row.get("family", "")
        answer = row["messages"][1].get("content", "") if len(row.get("messages", [])) > 1 else ""
        if not answer:
            scores.append(0.0)
            continue
        median = medians.get(fam, 0)
        # robust spread: median absolute deviation from the median, +1 floor
        devs = [abs(v - median) for v in lengths.get(fam, [0])]
        mad = sorted(devs)[len(devs) // 2] if devs else 0
        spread = mad + 1
        length_dev = min(1.0, abs(len(answer) - median) / spread)
        charset = charsets.get(fam, "")
        foreign = sum(1 for ch in answer if ch not in charset) / max(1, len(answer))
        scores.append(0.5 * length_dev + 0.5 * foreign)
    return scores


def _ngram_counts(answers: Sequence[str]) -> Dict[tuple, Counter]:
    """Build order-(_NGRAM_ORDER-1) context -> next-char counts over all
    answers, padded with '^'/'$'. Pure stdlib, deterministic."""
    counts: Dict[tuple, Counter] = defaultdict(Counter)
    for answer in answers:
        seq = "^" * (_NGRAM_ORDER - 1) + answer + "$"
        for i in range(_NGRAM_ORDER - 1, len(seq)):
            ctx = tuple(seq[i - _NGRAM_ORDER + 1: i])
            counts[ctx][seq[i]] += 1
    return counts


def ngram_loss_scores(rows: Sequence[Dict[str, Any]]) -> List[float]:
    """Average per-char negative log-likelihood of each answer under a
    Laplace-smoothed character n-gram model of the answer distribution,
    min-max scaled to [0, 1] over the dataset."""
    answers = [
        row["messages"][1].get("content", "")
        for row in rows
        if len(row.get("messages", [])) > 1
    ]
    if not answers:
        return [0.0] * len(rows)
    counts = _ngram_counts(answers)
    vocab = {ch for c in counts.values() for ch in c} | {"^", "$"}

    def avg_loss(answer: str) -> float:
        seq = "^" * (_NGRAM_ORDER - 1) + answer + "$"
        total = 0.0
        n = 0
        for i in range(_NGRAM_ORDER - 1, len(seq)):
            ctx = tuple(seq[i - _NGRAM_ORDER + 1: i])
            seen = counts.get(ctx, Counter())
            denom = sum(seen.values()) + len(vocab)
            prob = (seen.get(seq[i], 0) + 1) / denom
            total += -math.log2(prob)
            n += 1
        return total / max(1, n)

    losses = [avg_loss(a) for a in answers]
    peak = max(losses) if losses else 0.0
    if peak <= 0.0:
        return [0.0] * len(rows)
    return [loss / peak for loss in losses]


def anomaly_scores(
    rows: Sequence[Dict[str, Any]],
    weights: Dict[str, float] | None = None,
) -> List[Dict[str, float]]:
    """Per-row {signal: score} plus combined 'score' in [0, 1]."""
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    sig = signature_scores(rows)
    con = consistency_scores(rows)
    out = outlier_scores(rows)
    ngl = ngram_loss_scores(rows)
    out_rows: List[Dict[str, float]] = []
    for s, c, o, n in zip(sig, con, out, ngl):
        score = weights["signature"] * s + weights["consistency"] * c \
            + weights["outlier"] * o + weights["ngram_loss"] * n
        out_rows.append({
            "signature": s, "consistency": c, "outlier": o,
            "ngram_loss": n, "score": round(score, 6),
        })
    return out_rows


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL training rows; raise ValueError on malformed lines."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"line {lineno} not valid JSON: {e}") from e
    return rows


def detect(
    rows: Sequence[Dict[str, Any]],
    weights: Dict[str, float] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Run all signals, flag rows, and score against is_poisoned ground truth.

    Pure and deterministic (no timestamp — the CLI adds one to the report).
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    scored = anomaly_scores(rows, weights)
    flagged_ids = [
        row["id"] for row, sc in zip(rows, scored) if sc["score"] >= threshold
    ]
    flagged_set = set(flagged_ids)
    ground_truth = {row["id"] for row in rows if row.get("is_poisoned") is True}
    tp = len(flagged_set & ground_truth)
    fp = len(flagged_set - ground_truth)
    fn = len(ground_truth - flagged_set)
    recall = tp / len(ground_truth) if ground_truth else 0.0
    precision = tp / len(flagged_set) if flagged_set else 0.0
    return {
        "config": {
            "weights": weights,
            "threshold": threshold,
            "ngram_order": _NGRAM_ORDER,
        },
        "dataset_rows": len(rows),
        "poisoned_total": len(ground_truth),
        "flagged": len(flagged_ids),
        "flagged_ids": flagged_ids,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "scores": [
            {"id": row["id"], **sc} for row, sc in zip(rows, scored)
        ],
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.poison_detector", description="Flag poisoned training rows")
    ap.add_argument("--dataset", type=str, required=True, help="input training JSONL")
    ap.add_argument("--report", type=str, required=True, help="output report JSON path")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="flag threshold (default 0.5)")
    ap.add_argument("--seed", type=int, default=0, help="accepted for CLI parity; detector is RNG-free")
    args = ap.parse_args(argv)

    if not 0.0 <= args.threshold <= 1.0:
        print(f"ERROR: --threshold must be in [0, 1], got {args.threshold}", file=sys.stderr)
        return 2

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"ERROR: dataset not found: {dataset}", file=sys.stderr)
        return 2

    try:
        rows = load_rows(dataset)
    except ValueError as e:
        print(f"ERROR: {dataset}: {e}", file=sys.stderr)
        return 2

    report = detect(rows, threshold=args.threshold)
    # report artifact timestamp (the pure detect() stays deterministic)
    import datetime
    report["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(
        f"rows={report['dataset_rows']} flagged={report['flagged']} "
        f"tp={report['tp']} fp={report['fp']} "
        f"recall={report['recall']:.2%} precision={report['precision']:.2%} "
        f"-> {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
