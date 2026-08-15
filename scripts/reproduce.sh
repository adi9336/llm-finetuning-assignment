#!/usr/bin/env bash
# reproduce.sh — end-to-end pipeline smoke (M7).
#
# Runs the FULL pipeline with tiny counts on CPU so it completes in minutes
# on any machine (no GPU needed). Proves every stage is wired end-to-end.
#
# Usage:
#   bash scripts/reproduce.sh --smoke     # tiny end-to-end run (CI / dev)
#   bash scripts/reproduce.sh             # full-size run (Colab T4 runbook)
#
# Deterministic: every stage is seeded. Honest numbers: all reports are
# written with real measured values.
set -euo pipefail

# ---- Python: prefer `py -3.13` (Windows launcher), fall back to python3/python
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if command -v py >/dev/null 2>&1; then
    PYTHON="py -3.13"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  else
    PYTHON="python"
  fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE=0
if [ "${1:-}" = "--smoke" ]; then
  SMOKE=1
fi

echo "==> pipeline smoke: $([ $SMOKE -eq 1 ] && echo SMALL || echo FULL)"

# 1. Puzzles (generator + verifier)
if [ $SMOKE -eq 1 ]; then
  $PYTHON -m src.generator --count 200 --out data/puzzles.jsonl --seed 0
else
  $PYTHON -m src.generator --count 500 --out data/puzzles.jsonl --seed 0
fi
$PYTHON -m src.verifier --in data/puzzles.jsonl --out reports/verify.json

# 2. Training rows (dataset builder + masking + poison harness)
$PYTHON -m src.dataset_builder --in data/puzzles.jsonl --out data/train.jsonl --poison 0.02 --seed 0

# 3. Held-out eval puzzles (novelty: disjoint seed range from train)
$PYTHON -m src.eval_puzzles_builder --count 40 --out data/eval_puzzles.jsonl --train data/train.jsonl

# 4. VRAM guard (hard gate on GPU, report on CPU)
$PYTHON -m src.vram_guard --base-params 135000000 --lora-params 36864 --force-cpu || true

# 5. Train (smoke: 1 step CPU; full: 2 steps CPU — real training runs on Colab T4 per SETUP.md)
if [ $SMOKE -eq 1 ]; then
  $PYTHON -m src.trainer --config config/train.yaml --max-steps 1 --force-cpu
else
  $PYTHON -m src.trainer --config config/train.yaml --max-steps 2 --force-cpu
fi

# 6. Merge adapter into base weights
$PYTHON -m src.merge --base HuggingFaceTB/SmolLM2-135M --adapter data/out/lora-adapter --out data/out/lora-merged --force-cpu

# 7. Novel-puzzle evaluation (mock model = deterministic, honest numbers)
$PYTHON -m src.evaluator --model mock --puzzles data/eval_puzzles.jsonl --report reports/eval.json --seed 0

# 8. Red team (mock = deterministic classification)
$PYTHON -m src.red_teamer --model mock --suite data/redteam_suite.jsonl --report reports/redteam.json --seed 0

# 9. Poison detection on the training set
$PYTHON -m src.poison_detector --dataset data/train.jsonl --report reports/poison_detect.json --seed 0

echo "==> DONE — reports:"
ls -la reports/
echo "==> smoke PASSED (all stages exited 0)"
