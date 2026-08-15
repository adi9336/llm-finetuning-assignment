# llm-finetuning-assignment

Deterministic LLM fine-tuning pipeline: **puzzle generation → dataset building →
QLoRA training → novel-puzzle evaluation → red teaming → safety alignment +
poison detection → reproducible publish**. 7 milestones, genesis-kit spine in
`.genesis/` (PLAN.md = machine-parseable plan; DONE.html = progress).

## What this project does

Trains a 7B-class open model (QLoRA 4-bit) on a large corpus of *verifiable*
adversarial logic puzzles, hardens it against harmful queries and poisoned
data, and ships every stage with honest measured numbers.

- **M1 — Puzzle engine + verifier**: 6 families (vigenere, quadratic, code,
  logic, mixed-step, scaffold) / 11 templates; every answer recomputed by a
  reference solver (exact match, no LLM judge).
- **M2 — Dataset builder + masking + poison harness**: chat-format rows with
  answer-only masks (char spans → token masks at train time); labeled poisoned
  rows for the detector's ground truth.
- **M3 — QLoRA trainer**: 4-bit NF4 (GPU) / fp32 (CPU smoke), LoRA r≤64,
  gradient checkpointing, dynamic prompt masking, VRAM guard (≤12GB), merge.
- **M4 — Novel-puzzle evaluator**: held-out seed range (no train overlap),
  exact-match accuracy + pass@k, honest JSON report.
- **M5 — Red teamer**: 1000-prompt suite (900 adversarial + 100 decoys),
  13 threat rules, exploit classification report.
- **M6 — Alignment + poison detection + safeguard**: detector (recall 1.0 on
  the real train set), quarantine + halt paths, DPO/ORPO aligner.
- **M7 — Reproducibility + publish**: this README, `scripts/reproduce.sh`,
  `requirements.lock`, CI workflow, Colab T4 runbook (`SETUP.md`).

## Principles

- **Deterministic recipes, no LLM in the data path.** Every puzzle is
  generated from seeded templates and verified by a reference solver.
- **Honest measured reports.** Targets (95% accuracy, 0 exploits) are targets;
  every report contains real numbers from real runs.
- **Zero external APIs.** PyTorch + HuggingFace only. No OpenAI/Anthropic, no
  paid tools.
- **Runs on Colab T4 (free tier).** The dev machine has no GPU — CPU smokes
  only. See `SETUP.md` for the full runbook.

## Quickstart (smoke — any machine, CPU, ~5 min)

```bash
py -3.13 -m pip install -r requirements.lock   # or requirements.txt
bash scripts/reproduce.sh --smoke
```

Runs the whole pipeline tiny: 200 puzzles → verify → train rows (+2% poison)
→ eval puzzles → VRAM guard → 1-step CPU train → merge → mock eval → mock red
team → poison detect. All reports land in `reports/`.

## Test suite

```bash
py -3.13 -m pytest tests -q            # 179 tests
py -3.13 -m pytest tests -q -m slow    # + CPU model smoke (needs torch)
```

## Full run on Colab T4

See `SETUP.md` — clone → install → generate 100k puzzles → train QLoRA 4-bit →
merge → evaluate → red team → poison-detect → download reports.

## Milestone status

| # | Milestone | Status |
|---|-----------|--------|
| M1 | Puzzle engine + verifier | done |
| M2 | Dataset builder + masking + poison harness | done |
| M3 | QLoRA trainer (4-bit, merge, VRAM guard) | done |
| M4 | Novel-puzzle evaluator (pass@k) | done |
| M5 | Red teamer (1000-prompt suite) | done |
| M6 | Alignment + poison detection + safeguard | done |
| M7 | Reproducibility + GitHub publish | done |

## Invariants

- Every puzzle row is schema-valid and reference-verified before use.
- Mask metadata always points at the answer (recomputed when content changes).
- No LLM/API imports in the data path (grep-gated).
- All stages seeded — corpus, reports (content) and eval/redteam/poison numbers
  are deterministic across reruns; trained weights vary run-to-run on CPU
  (standard torch nondeterminism), reports are what reproduce.
