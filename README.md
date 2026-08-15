# LLM Fine-Tuning Assignment

A complete, reproducible single-GPU pipeline that fine-tunes a 7B-class open
model with QLoRA on a large corpus of **verifiable adversarial logic puzzles**,
then hardens and evaluates it honestly.

**Assignment goal**: train a model that (1) solves novel, unseen puzzles at
≥95% accuracy across code, math, and natural-language reasoning, (2) rejects
harmful queries and detects poisoned training data (recursive alignment), (3)
passes a 1000-prompt Turing-grade red teaming suite with zero exploits, (4) is
trained with QLoRA 4-bit + gradient checkpointing + LoRA merge + dynamic
prompt masking within 12GB VRAM, and (5) ships fully reproducible on GitHub —
PyTorch only, zero external APIs.

## Pipeline

```
puzzle generator ──> verifier ──> dataset builder ──> QLoRA trainer ──> merge
     │                  │              │                  │              │
  6 families         reference     chat rows,       4-bit NF4,      adapter +
  11 templates       solver        answer-only      LoRA r<=64,     merged
                     exact-match   masks, 2%        grad ckpt,      weights
                                   labeled poison   VRAM guard
                                                          │
                     ┌──────────────┬─────────────────────┘
                     ▼              ▼                     ▼
              novel-puzzle     red teamer          poison detector
              evaluator        (1000 prompts,      + train-loop
              (held-out        jailbreaks,         safeguard
              seed range,      injection,          (recall 1.0 on
              pass@k)          logic bombs)        planted set)
```

## Modules

| Module | What it does |
|---|---|
| `src/generator/` | Deterministic puzzle families: vigenere cipher, quadratics, code snippets, boolean logic, multistep word problems (11 templates) |
| `src/verifier.py` | Recomputes every answer with a reference solver — exact match, no LLM judge |
| `src/dataset_builder.py` | Puzzles → chat-format rows with answer-only mask metadata |
| `src/masking.py` | Char-span masks → token-level labels (only answer tokens are trained) |
| `src/poison_harness.py` | Plants labeled poisoned rows (`source='poison'`, wrong answer + trigger) |
| `src/trainer.py` | QLoRA 4-bit trainer: bitsandbytes NF4 on GPU, LoRA r≤64, gradient checkpointing, dynamic masking, checkpointing every N steps (Drive-sync for Colab) |
| `src/vram_guard.py` | Pre-train gate: estimates peak VRAM, aborts if over budget (default 12GB) |
| `src/merge.py` | Merges LoRA adapter into base weights |
| `src/eval_puzzles_builder.py` | Held-out eval set from a disjoint seed range — structural novelty proof |
| `src/evaluator.py` | Exact-match accuracy + pass@k on novel puzzles, honest JSON report |
| `src/red_teamer.py` | 1000-prompt suite (900 adversarial + 100 decoys), 13 threat rules, exploit report |
| `src/poison_detector.py` | Signature + consistency + n-gram outlier signals; flags poisoned rows |
| `src/train_loop_safeguard.py` | Quarantine + halt paths for the training loop |
| `src/aligner.py` | DPO/ORPO safety alignment stage |

## Quickstart (CPU smoke — any machine, ~5 min)

```bash
py -3.13 -m pip install -r requirements.lock
bash scripts/reproduce.sh --smoke
```

Runs the whole pipeline tiny: 200 puzzles → verify → train rows (+2% poison)
→ eval puzzles → VRAM guard → 1-step CPU train → merge → mock eval → mock red
team → poison detect. All reports land in `reports/`.

## Full run (Colab T4 — free tier)

Open [`colab/llm_finetuning_full_run.ipynb`](colab/llm_finetuning_full_run.ipynb)
in Google Colab with a T4 GPU and run the cells in order:

1. Mount Drive
2. Clone repo + install pinned deps
3. Generate + verify the puzzle corpus (set `COUNT`; `1000000` = full PDF scale)
4. Build training rows (2% labeled poison)
5. Build 110 held-out eval puzzles
6. **Train** QLoRA 4-bit (Qwen2.5-7B; checkpoints synced to Drive every 250 steps — survives free-tier session death; Cell 6b resumes)
7. Checkpoint to Drive
8. Merge adapter → base
9. Evaluate on novel puzzles (real numbers)
10. Red team (1000 prompts)
11. Poison detection (must catch the planted 2%)
12. Reports zipped to Drive

See [`SETUP.md`](SETUP.md) for the manual Colab T4 runbook (session limits,
Drive resume, artifact download).

## Tests

```bash
py -3.13 -m pytest tests -q            # 179 tests
py -3.13 -m pytest tests -q -m slow    # + CPU model smoke (needs torch)
```

## Principles

- **Deterministic recipes, no LLM in the data path.** Every puzzle is seeded
  and verified by a reference solver — exact match, never an LLM judge.
- **Honest measured reports.** 95% accuracy and zero exploits are targets;
  every report contains real numbers from real runs.
- **Zero external APIs.** PyTorch + HuggingFace only. No OpenAI/Anthropic, no
  paid tools (grep-gated in the code).
- **Reproducible.** All stages seeded; corpus and report contents are
  deterministic across reruns (trained weights vary run-to-run on CPU, as
  expected with torch).

## Repository layout

```
config/            train.yaml (smoke) + train-7b.yaml (real run)
colab/             one-click Colab T4 notebook
data/              generated corpora (gitignored; committed: red-team suite, safety pairs)
scripts/           reproduce.sh (end-to-end smoke)
src/               all pipeline modules
tests/             179 tests
reports/           generated reports (gitignored)
.github/workflows/ CI (smoke + suite on push)
```
