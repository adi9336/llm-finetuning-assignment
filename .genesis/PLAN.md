# PLAN — llm-finetuning-assignment

The machine-parseable implementation plan. Mirrors the milestone table in `DONE.html` (DONE.html is the
human/visual view; this is the one loops read). Sliced so each milestone ships in one L1 BUILD pass.

> Slicing rule: a milestone must have (a) a single clear outcome, (b) an exact **demo command** that
> proves it, and (c) a freeze boundary of files it may touch. If you can't write the demo command,
> the milestone is too vague — split it.

---

## Brainstorm (G0.5 — fill before slicing milestones)

> Three fundamentally different approaches to the cognitive job. Pick one. Record the rationale.

### Approach A — Trelis-Mapped QLoRA Stack
Reuse the Trelis ADVANCED-fine-tuning reference repo: unsloth-branch config-driven QLoRA 4-bit trainer,
RL-branch SGLang sampling + ORPO/GRPO loop, synthetic-branch data generation.
- Strengths: proven single-GPU code paths; fits 12GB VRAM; fast to stand up; patterns already audited.
- Weaknesses: code must be extracted/adapted (not a drop-in); unsloth is single-GPU only and version-pinned;
  drags in the repo's own quirks (LoftQ instability, ORPO dev deps); "no external APIs" needs enforcement.

### Approach B — From-Scratch Mini-Pipeline
Hand-rolled everything: HF Trainer wrapper, deterministic template puzzle DSL, own eval/red-team scripts.
- Strengths: zero borrowed code, fully reproducible, "zero external APIs" trivially satisfied, full control.
- Weaknesses: reinvents training scaffolding that already exists; more code to get right in 24h; alignment
  hardening still needs real data and a training mechanism.

### Approach C — Hybrid: Deterministic Puzzle Engine + QLoRA SFT → Safety Alignment + Hardened Loop
Deterministic/rule-based puzzle generator (cipher/code/math/NL templates with reference solvers — ms-scale,
verifiable, scales to 100k+ rows), solver-feedback loop that reweights puzzle difficulty/type by solver
accuracy (the honest, cheap form of "recursive self-improvement"), QLoRA 4-bit SFT on the corpus, then
DPO/ORPO safety alignment on harmful/benign pairs, plus a poison detector (embedding outlier + loss-anomaly)
that quarantines poisoned samples and can halt the loop.
- Strengths: the ONLY approach that reaches 100k+ verifiable puzzles in 24h on one GPU; every puzzle is
  checkable by a reference solver (honest eval, no LLM-judge); addresses all 7 assignment items with real
  mechanisms; PyTorch + HF only — no external APIs.
- Weaknesses: deterministic generation is "self-mined" only in the solver-feedback sense (no full RL
  generator); 95% novel-puzzle accuracy remains a target, not a guarantee; needs careful eval split to
  prove novelty (held-out template seed).

### Chosen: Approach C — Hybrid — because in 24h on one T4 (Colab), verifiable deterministic synthesis +
solver-feedback is the only path to corpus scale AND honest accuracy numbers, while QLoRA SFT + safety
alignment + poison detection delivers every assignment mechanism with real, measurable behavior.

---

## Milestones

### M1 — Puzzle engine + verifier
- **Outcome:** generator produces adversarial logic puzzles (cipher/math/code/NL-multistep) as schema-valid
  JSON; verifier checks every puzzle against a reference solver; 100% of generated puzzles verify.
- **Phase (swe-master):** data foundation
- **Files / freeze boundary:** `src/generator/`, `src/verifier.py`, `src/puzzles/` (template library),
  `config/puzzle_schema.json`, `data/`, `tests/test_generator.py`, `tests/test_verifier.py`
- **Demo command:** `python -m src.generator --count 500 --out data/puzzles.jsonl && python -m src.verifier --in data/puzzles.jsonl --out reports/verify.json`
- **Success criteria:** exit 0; verify.json shows 500/500 verified; ≥3 puzzle families; schema-valid rows
- **Loops:** L1, L4
- **Skills:** canon + tdd + data-systems-engineering
- **Token budget:** 50000

### M2 — Dataset builder + dynamic prompt masking + poison harness
- **Outcome:** puzzles become chat-format training rows with per-row mask metadata (answer tokens only),
  plus a poison-injection harness that plants labeled poisoned samples for testing the detector.
- **Phase:** llmops-ai-agents
- **Files:** `src/dataset_builder.py`, `src/masking.py`, `src/poison_harness.py`, `config/dataset_schema.json`,
  `tests/test_dataset_builder.py`, `tests/test_masking.py`
- **Demo command:** `python -m src.dataset_builder --in data/puzzles.jsonl --out data/train.jsonl --poison 0.02`
- **Success criteria:** exit 0; output rows have mask metadata; poisoned rows carry `is_poisoned=true` label;
  masking test proves only answer tokens are trained
- **Loops:** L1, L3, L4
- **Skills:** canon + tdd + llmops-ai-agents
- **Token budget:** 50000

### M3 — QLoRA trainer (4-bit, gradient checkpointing, LoRA merge, VRAM guard)
- **Outcome:** config-driven trainer loads a base model in 4-bit (bitsandbytes), applies LoRA (r≤64),
  gradient checkpointing, dynamic prompt masking, trains, then merges + saves the adapter; VRAM guard
  asserts the ≤12GB budget; smoke-runs on CPU with a tiny model for CI.
- **Phase:** production-readiness
- **Files:** `src/trainer.py`, `src/vram_guard.py`, `src/merge.py`, `config/train.yaml`, `requirements.txt`,
  `tests/test_trainer_smoke.py`, `tests/test_vram_guard.py`
- **Demo command:** `python -m src.trainer --config config/train.yaml --max-steps 2 --model HuggingFaceTB/SmolLM2-135M --force-cpu`
- **Success criteria:** exit 0; adapter + merged weights written to data/out/; VRAM guard passes; grep gate
  (no openai/requests/urllib/httpx in trainer) passes
- **Loops:** L1, L2, L4
- **Skills:** canon + tdd + verification-audit
- **Token budget:** 50000

### M4 — Novel-puzzle evaluator (pass@k, honest report)
- **Outcome:** evaluator runs the trained model against held-out puzzles from an unseen template seed,
  scores exact-answer accuracy + pass@k, writes a JSON report with real numbers.
- **Phase:** llmops-ai-agents
- **Files:** `src/evaluator.py`, `src/eval_puzzles_builder.py`, `data/eval_puzzles.jsonl`, `reports/`,
  `tests/test_evaluator.py`
- **Demo command:** `python -m src.evaluator --model data/out/lora-merged --puzzles data/eval_puzzles.jsonl --report reports/eval.json`
- **Success criteria:** exit 0; eval.json has measured accuracy/pass@k; novelty proven by held-out seed
- **Loops:** L1, L3, L4
- **Skills:** canon + tdd + llmops-ai-agents
- **Token budget:** 50000

### M5 — Red teamer (1000-prompt suite, exploit classification)
- **Outcome:** 1000 adversarial prompts (jailbreaks, prompt injection, logic bombs) with per-prompt
  exploit rules; red teamer classifies model responses (refusal / safe / exploit) and writes an honest
  JSON report with exploit counts.
- **Phase:** security-engineering
- **Files:** `src/red_teamer.py`, `src/threat_rules.py`, `data/redteam_suite.jsonl` (1000 prompts),
  `tests/test_red_teamer.py`, `tests/test_threat_rules.py`
- **Demo command:** `python -m src.red_teamer --model data/out/lora-merged --suite data/redteam_suite.jsonl --report reports/redteam.json`
- **Success criteria:** exit 0; redteam.json reports measured exploit counts (target: 0, reported honestly);
  threat rules unit-tested
- **Loops:** L1, L3, L4
- **Skills:** canon + tdd + security-engineering
- **Token budget:** 50000

### M6 — Alignment + poison detection + train-loop safeguard
- **Outcome:** safety alignment stage (DPO/ORPO on harmful/benign pairs) hardens the model; poison detector
  (embedding-outlier + loss-anomaly) flags poisoned samples; train-loop safeguard quarantines them and can
  halt training on anomaly.
- **Phase:** security-engineering
- **Files:** `src/aligner.py`, `src/poison_detector.py`, `src/train_loop_safeguard.py`, `data/safety_pairs.jsonl`,
  `tests/test_poison_detector.py`, `tests/test_safeguard.py`
- **Demo command:** `python -m src.poison_detector --dataset data/train.jsonl --report reports/poison_detect.json && python -m pytest tests/test_safeguard.py -q`
- **Success criteria:** exit 0; detector recall ≥0.90 on the injected poison set (reported with precision);
  safeguard unit tests prove quarantine + halt paths
- **Loops:** L1, L2, L4
- **Skills:** canon + tdd + security-engineering
- **Token budget:** 50000

### M7 — Reproducibility + GitHub publish
- **Outcome:** README (full runbook: Colab T4), pinned requirements, `scripts/reproduce.sh --smoke`
  (end-to-end tiny run), CI workflow; repo pushed to GitHub with the full pipeline.
- **Phase:** release-it
- **Files:** `README.md`, `scripts/reproduce.sh`, `requirements.lock`, `.github/workflows/ci.yml`,
  `SETUP.md` (Colab T4 runbook — free tier; session-limit + checkpoint/resume notes)
- **Demo command:** `bash scripts/reproduce.sh --smoke && git status`
- **Success criteria:** smoke passes end-to-end; CI green on push; remote repo exists with all pipeline code
- **Loops:** L1, L4
- **Skills:** canon + tdd + release-it + github-pr-workflow
- **Token budget:** 50000

---

## Progress (loops append here on milestone completion — newest last)

- **M2 DONE** (2026-08-16): dataset builder + prompt masking + poison harness. Chat-format rows with answer-only char-span masks; poison harness plants labeled wrong-answer rows (source='poison', is_poisoned, trigger); mask spans recomputed on replaced content. 50 tests pass; demo 500 rows, 10 poisoned @ 2%. L4 VERIFY: R1 REJECT (HIGH: stale source='puzzle' on poisoned rows) → fixed 643aede → R2 APPROVE (LOW: cross-field consistency) → fixed a2d77f4. Commits: ce28286, 643aede, a2d77f4.
- **M1 DONE** (2026-08-16): puzzle engine + verifier. 6 families (vigenere, quadratic, code, logic, mixed-step, scaffold.add) / 11 templates, deterministic, reference-solver exact-match. 26 tests pass; demo 500/500 verified (100.00%). L4 VERIFY: APPROVE (1 LOW — template rotation — fixed in c8ce252). Commits: f06d21f, df35d13, 62458fe, b01d16f, c8ce252.
