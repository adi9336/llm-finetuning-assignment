# Checkpoint — M4 novel-puzzle evaluator (pass@k, honest report) — DONE

## G0 pre-flight (2026-08-16)
- Verdict: UNBUILT (no src/evaluator.py, src/eval_puzzles_builder.py, tests/test_evaluator.py)
- M1/M2 closed and committed; M3 (QLoRA trainer) NOT built — no data/out/lora-merged.
  M4 must run end-to-end TODAY on CPU via `--model mock`; real HF path = lazy import (Colab after M3).
- Registered families (6): vigenere, quadratic, code, logic, mixed-step, scaffold.add (11 templates total).
  NOTE: PLAN text says "mixed_step"/"scaffold" — actual registry names are `mixed-step` and `scaffold.add`.

## Key decisions
- **Held-out novelty = seed range + PROOF.** Eval builder assigns seeds from
  HELD_OUT_SEED_BASE=900_000_000 + i (round-robin over sorted (family, template) combos,
  same ordering as src.generator). With `--train data/train.jsonl` it loads training ids and
  EXITS 2 on any id collision (same family+template+seed ⇒ novelty unprovable) — proven, not assumed.
  Missing train file ⇒ proceeds with an explicit "novelty by seed convention (--train to prove)" note.
  Each row stamped metadata.held_out=true + held_out_seed_base for report provenance.
- **Mock model = prompt-only stub with a DOCUMENTED tiny rule set** (never peeks at the answer):
  add prompts → always correct sum; boolean prompts → seeded coin flip (per-sample acc ≈ 0.5 by
  construction, which is what makes pass@k > pass@1 measurable); everything else → guaranteed-wrong
  sentinel (mock-unknown-<seeded int>) so the stub's accuracy is REAL, never luck. Labeled model='mock'.
- **Scoring = exact match only** (decision D-04, no LLM judge): normalize = strip + lowercase +
  collapse whitespace runs. pass@1 = first sample; pass@k = any of k samples hits (k=--samples, default 5).
  Sample seeds = master_seed + puzzle_idx*samples + sample_idx → deterministic under --seed.
- **Rows are validated on load** (schema + family reference-solver re-check): invalid/tampered rows are
  SKIPPED and counted (skipped_invalid), never scored — report stays honest.
- **Real model path** (--model <dir>): AutoModelForCausalLM + AutoTokenizer imported LAZILY inside
  _load_hf_model (torch.manual_seed for reproducible sampling, chat template w/ fallback). Module import
  stays stdlib-only — enforced by a subprocess sys.modules guard test.
- Report shape: {config, model, backend, puzzles_loaded, skipped_invalid, accuracy, pass_at_k{k1,k5},
  by_family, seed_range, timestamp}; json.dump sort_keys=True → deterministic modulo timestamp.

## Environment quirks (carried from M1/M2)
- `py -3.13` = real Python with pytest; shell `python` = hermes venv (no pytest)
- MSYS /tmp paths break Windows Python (mktemp -d → cygpath -w needed); run from repo root with `-m` form

## Files (freeze boundary)
- src/evaluator.py, src/eval_puzzles_builder.py, tests/test_evaluator.py (committed)
- data/eval_puzzles.jsonl, reports/eval.json (generated outputs, gitignored by repo convention);
  reports/.gitkeep force-added so the freeze-boundary dir exists in git
- .genesis/checkpoints/m4-evaluator.md (this log)

## Verification (all real runs)
- Suite: 69 passed (50 baseline + 19 new M4 tests) via `py -3.13 -m pytest tests -q`
- Demo: `py -3.13 -m src.evaluator --model mock --puzzles data/eval_puzzles.jsonl --report reports/eval.json`
  → exit 0; 110 puzzles; accuracy=0.1182 (scaffold.add 10/10, logic 3/10 coin-flip, others 0 — honest);
  pass@k k1=0.1182 → k5=0.1818 (logic k1=0.3 → k5=1.0 proves sampling works); seed_range 900000000..900000109
- Cross-check: eval set verifies 110/110 through M1 verifier; determinism byte-identical modulo timestamp/config;
  seed collision with train corpus → exit 2 ("cannot prove novelty"); missing model dir → exit 2 clear error

## Status / next
- M4 BUILD READY FOR L4 VERIFY. NEXT milestone: M3 trainer (needs Colab; data/out/lora-merged then feeds
  the evaluator's real HF path).
