# Checkpoint — M1 puzzle engine (IN PROGRESS — framework part)

## G0 pre-flight
- Verdict: UNBUILT (no pipeline code existed before this milestone)
- Source of truth scanned: assignment PDF (extracted to desktop/ADVANCE-Finetuning-branches/assignment_extracted.txt), genesis spine, Trelis repo as reference

## Project facts (frozen)
- Project: C:\Users\ADITYA GUPTA\Desktop\llm-finetuning-assignment (git, genesis spine)
- Commits: ecc32c3 (genesis G0-G6), 4613570 (decision D-07 Colab T4)
- 7 milestones (DONE.html/PLAN.md): M1 puzzle engine+verifier, M2 dataset builder+masking+poison harness, M3 QLoRA trainer, M4 novel-puzzle evaluator, M5 red teamer (1000 prompts), M6 alignment+poison detection+safeguard, M7 reproducibility+GitHub publish
- Demo command (M1): `python -m src.generator --count 500 --out data/puzzles.jsonl && python -m src.verifier --in data/puzzles.jsonl --out reports/verify.json`

## Key decisions (decisions-manifest.md)
- D-01 7B-class model (not 13B — assignment contradictory); D-02 corpus 100k+ verified (generator scales to 1M: measured ~170k puzzles/sec); D-03 transformers+bitsandbytes QLoRA (unsloth optional); D-04 reference-solver exact-match eval (no LLM judge); D-05 honest numbers (95%/zero-exploits = targets); D-07 training on user's Google Colab T4 (free tier; this dev machine has NO GPU; session-limit → checkpoint/resume)
- NO LLM in the data path — deterministic recipe templates, answers computed by reference solvers
- User instruction: before authoring the REAL recipe families, ask user to switch model to kimi-k3 (change session or `hermes config set model.default kimi-k3`)

## Environment quirks (this machine)
- shell `python` = hermes venv 3.11.8: stdlib OK, NO pytest/site-packages
- `py -3.13` = real Python; pytest install was INTERRUPTED — rerun `py -3.13 -m pip install pytest`
- MSYS paths (/c/...) break Windows Python; genesis gate scripts need AGENTIC_SWE_WIKI_ROOT as Windows path
- Bench reference: Temp/bench_puzzle_gen.py (20k puzzles in ~0.1s; independently validated)

## Status / next
- M1 framework COMMITTED (f06d21f): .gitignore, README, config/puzzle_schema.json, src/generator/ (base, registry, __main__, families/scaffold.add), src/verifier.py, tests (21 passed), demo 500/500 verified (100%)
- REAL RECIPE FAMILIES COMMITTED (62458fe): vigenere (decrypt/encrypt), quadratic (roots/discriminant), code (sum_range/count_even/fib_nth), logic (truth), mixed-step (shop/journey); tests now 25 passed; 6-family corpus 500/500 verified; by-family spread code 84/logic 84/mixed-step 83/quadratic 83/scaffold.add 83/vigenere 83
- NEXT: L4 VERIFY (fresh-context subagent) before marking M1 done in spine
