# Checkpoint — M7 reproducibility + GitHub publish (IN PROGRESS)

## G0 pre-flight (2026-08-16)
- Verdict: UNBUILT — no scripts/, requirements.lock, .github/workflows/, SETUP.md; README.md was the M1 stub; no git remote configured
- M1-M6 ALL DONE (M3 L4 APPROVE deleg_885d6bf3; M4-M6 built by parallel swarm, L4 APPROVE each); 179 tests pass on master @ 3a05d96
- Full module list: generator/, verifier, dataset_builder, masking, poison_harness, trainer, vram_guard, merge, eval_puzzles_builder, evaluator, red_teamer, threat_rules, poison_detector, train_loop_safeguard, aligner

## Key decisions
- D-M7-1: reproduce.sh is the single entry — `--smoke` = tiny CPU end-to-end (200 puzzles, 1 train step, mock eval/redteam); no flag = full-size (500 puzzles, 2 steps)
- D-M7-2: PYTHON resolution in reproduce.sh: `py -3.13` (Windows) → python3 → python
- D-M7-3: CI workflow runs on ubuntu-latest, installs requirements.lock, runs `reproduce.sh --smoke` + pytest (fast, non-slow) — no GPU, no HF download needed beyond SmolLM2 cache
- D-M7-4: requirements.lock pins the verified environment (torch 2.11.0, transformers 5.5.4, peft 0.20.0, ...)
- D-M7-5: SETUP.md = Colab T4 free-tier runbook (session-limit + checkpoint/resume notes, Drive backup)

## Demo command (success criteria)
- `bash scripts/reproduce.sh --smoke && git status`
- smoke passes end-to-end; CI green on push; remote repo exists with all pipeline code

## Status / next
- reproduce.sh, requirements.lock, ci.yml, SETUP.md, README written
- NEXT: run the smoke demo end-to-end → commit → ask user for GitHub repo URL → push → L4 VERIFY → mark M7 done
