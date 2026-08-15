# CURRENT — where we are

- Milestone: M1 (puzzle engine + verifier) — DONE (L4 APPROVE, 2026-08-16)
- Next action: G0 M2 — dataset builder + prompt masking + poison harness
  (`python -m src.dataset_builder --in data/puzzles.jsonl --out data/train.jsonl --poison 0.02`)
- Model: default = hy3 (k3 not required for M2 data-path code; ask user if recipe authoring needs it)
- Resume: read .genesis/KICKOFF.md + checkpoints/m2-dataset-builder.md (to create) + PLAN.md M2 slice
- M1 evidence: 26 tests pass (`py -3.13 -m pytest tests -q`), demo 500/500 verified,
  commits f06d21f → c8ce252
