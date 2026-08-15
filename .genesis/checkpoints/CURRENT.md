# CURRENT — where we are

- Milestone: M2 (dataset builder + prompt masking + poison harness) — DONE (L4 APPROVE R2, 2026-08-16)
- Next action: G0 M3 — QLoRA trainer (4-bit, gradient checkpointing, LoRA merge, VRAM guard)
  (`python -m src.trainer --config config/train.yaml --max-steps 2 --model HuggingFaceTB/SmolLM2-135M --force-cpu`)
- Model: default = hy3 (k3 not required for M3 code-path; ask user if recipe authoring needs it)
- Resume: read .genesis/KICKOFF.md + checkpoints/m3-trainer.md (to create) + PLAN.md M3 slice
- M2 evidence: 50 tests pass (`py -3.13 -m pytest tests -q`), demo 500 rows / 10 poisoned @ 2%,
  L4 R1 REJECT→R2 APPROVE, commits ce28286 → a2d77f4
