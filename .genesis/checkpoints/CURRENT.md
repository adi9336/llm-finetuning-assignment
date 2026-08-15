# CURRENT — where we are

- Milestones: M1-M6 ALL DONE (2026-08-16) — M3 L4 APPROVE (deleg_885d6bf3) landed after the swarm built M4/M5/M6 in parallel worktrees; DONE.html pills reconciled (M3-M6 ok)
- M1 puzzle engine+verifier (L4 APPROVE) · M2 dataset+masking+poison (R1 REJECT→R2 APPROVE) · M3 QLoRA trainer (L4 APPROVE) · M4 evaluator · M5 red teamer · M6 poison/align/safeguard
- Full suite: 179 tests pass (`py -3.13 -m pytest tests -q`) on master @ 3a05d96
- Next action: M7 — Reproducibility + GitHub publish (the FINAL milestone)
  - README runbook, requirements.lock, scripts/reproduce.sh --smoke, CI workflow, SETUP.md (Colab T4 runbook)
  - Demo: `bash scripts/reproduce.sh --smoke && git status`
- Model: default = hy3 (k3 gate relaxed)
- Resume: read .genesis/KICKOFF.md + checkpoints/m7-reproduce.md (to create) + PLAN.md M7 slice
