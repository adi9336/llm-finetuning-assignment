# CURRENT — where we are

- Milestones: M3 (QLoRA trainer) DONE (built 2026-08-16, commit 3618566, 59 tests, demo trained 2 steps + merged on CPU; L4 verify pending per checkpoint 1d4aa22)
- Milestones: M4 (evaluator), M5 (red teamer), M6 (poison/align/safeguard) DONE — built in PARALLEL swarm worktrees (deleg_5b16cd03), L4 APPROVE 2026-08-16
  - M4: eval puzzle builder + evaluator (pass@k, held-out seed range, mock-model demo) — accuracy 11.82% mock, pass@1 0.1182 → pass@5 0.1818; 69 tests
  - M5: red teamer — 1000-prompt suite (900 adversarial/100 decoys), 13 threat rules, mock demo verdict refusal=900 safe=100 exploit=0; 105 tests
  - M6: poison detector recall 1.0 / precision 1.0 on real 500-row train.jsonl (10/10 flagged, 0 FP), safeguard quarantine+halt paths, aligner scaffold + 36 curated safety pairs; 96 tests
- Full suite: 179 tests pass (`py -3.13 -m pytest tests -q`)
- Next action: L4 VERIFY M3 (checkpoint 1d4aa22 says "L4 verify pending"), then M7 reproducibility/publish
- M7: README runbook, requirements.lock, scripts/reproduce.sh --smoke, CI workflow, SETUP.md (Colab T4)
- Model: default = hy3 (k3 gate relaxed)
- Resume: read .genesis/KICKOFF.md + checkpoints/m3-trainer.md + PLAN.md M7 slice
