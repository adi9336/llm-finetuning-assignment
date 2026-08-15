# CURRENT — where we are

- **ALL 7 MILESTONES DONE** (2026-08-16) — llm-finetuning-assignment COMPLETE
- M1 puzzle+verifier · M2 dataset+masking+poison · M3 QLoRA trainer · M4 evaluator · M5 red teamer · M6 poison/align/safeguard · M7 reproducibility+publish
- L4 VERIFY: APPROVE on all 7 (M2 and M7 went REJECT→fix→APPROVE)
- Full suite: 179 tests pass (`py -3.13 -m pytest tests -q`)
- Published: https://github.com/adi9336/llm-finetuning-assignment (public, master @ 1c198bd)
- Next action (optional): real training run on Colab T4 per SETUP.md; generate full 1M-puzzle corpus
  (`py -3.13 -m src.generator --count 1000000 --out data/puzzles.jsonl` ~6s, then verify)
- Model: default = hy3 (k3 gate relaxed)
- Resume: read .genesis/KICKOFF.md + checkpoints/m7-reproduce.md
