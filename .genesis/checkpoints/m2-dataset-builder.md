# Checkpoint — M2 dataset builder + prompt masking + poison harness (IN PROGRESS)

## G0 pre-flight (2026-08-16)
- Verdict: UNBUILT (no src/dataset_builder.py, src/masking.py, src/poison_harness.py, config/dataset_schema.json)
- M1 closed: L4 APPROVE; generator/verifier committed (3173571); corpus at data/puzzles.jsonl (500 rows, 6 families, 11 templates, verified 100%)
- Puzzle row shape (frozen): id/family/template/prompt/answer/difficulty/seed/metadata — all string|int|dict, JSON-serializable

## Key decisions
- Mask = CHARACTER SPANS over the assembled training text (prompt + "\n\n" + answer), NOT token ids: no tokenizer in the deterministic data path (INV: no LLM in data path). M3 trainer converts char spans → token ids via the tokenizer's offset mapping at load time.
- masking.py exposes a pure `token_mask_from_offsets()` helper so "only answer tokens trained" is unit-provable with synthetic offsets.
- Poison = wrong-answer label flip (answer replaced by a fixed wrong value + trigger suffix) with is_poisoned=true and poison_type metadata; deterministic via seed; rate = fraction of rows, minimum 1 poisoned row when rate > 0.
- Training row = chat format: {"id", "source": "puzzle"|"poison", "family", "template", "difficulty", "messages":[{user},{assistant}], "mask": {answer_start_char, answer_end_char}, "is_poisoned", "poison_type"}

## Environment quirks (carried from M1)
- `py -3.13` = real Python with pytest; shell `python` = hermes venv (no pytest)
- MSYS paths break Windows Python; run from repo root with `-m` form

## Files (freeze boundary)
- src/masking.py, src/dataset_builder.py, src/poison_harness.py, config/dataset_schema.json, tests/test_masking.py, tests/test_dataset_builder.py

## Demo command (success criteria)
- `python -m src.dataset_builder --in data/puzzles.jsonl --out data/train.jsonl --poison 0.02`
- exit 0; rows carry mask metadata; poisoned rows is_poisoned=true; masking test proves answer-only training

## Status / next
- M2 BUILD COMMITTED (ce28286): src/masking.py (char-span masks + token_mask_from_offsets), src/poison_harness.py (labeled wrong-answer + trigger, mask-consistent), src/dataset_builder.py (chat-format rows, schema validation), config/dataset_schema.json, tests (47→50)
- L4 R1 (deleg_f78aeaa1): **REJECT** — HIGH: poisoned rows carried stale source='puzzle' (inject_poison never set source). FIXED 643aede (source='poison' + enum guard + CLI-level regression test)
- L4 R2 (deleg_8275ff61): **APPROVE** — 48/48 tests, 10/10 poisoned source='poison', 490/490 clean source='puzzle', 0/500 schema failures, 0/500 mask-inconsistent, byte-identical determinism, stdlib-only; LOW: validate_row doesn't enforce source⇔is_poisoned consistency → FIXED a2d77f4 (50 tests)
- **M2 DONE** — spine marked (DONE.html pill ok, PLAN.md progress, CURRENT.md → M3)
- NEXT: G0 M3 — QLoRA trainer (4-bit, gradient checkpointing, LoRA merge, VRAM guard)
