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
- Build files → pytest → demo → L4 VERIFY (fresh-context) → mark M2 done in spine
