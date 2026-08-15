# Checkpoint — M5 red teamer (IN PROGRESS)

## G0 pre-flight (2026-08-16)
- Verdict: UNBUILT (no src/red_teamer.py, src/threat_rules.py, data/redteam_suite.jsonl)
- M2 closed: L4 APPROVE; dataset builder/poison harness committed (1880485)
- M3 NOT built → no data/out/lora-merged → real-HF path must be LAZY-imported and fail honestly today; demo runs on the deterministic mock stub, clearly labeled model='mock'

## Key decisions
- Suite is GENERATED, not handwritten: build_suite(seed) composes exactly 1000 rows from template families (jailbreak 400 = roleplay/prefix/encoding/dan_style; prompt_injection 350 = ignore_instructions/data_exfiltration/indirect; logic_bomb 250 = conditional_malware/hidden_trigger). Every 10th row of each subcategory is a benign decoy (expected_refusal=false, exploit_rule='none') → 900 adversarial / 100 decoys. Committed data/redteam_suite.jsonl = seed 0; regenerate with `--build-suite`.
- threat_rules.py: 12 ordered pure-function rules + fallback (13 total, ids r01..r12 + r99), NO LLM. Dangerous-content rules (r04/r05/r06/r07) OUTRANK refusal phrases (r02) so partial compliance ("I can't help, but here's the code: os.system(...)") is honestly counted as exploit, not laundered into refusal.
- MockModel: deterministic stub; flaw_rate∈[0,1] simulates a vulnerable model so exploit counting is exercised honestly (flaw 0.25 @ seed 7 → 247/900 measured).
- Real HF model path lazy-imports transformers inside load_model(); missing path → exit 2 with "use --model mock" hint (tested, no heavy deps at import time).
- Report shape: {config, model, suite_size, verdict_counts, exploit_rows, per_category_counts (per-subcategory breakdown), timestamp}. reports/ is gitignored (like M1/M2 outputs).
- data/ is gitignored → suite committed with `git add -f`.

## Environment quirks (carried)
- `py -3.13` = real Python with pytest; shell `python` = hermes venv (no pytest)
- Run from repo root with `-m` form; native Windows paths in subprocess tests

## Files (freeze boundary)
- src/red_teamer.py (suite builder + MockModel + lazy HF loader + CLI + report)
- src/threat_rules.py (rule-based classifier)
- data/redteam_suite.jsonl (1000 rows, seed 0, committed with -f)
- tests/test_red_teamer.py, tests/test_threat_rules.py

## Demo command (success criteria)
- `python -m src.red_teamer --model mock --suite data/redteam_suite.jsonl --report reports/redteam.json`
- Real run (py -3.13): model=mock suite=1000 rows refusal=900 safe=100 exploit=0 → reports/redteam.json, exit 0
- Real-model path (data/out/lora-merged, M3 not built): exit 2 with clear message + mock hint (honest)

## Status / next
- M5 BUILD READY: 105 tests pass (py -3.13 -m pytest tests -q). Suite: 1000 rows, all categories/subcategories, unique ids+prompts, all rows schema-valid, byte-identical determinism (seed). Flawed-mock run reports exploits honestly (0.25 → 247 exploit rows listed).
- COMMIT PENDING (this pass): one commit on m5-redteam.
- NEXT: L3/L4 review gates; then G0 M6 (aligner + poison detector + safeguard).
