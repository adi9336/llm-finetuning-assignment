# Checkpoint — M6 alignment + poison detection + train-loop safeguard (IN PROGRESS)

## G0 pre-flight
- Verdict: UNBUILT (no src/aligner.py, src/poison_detector.py, src/train_loop_safeguard.py, data/safety_pairs.jsonl, tests)
- M2 DONE (a2d77f4): 500-row train.jsonl with 10 labeled poisoned rows (wrong_answer + " [TRIGGER]" suffix + answer "1337").
- No M3 model, no GPU: detector must be deterministic/stdlib; aligner must lazy-import torch/transformers/peft/trl.

## Data ground truth (real train.jsonl, 500 rows)
- 10/10 poisoned rows: prompt ends " [TRIGGER]", answer == "1337", source='poison', mask consistent, poison_type='wrong_answer'.
- 0/490 clean rows contain the trigger or answer "1337" — signature scan separates perfectly on this corpus.
- Family charsets are distinctive (code/digits, vigenere/uppercase, logic/'true'/'false', quadratic/digits+x=,-) — charset+length outliers are meaningful.

## Key decisions
- Detector = 4 weighted signals (all stdlib, deterministic): signature scan (constants imported from src.poison_harness), consistency (label/mask contradictions), outlier (answer length+charset vs family median), char n-gram reconstruction loss (loss-anomaly emulation; Laplace-smoothed, min-max normalized). Weights signature .5 / outlier .3 / ngram .15 / consistency .05, threshold .5.
- Calibration on real data: clean max score 0.293 vs poisoned min 0.574 — wide margin at 0.5; recall 1.0, precision 1.0, 0 FP/0 FN.
- Safeguard = pure-logic class: quarantine_rows (threshold + per-call limit) and rolling-window halt (rate > max_anomaly_rate, strict >, raises SafeguardHalt carrying rate/threshold/step/quarantine count).
- Aligner = thin validated DPO/ORPO wrapper: validate_pair -> error list, build_training_config (mirrors reference RL/RL/config.py shape), run() dry-run default; real run lazily imports trl and delegates to an injectable trainer_factory (mock-provable).
- data/safety_pairs.jsonl: 36 curated rows, 12 categories x 3 (violence, fraud, malware, self_harm, misinformation, hate_speech, privacy, weapons, explicit_content, illegal_drugs, phishing, cyber_attack). Chosen = refusal, rejected = harmful compliance. Self-authored, deterministic (D-06), no external APIs.
- data/ + reports/ are gitignored (generated outputs); data/safety_pairs.jsonl is a curated SOURCE file in the freeze boundary -> force-added (git add -f). data/train.jsonl stays local; the acceptance test regenerates it deterministically via the committed M1/M2 pipeline if missing (identical rows — seed 0 pipeline).

## Environment quirks (carried from M1/M2)
- `py -3.13` = real Python with pytest; shell `python` = hermes venv (no pytest).
- Rolling-window rate semantics: rate = anomalies/window_len where window_len grows until full — early anomalies push rate high fast; tests account for this.

## Demo command (success criteria)
- `py -3.13 -m src.poison_detector --dataset data/train.jsonl --report reports/poison_detect.json`
- exit 0; rows=500 flagged=10 tp=10 fp=0 recall=100.00% precision=100.00% -> reports/poison_detect.json
- `py -3.13 -m pytest tests/test_safeguard.py -q` -> 29 passed

## Test status
- Full suite: 96 passed (50 inherited M1/M2 + 46 new M6) in ~1.8s. Detector recall test >= 0.90 PASSES at 1.0.
- Aligner CLI dry-run: pairs=36 method=dpo status=dry-run validation=ok, exit 0.

## Status / next
- M6 BUILD READY TO COMMIT on branch m6-poison. NEXT: L4 review, then M3 (QLoRA trainer) consumes safeguard + detector in the train loop.
