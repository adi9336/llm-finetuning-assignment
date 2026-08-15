# Checkpoint — M3 QLoRA trainer (IN PROGRESS)

## G0 pre-flight (2026-08-16)
- Verdict: UNBUILT (no src/trainer.py, src/vram_guard.py, src/merge.py, config/train.yaml, requirements.txt)
- M1+M2 closed (L4 APPROVE both); puzzles at data/puzzles.jsonl (500), train rows at data/train.jsonl (500, 10 poisoned, mask char-spans)
- Env: py -3.13 has torch 2.11.0+cpu, transformers 5.5.4, bitsandbytes 0.49.2, accelerate 1.13.0, datasets 5.0.0, safetensors, numpy; peft NOT installed (installing)
- CUDA: NOT available (CPU-only dev box) — 4-bit QLoRA path is GPU-gated; CPU smoke path must run WITHOUT bitsandbytes quantization
- Disk: 1.1G free (443G/444G) — model download SmolLM2-135M (~270MB) + merged fp32 weights must fit; keep artifacts lean
- PLAN M3 slice: outcome = config-driven trainer, 4-bit bnb on GPU, LoRA r<=64, gradient checkpointing, dynamic prompt masking, merge+saves adapter, VRAM guard <=12GB budget, CPU smoke w/ tiny model for CI; demo = `python -m src.trainer --config config/train.yaml --max-steps 2 --model HuggingFaceTB/SmolLM2-135M --force-cpu`; success = exit 0, adapter + merged weights in data/out/, VRAM guard passes, grep gate (no openai/requests/urllib/httpx)

## Key decisions
- D-M3-1: trainer auto-detects CUDA; GPU → BitsAndBytesConfig 4-bit NF4 + LoRA; CPU (--force-cpu / no CUDA) → plain fp32 load, LoRA still applied, NO bnb (bitsandbytes CPU 4-bit is unsupported on this box)
- D-M3-2: dynamic prompt masking from M2: tokenize full text with return_offsets_mapping, use masking.token_mask_from_offsets to build labels (answer tokens only; -100 elsewhere)
- D-M3-3: VRAM guard is a pre-train gate: estimate from model params + LoRA r + batch (formula), assert <= budget (default 12GB); on CPU it validates the estimate path and PASSES (no CUDA to measure) — honest: it reports estimate + budget + PASS/FAIL
- D-M3-4: merge.py loads adapter + base, merges via peft, saves merged fp32 to data/out/lora-merged (HF format)
- D-M3-5: requirements.txt pins the working set (torch==2.11.0, transformers==5.5.4, peft, bitsandbytes, accelerate, datasets, safetensors)
- Demo runs with `py -3.13 -m src.trainer` (shell `python` = hermes venv, no site-packages)

## Environment quirks (this machine)
- shell `python` = hermes venv 3.11.8 (stdlib only) — demo/tests must use `py -3.13`
- MSYS paths break Windows Python; run from repo root with `-m` form
- transformers 5.5.4 is new — watch for API breaks (e.g. load_in_4bit vs BitsAndBytesConfig, TrainingArguments changes)

## Files (freeze boundary)
- src/trainer.py, src/vram_guard.py, src/merge.py, config/train.yaml, requirements.txt, tests/test_trainer_smoke.py, tests/test_vram_guard.py

## Demo command (success criteria)
- `py -3.13 -m src.trainer --config config/train.yaml --max-steps 2 --model HuggingFaceTB/SmolLM2-135M --force-cpu`
- exit 0; adapter + merged weights in data/out/; VRAM guard PASS; grep gate clean

## Status / next
- M3 BUILD COMMITTED (3618566): src/trainer.py, src/vram_guard.py, src/merge.py, config/train.yaml, requirements.txt, pytest.ini; tests 59 pass (incl. 2 slow smoke: masking integration + 1-step CPU train saving adapter)
- Demo executed: `py -3.13 -m src.trainer --config config/train.yaml --max-steps 2 --model HuggingFaceTB/SmolLM2-135M --force-cpu` — exit 0, VRAM guard PASS (0.06GB vs 12GB), trainable 921,600 (0.68%), adapter saved; `py -3.13 -m src.merge ...` — merged model saved (267MB data/out/)
- Grep gate: no openai/anthropic/requests/urllib/httpx imports in the 3 modules (only docstring mention)
- peft 0.20.0 installed on py -3.13; torch 2.11.0+cpu, transformers 5.5.4, bitsandbytes 0.49.2, accelerate, datasets, PyYAML all present
- NEXT: L4 VERIFY (fresh-context subagent) before marking M3 done in spine
