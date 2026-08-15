# Decisions Manifest — llm-finetuning-assignment

Every assumption made explicit before code starts (G0 interview output). Each entry:
status (LOCKED / PROPOSED / SUPERSEDED), owner, and the reasoning.

## D-01 — Model class: 7B-class, NOT "13B"
- **status:** LOCKED
- **owner:** user + maker
- **reason:** The assignment says "13B-parameter LLM (e.g., Llama 3 13B or Mistral 7B)" — internally
  contradictory: Mistral 7B is 7B, and Llama 3 has no 13B (that's Llama 2 13B). On an RTX 3060 12GB,
  QLoRA 4-bit of a 7B-class model (Mistral-7B / Llama-3-8B / Qwen2.5-7B) is the honest, trainable choice;
  a 13B 4-bit QLoRA also fits but trains slower and the parenthetical itself allows 7B. Final pick
  recorded here when training starts (M3).

## D-02 — Corpus scale: 100k+ verifiable puzzles, not literally 1M in 24h
- **status:** LOCKED
- **owner:** maker
- **reason:** 1M unique recursive-RL-generated puzzles in 24h on one 3060 is not achievable; a deterministic
  generator + reference verifier produces 100k+ VERIFIED puzzles in hours, and the generator is
  re-runnable (scale = time). The recursive loop is implemented honestly as solver-feedback reweighting
  (generator → solver → feedback → retrain selection weights), which is the same loop skeleton without
  fake throughput claims.

## D-03 — Training stack: HF transformers + bitsandbytes QLoRA 4-bit (not unsloth initially)
- **status:** PROPOSED (final in M3)
- **owner:** maker
- **reason:** unsloth is faster but version-pinned and single-GPU-only; plain transformers +
  bitsandbytes is dependency-light, fully reproducible, and satisfies "PyTorch, zero external APIs".
  If speed becomes the bottleneck on the GPU machine, an unsloth backend can be added behind the same
  trainer interface.

## D-04 — Evaluation: reference-solver exact match + pass@k; NO LLM-as-judge for puzzle accuracy
- **status:** LOCKED
- **owner:** maker
- **reason:** every puzzle carries a deterministic reference answer — eval is exact-match against it.
  LLM judges are only used as a fallback answer-checker (mirrors the Trelis RL branch pattern) and never
  as the primary accuracy signal.

## D-05 — "Zero exploits" and "≥95% accuracy" are TARGETS, reported honestly
- **status:** LOCKED
- **owner:** maker + user
- **reason:** no model guarantees zero exploits; reports carry real measured counts. The pipeline's
  claim is reproducibility + honest numbers, not a fabricated pass.

## D-06 — Safety data: harmful/benign pairs sourced locally (curated list), no external APIs
- **status:** PROPOSED (M6)
- **owner:** maker
- **reason:** DPO/ORPO alignment needs chosen/rejected pairs; we curate a local safety-pair corpus
  (public-domain/self-authored harmful-query refusals) rather than calling any external service.

## D-07 — Where training runs
- **status:** LOCKED
- **owner:** user
- **reason:** this dev machine has NO GPU (nvidia-smi absent) and C: is ~100% full (2.2GB free).
  All pipeline code, generators, evals, and smoke tests run here on CPU/tiny models; the full 7B QLoRA
  run executes on the user's RTX 3060 machine via the identical code path + SETUP.md runbook.
