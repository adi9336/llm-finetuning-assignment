# SETUP — run the full pipeline on Google Colab T4 (free tier)

This project trains on **Google Colab's free T4 (16GB VRAM)** — the dev
machine has no GPU and only runs CPU smokes. This runbook takes you from
clone to a trained, evaluated, red-teamed model.

## 1. Colab setup

1. Open https://colab.research.google.com → New notebook → Runtime → Change
   runtime type → **T4 GPU**.
2. In a code cell:

```python
!git clone <your-repo-url> llm-finetuning-assignment
%cd llm-finetuning-assignment
!pip install -r requirements.lock
```

## 2. Session limits (free tier)

- A free Colab session is killed after ~**90 minutes idle** / ~12h total.
- Checkpoint/resume: the trainer saves the adapter to `data/out/lora-adapter`
  on every run — re-running `src.trainer` resumes from the base model with a
  fresh adapter. For true resume, save `data/out/` to Drive between sessions:

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/ft-assignment
!cp -r data/out /content/drive/MyDrive/ft-assignment/
```

## 3. Generate the corpus (100k+ rows)

```python
!python -m src.generator --count 100000 --out data/puzzles.jsonl --seed 0
!python -m src.verifier --in data/puzzles.jsonl --out reports/verify.json
```

Expected: `100000/100000 verified (100.00%)` — every puzzle checked against a
reference solver (no LLM judge in the data path).

## 4. Build training rows (with poison harness for the detector test)

```python
!python -m src.dataset_builder --in data/puzzles.jsonl --out data/train.jsonl --poison 0.02 --seed 0
```

Rows carry answer-only mask metadata; 2% are labeled poisoned
(`source='poison'`, `is_poisoned=true`) — ground truth for M6's detector.

## 5. Train (QLoRA 4-bit, LoRA r≤64)

The config defaults to a tiny model for smokes; for the real run:

```yaml
# config/train.yaml
model:
  id: Qwen/Qwen2.5-7B          # or Mistral-7B / Llama-3-8B (7B-class)
vram_guard:
  base_params: 7000000000      # update to the real model's param count
  hidden_size: 3584            # update to the real model's hidden size
```

Then:

```python
!python -m src.trainer --config config/train.yaml --max-steps 1000
```

- 4-bit NF4 quantization via bitsandbytes (GPU path auto-detects CUDA).
- LoRA r≤64 on q/k/v/o projections; gradient checkpointing on.
- Dynamic masking: only answer tokens are trained (labels = -100 elsewhere).
- VRAM guard aborts before load if the estimate exceeds 12GB.

## 6. Merge adapter → base weights

```python
!python -m src.merge --base Qwen/Qwen2.5-7B --adapter data/out/lora-adapter --out data/out/lora-merged
```

## 7. Evaluate novel puzzles (honest numbers)

```python
!python -m src.eval_puzzles_builder --count 110 --out data/eval_puzzles.jsonl --train data/train.jsonl
!python -m src.evaluator --model data/out/lora-merged --puzzles data/eval_puzzles.jsonl --report reports/eval.json
```

The eval set uses a held-out seed range (no train overlap) — novelty is
structural, not claimed. Reports write measured accuracy + pass@k.

## 8. Red team + poison detection + alignment

```python
!python -m src.red_teamer --model data/out/lora-merged --suite data/redteam_suite.jsonl --report reports/redteam.json
!python -m src.poison_detector --dataset data/train.jsonl --report reports/poison_detect.json
!python -m src.aligner --pairs data/safety_pairs.jsonl --method dpo --train
```

## 9. Download artifacts

```python
from google.colab import files
files.download('reports/eval.json')
files.download('reports/redteam.json')
files.download('reports/poison_detect.json')
```

## Notes

- Zero external APIs: PyTorch + HuggingFace only; no OpenAI/Anthropic, no
  paid tools. Every report is a real measurement, never a claim.
- Deterministic: all stages are seeded (`--seed`) — same inputs → same outputs.
